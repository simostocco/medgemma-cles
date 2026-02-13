import argparse
import os
from datetime import datetime
from medgemma.pipeline.orchestrator import run_pipeline


def save_markdown_report(res: dict, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)

    disease = res["metadata"]["disease"].replace(" ", "_")
    drug = res["metadata"]["drug"].replace(" ", "_")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    filename = f"{drug}__{disease}__{timestamp}.md"
    path = os.path.join(out_dir, filename)

    trust_score = res.get("trust_score", 0.0)
    report = res.get("report", "")
    sources = res.get("sources", [])

    with open(path, "w", encoding="utf-8") as f:
        f.write(f"# MedGemma Research Report\n\n")
        f.write(f"**Drug:** {drug}\n\n")
        f.write(f"**Disease:** {disease}\n\n")
        f.write(f"**Trust Score:** {trust_score}%\n\n")
        f.write("---\n\n")
        f.write(report)
        f.write("\n\n---\n\n")
        f.write("## Sources\n\n")

        for s in sources:
            pmid = s.get("pmid")
            title = s.get("title", "No Title")
            sid = s.get("sid")
            if pmid:
                f.write(f"- {sid}: {title} — https://pubmed.ncbi.nlm.nih.gov/{pmid}/\n")

    return path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--drug", required=True)
    ap.add_argument("--disease", required=True)
    ap.add_argument("--out", default="reports")
    ap.add_argument("--agentic", action="store_true", help="Enable agentic self-repair loop")

    args = ap.parse_args()

    if args.agentic:
        from medgemma.generation.lmstudio_backend import generate_report_lmstudio
        from medgemma.agentic.repair_lmstudio import agentic_research_pipeline_lmstudio

        def llm_generate(prompt: str, max_tokens: int):
            return generate_report_lmstudio(prompt, max_tokens=max_tokens)

        res = agentic_research_pipeline_lmstudio(
            disease=args.disease,
            drug=args.drug,
            llm_generate=llm_generate,
            max_retries=3,
            target_coverage=90.0,
        )
    else:
        res = run_pipeline(disease=args.disease, drug=args.drug)

    if "error" in res:
        print("ERROR:", res["error"])
        return

    print("\n==== TRUST SCORE ====")
    print(res.get("trust_score"))
    print("Bad refs:", res.get("metrics", res.get("metrics_all", {})).get("bad_reference_nums"))
    print("Missing examples:", res.get("metrics", res.get("metrics_all", {})).get("missing_examples"))
    print("Agentic used:", res.get("agentic_used", False), "attempts:", res.get("agentic_attempts", 0))
    print("Rewritten to insufficiency:", res.get("n_rewritten_to_insufficient", 0))

    print("\n==== REPORT ====\n")
    print(res.get("report", ""))


if __name__ == "__main__":
    main()
