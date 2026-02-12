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

    args = ap.parse_args()

    res = run_pipeline(
        disease=args.disease,
        drug=args.drug,
    )

    if "error" in res:
        print("ERROR:", res["error"])
        return

    print("\n==== TRUST SCORE ====")
    print(res.get("trust_score"))
    print("Bad refs:", res.get("metrics", {}).get("bad_reference_nums"))
    print("Missing examples:", res.get("metrics", {}).get("missing_examples"))

    print("\n==== REPORT ====\n")
    print(res.get("report", ""))

    path = save_markdown_report(res, args.out)
    print(f"\nSaved report to: {path}")


if __name__ == "__main__":
    main()
