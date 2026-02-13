import os
from datetime import datetime

def save_markdown_report(res: dict, out_dir: str) -> str:
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