# src/medgemma/cli.py
import argparse
from medgemma.pipeline.orchestrator import run_pipeline


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--drug", required=True)
    ap.add_argument("--disease", required=True)

    # Optional overrides for LM Studio
    ap.add_argument("--max_tokens", type=int, default=384)
    ap.add_argument("--n_papers", type=int, default=25)
    ap.add_argument("--max_snippets", type=int, default=10)
    ap.add_argument("--sort", type=str, default="relevance")

    args = ap.parse_args()

    res = run_pipeline(
        disease=args.disease,
        drug=args.drug,
        n_papers=args.n_papers,
        max_snippets=args.max_snippets,
        sort=args.sort,
    )

    if "error" in res:
        print("ERROR:", res["error"])
        return

    print("\n==== TRUST SCORE ====")
    print(res.get("trust_score"))

    print("\n==== REPORT ====\n")
    print(res.get("report", ""))


if __name__ == "__main__":
    main()