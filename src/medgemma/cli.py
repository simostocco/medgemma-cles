import os
import argparse

from medgemma.generation.model import load_txgemma_submit_safe
from medgemma.pipeline.orchestrator import run_pipeline

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--drug", required=True)
    ap.add_argument("--disease", required=True)
    ap.add_argument("--model_id", default="google/txgemma-9b-chat")
    args = ap.parse_args()

    token = os.getenv("HF_TOKEN")
    tokenizer, model, _ = load_txgemma_submit_safe(args.model_id, token=token)

    res = run_pipeline(
        disease=args.disease,
        drug=args.drug,
        tokenizer=tokenizer,
        model=model,
    )

    print("\n==== TRUST SCORE ====")
    print(res.get("trust_score"))
    print("\n==== REPORT ====")
    print(res.get("report", ""))

if __name__ == "__main__":
    main()
