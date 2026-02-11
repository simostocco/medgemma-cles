import argparse
from medgemma.pipeline.orchestrator import run_pipeline

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--drug", required=True)
    parser.add_argument("--disease", required=True)

    args = parser.parse_args()

    result = run_pipeline(args.disease, args.drug)

    print(result["report"])

if __name__ == "__main__":
    main()