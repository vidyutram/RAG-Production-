import asyncio
from eval.pipeline.eval_pipeline import run_pipeline
from eval.pipeline.regression import load_eval_results, compare_runs, print_regression_report

GOLDEN_DATASET_PATH = "eval/golden_dataset.json"
TOLERANCE = 0.05


async def main():
    print("Running eval pipeline...")
    filename = await run_pipeline(GOLDEN_DATASET_PATH)
    print(f"\nResults saved to {filename}")


async def run_regression(baseline_path: str, candidate_path: str):
    baseline = load_eval_results(baseline_path)
    candidate = load_eval_results(candidate_path)
    report = compare_runs(baseline, candidate, TOLERANCE)
    print_regression_report(report)


if __name__ == "__main__":
    asyncio.run(main())