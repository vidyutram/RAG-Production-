import os
import json


def load_eval_results(filepath: str) -> dict:
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Eval results not found at path: {filepath}")
    with open(filepath, "r", encoding="utf-8") as file:
        data = json.load(file)
        return data["summary"]


def compare_metric(baseline_score: float, candidate_score: float, tolerance: float) -> dict:
    delta = candidate_score - baseline_score

    if delta < 0 and abs(delta) > tolerance:
        status = "REGRESSION"
    elif delta > 0:
        status = "IMPROVEMENT"
    else:
        status = "STABLE"

    return {
        "baseline": round(baseline_score, 4),
        "candidate": round(candidate_score, 4),
        "delta": round(delta, 4),
        "status": status,
    }


def compare_runs(baseline: dict, candidate: dict, tolerance: float) -> dict:
    metrics = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]

    overall = {}
    for metric in metrics:
        overall[metric] = compare_metric(
            baseline["overall"][metric],
            candidate["overall"][metric],
            tolerance,
        )

    by_category = {}
    for cat in baseline["by_category"]:
        by_category[cat] = {}
        for metric in metrics:
            by_category[cat][metric] = compare_metric(
                baseline["by_category"][cat][metric],
                candidate["by_category"][cat][metric],
                tolerance,
            )

    regressions = []
    for metric, result in overall.items():
        if result["status"] == "REGRESSION":
            regressions.append({
                "category": "overall",
                "metric": metric,
                "result": result,
            })

    for cat, cat_results in by_category.items():
        for metric, result in cat_results.items():
            if result["status"] == "REGRESSION":
                regressions.append({
                    "category": cat,
                    "metric": metric,
                    "result": result,
                })

    return {
        "passed": len(regressions) == 0,
        "overall": overall,
        "by_category": by_category,
        "regressions": regressions,
    }


def print_regression_report(report: dict):
    print("REGRESSION TEST REPORT")
    status = "PASSED" if report["passed"] else "FAILED"
    print(f"Result: {status}")

    print("\nOVERALL SCORES")
    print("-" * 60)
    for metric, result in report["overall"].items():
        print(f"{metric:<25} baseline: {result['baseline']}  candidate: {result['candidate']}  delta: {result['delta']}  {result['status']}")

    if report["regressions"]:
        print("\nREGRESSIONS FOUND")
        print("-" * 60)
        for regression in report["regressions"]:
            print(f"category: {regression['category']}  metric: {regression['metric']}  delta: {regression['result']['delta']}")