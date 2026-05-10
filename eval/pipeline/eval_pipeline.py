import os
import json
import asyncio
from datetime import datetime
from collections import defaultdict
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

THRESHOLDS = {
    "faithfulness": 0.7,
    "answer_relevancy": 0.7,
    "context_precision": 0.7,
    "context_recall": 0.7,
}


def load_golden_dataset(filepath: str) -> list[dict]:
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Golden dataset not found at path: {filepath}")
    with open(filepath, "r", encoding="utf-8") as file:
        return json.load(file)


async def evaluate_single(case: dict, llm: ChatOpenAI, embeddings: OpenAIEmbeddings) -> dict:
    dataset = Dataset.from_list([{
        "question": case["question"],
        "answer": case["answer"],
        "contexts": case["contexts"],
        "ground_truth": case["ground_truth"],
    }])

    results = await evaluate(
        dataset=dataset,
        metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
        embeddings=embeddings,
        llm=llm,
    )

    return {
        "id": case["id"],
        "question": case["question"],
        "category": case["category"],
        "scores": {
            "faithfulness": results["faithfulness"],
            "answer_relevancy": results["answer_relevancy"],
            "context_precision": results["context_precision"],
            "context_recall": results["context_recall"],
        }
    }


def aggregate_results(results: list[dict], thresholds: dict) -> dict:
    metrics = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]

    overall = {}
    for metric in metrics:
        scores = [r["scores"][metric] for r in results]
        overall[metric] = sum(scores) / len(scores)

    category_groups = defaultdict(list)
    for r in results:
        category_groups[r["category"]].append(r)

    by_category = {}
    for category, category_results in category_groups.items():
        by_category[category] = {}
        for metric in metrics:
            scores = [r["scores"][metric] for r in category_results]
            by_category[category][metric] = sum(scores) / len(scores)

    violations = []
    for metric, score in overall.items():
        if score < thresholds[metric]:
            violations.append(f"overall.{metric}: {score:.2f} is below threshold {thresholds[metric]}")

    for category, cat_scores in by_category.items():
        for metric, score in cat_scores.items():
            if score < thresholds[metric]:
                violations.append(f"by_category.{category}.{metric}: {score:.2f} is below threshold {thresholds[metric]}")

    return {
        "overall": overall,
        "by_category": by_category,
        "violations": violations,
    }


def persist_results(results: list[dict], summary: dict) -> str:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"eval/results/eval_results_{timestamp}.json"
    with open(filename, "w", encoding="utf-8") as f:
        json.dump({"results": results, "summary": summary}, f, indent=2)
    return filename


async def run_pipeline(filepath: str, batch_size: int = 10) -> str:
    llm = ChatOpenAI(model="gpt-4o-mini")
    embeddings = OpenAIEmbeddings()

    test_cases = load_golden_dataset(filepath)
    all_results = []

    for i in range(0, len(test_cases), batch_size):
        batch = test_cases[i: i + batch_size]
        batch_results = await asyncio.gather(
            *[evaluate_single(case, llm, embeddings) for case in batch]
        )
        all_results.extend(batch_results)
        print(f"Completed batch {i // batch_size + 1}")

    summary = aggregate_results(all_results, THRESHOLDS)
    filename = persist_results(all_results, summary)

    if summary["violations"]:
        print("VIOLATIONS DETECTED:")
        for v in summary["violations"]:
            print(f"  {v}")
    else:
        print("All metrics within threshold.")

    return filename