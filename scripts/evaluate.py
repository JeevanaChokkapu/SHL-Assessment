import json
import sys
from pathlib import Path
from urllib.request import Request, urlopen

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.recommender import SHLRecommender


TEST_CASES = [
    {
        "query": "I am hiring for Java developers who can collaborate effectively with business teams. Looking for assessments that can be completed in 40 minutes.",
        "expected": {
            "java-8-new",
            "core-java-entry-level-new",
            "core-java-advanced-level-new",
            "agile-software-development",
            "computer-science-new",
        },
        "max_duration": 40,
    },
    {
        "query": "I want to hire new graduates for a sales role in my company. The budget is about an hour for each test.",
        "expected": {
            "entry-level-sales-solution",
            "sales-representative-solution",
            "sales-support-specialist-solution",
            "technical-sales-associate-solution",
            "sales-and-service-phone-solution",
        },
        "max_duration": 60,
    },
    {
        "query": "Recommend assessments for customer service call center agents who need phone and email communication skills.",
        "expected": {
            "sales-and-service-phone-solution",
            "contact-center",
            "writex-email-writing-customer-service-new",
        },
        "max_duration": 60,
    },
]


def slug_from_url(url: str) -> str:
    return url.rstrip("/").split("/")[-1].lower()


def retrieval_metrics(recommender: SHLRecommender, k: int = 10) -> dict:
    recall_total = 0.0
    mrr_total = 0.0
    duration_total = 0.0
    grounded_total = 0.0
    details = []

    catalog_urls = {product.link for product in recommender.products}
    catalog_descriptions = {product.description for product in recommender.products}

    for case in TEST_CASES:
        products = recommender.recommend(case["query"], limit=k)
        slugs = [slug_from_url(product.link) for product in products]
        expected = case["expected"]

        matched_terms = {
            term
            for slug in slugs
            for term in expected
            if term in slug
        }
        hit_ranks = [
            index
            for index, slug in enumerate(slugs, start=1)
            if any(term in slug for term in expected)
        ]
        recall = len(matched_terms) / len(expected)
        mrr = 1 / hit_ranks[0] if hit_ranks else 0.0
        duration_ok = sum(
            1 for product in products if not product.duration or product.duration <= case["max_duration"]
        ) / max(1, len(products))
        grounded = sum(
            1
            for product in products
            if product.link in catalog_urls and product.description in catalog_descriptions
        ) / max(1, len(products))

        recall_total += recall
        mrr_total += mrr
        duration_total += duration_ok
        grounded_total += grounded
        details.append(
            {
                "query": case["query"],
                "recall_at_10": round(recall, 3),
                "mrr": round(mrr, 3),
                "duration_relevance": round(duration_ok, 3),
                "groundedness": round(grounded, 3),
                "top_results": [product.name for product in products[:5]],
            }
        )

    count = len(TEST_CASES)
    return {
        "macro_recall_at_10": round(recall_total / count, 3),
        "macro_mrr": round(mrr_total / count, 3),
        "duration_relevance": round(duration_total / count, 3),
        "groundedness": round(grounded_total / count, 3),
        "cases": details,
    }


def api_smoke(base_url: str) -> dict:
    health = urlopen(f"{base_url.rstrip('/')}/health", timeout=10).read().decode("utf-8")
    payload = json.dumps(
        {
            "message": TEST_CASES[0]["query"],
            "max_recommendations": 3,
        }
    ).encode("utf-8")
    request = Request(
        f"{base_url.rstrip('/')}/chat",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    chat = urlopen(request, timeout=20).read().decode("utf-8")
    return {
        "health": json.loads(health),
        "chat_keys": sorted(json.loads(chat).keys()),
    }


def main() -> None:
    recommender = SHLRecommender()
    recommender.load()
    report = retrieval_metrics(recommender)
    if len(sys.argv) > 1:
        report["api_smoke"] = api_smoke(sys.argv[1])
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
