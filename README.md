# SHL Assessment Recommendation System

FastAPI implementation for the SHL AI Intern assignment. It ingests the SHL product catalog JSON, builds a lightweight retrieval index, and exposes the required health and recommendation endpoints.

## Run Locally

```bash
pip install -r requirements.txt
uvicorn app.main:app --reload
```

Open `http://localhost:8000` for the demo page.

## API

### `GET /health`

Returns service status.

### `POST /recommend`

Request:

```json
{
  "query": "I am hiring Java developers who can collaborate with business teams. The test should be completed in 40 minutes.",
  "max_recommendations": 10
}
```

### `POST /chat`

Request:

```json
{
  "message": "I need Java developer assessments under 40 minutes.",
  "history": [
    {"role": "user", "content": "I am hiring software engineers."}
  ],
  "max_recommendations": 10
}
```

Response includes a grounded natural-language answer, clarifying questions when the query is underspecified, recommendations, and a compact comparison table based on catalog evidence.

Response:

```json
{
  "recommended_assessments": [
    {
      "url": "https://www.shl.com/products/product-catalog/view/core-java-entry-level-new/",
      "adaptive_support": "No",
      "description": "Multi-choice test...",
      "duration": 16,
      "remote_support": "Yes",
      "test_type": ["Knowledge & Skills"]
    }
  ]
}
```

## Approach

- Fetches the SHL-provided catalog from `shl_product_catalog.json` and caches it under `data/`.
- Normalizes catalog fields including URL, description, duration, remote support, adaptive support, job levels, languages, and test type.
- Builds an in-memory TF-IDF index over names, descriptions, test types, job levels, and languages.
- Adds query understanding for time limits, seniority hints, and common hiring domains such as software, sales, and customer service.
- Supports direct job-description URLs by fetching page text before retrieval.

The implementation is dependency-light so it can deploy cleanly on Render, Railway, Hugging Face Spaces, or similar Python hosts.

## Evaluation Helper

```bash
python scripts/evaluate.py
```

This reports macro Recall@10, MRR, duration relevance, and groundedness. To smoke-test a deployed API:

```bash
python scripts/evaluate.py https://your-public-base-url
```
