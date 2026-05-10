from contextlib import asynccontextmanager
from typing import List

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse

from app.models import Assessment, ChatAssessment, ChatRequest, ChatResponse, RecommendRequest, RecommendationResponse
from app.recommender import SHLRecommender


recommender = SHLRecommender()


@asynccontextmanager
async def lifespan(app: FastAPI):
    recommender.load()
    yield


app = FastAPI(
    title="SHL Assessment Recommendation API",
    description="Retrieval-based recommender for SHL assessments.",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health")
def health_check():
    return {"status": "healthy" if recommender.ready else "unhealthy"}


@app.post("/recommend", response_model=RecommendationResponse)
def recommend(request: RecommendRequest):
    try:
        products = recommender.recommend(request.query, request.max_recommendations or 10)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Recommendation failed: {exc}") from exc

    assessments: List[Assessment] = [
        Assessment(
            url=product.link,
            adaptive_support=product.adaptive,
            description=product.description or product.name,
            duration=product.duration,
            remote_support=product.remote,
            test_type=product.test_type,
        )
        for product in products[: request.max_recommendations or 10]
    ]
    return RecommendationResponse(recommended_assessments=assessments)


def _assessment_from_product(product) -> Assessment:
    return Assessment(
        url=product.link,
        adaptive_support=product.adaptive,
        description=product.description or product.name,
        duration=product.duration,
        remote_support=product.remote,
        test_type=product.test_type,
    )


def _evidence_for(product) -> List[str]:
    evidence = [
        f"Catalog description: {product.description or product.name}",
        f"Duration: {product.duration or 'N/A'} minutes",
        f"Remote support: {product.remote}; adaptive support: {product.adaptive}",
        f"Test type: {', '.join(product.test_type)}",
    ]
    if product.job_levels:
        evidence.append(f"Job levels: {', '.join(product.job_levels[:5])}")
    return evidence


def _needs_clarification(text: str) -> List[str]:
    lowered = text.lower()
    questions: List[str] = []
    role_terms = ("developer", "engineer", "sales", "manager", "customer", "service", "analyst", "graduate", "java", "python", "call center")
    if not any(term in lowered for term in role_terms):
        questions.append("What role or job family are you hiring for?")
    if not any(term in lowered for term in ("minute", "hour", "timed", "duration", "budget")):
        questions.append("Do you have a maximum assessment duration?")
    if not any(term in lowered for term in ("entry", "graduate", "junior", "senior", "manager", "professional")):
        questions.append("What experience level should the assessment target?")
    return questions[:3]


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    message = request.message or request.query
    if not message:
        raise HTTPException(status_code=422, detail="message or query is required")

    context_parts = [
        item.content
        for item in request.history[-6:]
        if item.role.lower() in {"user", "assistant"} and item.content.strip()
    ]
    combined_query = " ".join(context_parts + [message])

    try:
        products = recommender.recommend(combined_query, request.max_recommendations or 10)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Chat recommendation failed: {exc}") from exc

    clarifying_questions = _needs_clarification(combined_query)
    chat_assessments = [
        ChatAssessment(
            **_assessment_from_product(product).model_dump(),
            name=product.name,
            evidence=_evidence_for(product),
        )
        for product in products[: request.max_recommendations or 10]
    ]

    comparison = [
        {
            "name": product.name,
            "url": product.link,
            "duration": product.duration,
            "remote_support": product.remote,
            "adaptive_support": product.adaptive,
            "why_recommended": product.description or product.name,
        }
        for product in products[: min(3, request.max_recommendations or 10)]
    ]
    if clarifying_questions:
        answer = (
            "I found provisional matches from the SHL catalog, but these details would improve the ranking: "
            + " ".join(clarifying_questions)
        )
    else:
        answer = (
            "I ranked these SHL assessments using catalog descriptions, duration, remote/adaptive flags, "
            "test type, and job-level evidence. If you change constraints such as time limit or seniority, "
            "send the new constraint with the prior history and the ranking will be refined."
        )

    return ChatResponse(
        answer=answer,
        clarifying_questions=clarifying_questions,
        recommended_assessments=chat_assessments,
        comparison=comparison,
        groundedness="All recommendations and comparison fields are copied or derived from the SHL catalog records returned by retrieval.",
    )


@app.get("/", response_class=HTMLResponse)
def demo_page():
    return """
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>SHL Assessment Recommender</title>
  <style>
    body { margin: 0; font-family: Inter, Segoe UI, Arial, sans-serif; background: #f7f8fb; color: #172033; }
    main { max-width: 980px; margin: 0 auto; padding: 36px 20px; }
    h1 { font-size: 32px; margin: 0 0 10px; letter-spacing: 0; }
    p { color: #5b6475; }
    form { display: grid; gap: 12px; margin: 24px 0; }
    textarea { min-height: 130px; padding: 14px; border: 1px solid #cfd6e4; border-radius: 8px; font: inherit; resize: vertical; }
    button { width: fit-content; border: 0; border-radius: 8px; background: #1264a3; color: white; padding: 11px 18px; font-weight: 700; cursor: pointer; }
    table { width: 100%; border-collapse: collapse; background: white; border: 1px solid #dde3ee; }
    th, td { padding: 12px; border-bottom: 1px solid #edf1f7; text-align: left; vertical-align: top; font-size: 14px; }
    th { background: #eef3f8; color: #2d3a4f; }
    a { color: #1264a3; }
    .muted { color: #6a7280; }
  </style>
</head>
<body>
  <main>
    <h1>SHL Assessment Recommender</h1>
    <p>Enter a hiring need, job description, or URL. The API returns up to 10 relevant SHL assessments.</p>
    <form id="form">
      <textarea id="query">I am hiring Java developers who can collaborate with business teams. The assessment should be completed in 40 minutes.</textarea>
      <button type="submit">Recommend</button>
    </form>
    <div id="status" class="muted"></div>
    <table id="results" hidden>
      <thead><tr><th>Assessment</th><th>Duration</th><th>Remote</th><th>Adaptive</th><th>Test Type</th></tr></thead>
      <tbody></tbody>
    </table>
  </main>
  <script>
    const form = document.querySelector("#form");
    const status = document.querySelector("#status");
    const table = document.querySelector("#results");
    const tbody = table.querySelector("tbody");
    form.addEventListener("submit", async (event) => {
      event.preventDefault();
      status.textContent = "Finding matches...";
      table.hidden = true;
      tbody.innerHTML = "";
      const response = await fetch("/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ message: document.querySelector("#query").value, max_recommendations: 10 })
      });
      const data = await response.json();
      for (const item of data.recommended_assessments || []) {
        const row = document.createElement("tr");
        row.innerHTML = `<td><a href="${item.url}" target="_blank" rel="noreferrer">${item.description.slice(0, 110)}...</a></td><td>${item.duration || "N/A"} min</td><td>${item.remote_support}</td><td>${item.adaptive_support}</td><td>${item.test_type.join(", ")}</td>`;
        tbody.appendChild(row);
      }
      status.textContent = "";
      table.hidden = false;
    });
  </script>
</body>
</html>
"""
