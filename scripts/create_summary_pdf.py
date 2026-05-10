from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer


DOCS_DIR = Path("docs")
PDF_OUTPUT = DOCS_DIR / "approach_summary.pdf"
MD_OUTPUT = DOCS_DIR / "approach_summary.md"

SECTIONS = [
    (
        "SHL Assessment Recommendation System",
        [
            "Goal: build an API that recommends relevant SHL assessments from a natural language hiring need, job description, or URL. The service exposes GET /health, POST /recommend, and POST /chat.",
            "Design choices: I used a lightweight retrieval augmented generation style pipeline without an external hosted LLM so the service is cheap, reproducible, and fast to deploy. The system fetches the SHL-provided catalog JSON, tolerates minor malformed control characters in the source JSON, normalizes product fields, and caches the catalog locally when possible.",
            "Retrieval setup: each product is represented with name, description, test type, job level, language, duration, remote support, and adaptive support. The retriever builds an in-memory TF-IDF index over these fields. Query processing expands common aliases such as JS, QA, SDE, BPO, and extracts time constraints such as 40 minutes or one hour. Ranking combines TF-IDF similarity with transparent boosts for title matches, job family, seniority, remote/online constraints, and duration fit.",
            "Chat behavior: POST /chat accepts a latest message plus optional history. The endpoint combines recent history with the new message so changed constraints refine the ranking. If the request is underspecified, it returns clarifying questions about role, duration, and experience level. It also returns a compact comparison of the top assessments.",
        ],
    ),
    (
        "Grounding, Evaluation, and Lessons",
        [
            "Grounding: every recommendation is backed by catalog fields. The API returns evidence snippets containing the catalog description, duration, remote support, adaptive support, test type, and job levels. The comparison table uses only these retrieved product records, which reduces hallucination risk.",
            "Prompt design: because the deployed path is deterministic RAG rather than a hosted LLM, the prompt is represented as response rules in code: ask for missing hiring constraints, answer only from catalog evidence, and explain ranking in terms of duration, role match, and test type. If an LLM is added later, these same rules can become the system prompt.",
            "Evaluation method: scripts/evaluate.py measures retrieval quality and recommendation effectiveness with assignment-style test cases. It reports macro Recall@10 against expected product families, MRR for first relevant result, duration relevance for constraint satisfaction, and groundedness by verifying that returned URLs and descriptions exist in the catalog. It can also smoke-test a public deployment by calling GET /health and POST /chat.",
            "What did not work: the raw SHL JSON sometimes includes control characters, so strict JSON parsing failed. The loader was changed to use tolerant parsing. A tiny fallback catalog was useful for startup resilience but produced weak recommendations, so the full downloaded catalog is now cached locally for development.",
            "Measured improvement: after switching from the fallback records to the full catalog and returning the top-N ranked results, Java and sales queries retrieved relevant catalog families in the top results. Remaining improvement areas are semantic embeddings, richer labelled evaluation data, and optional LLM summarization for more natural comparisons.",
        ],
    ),
]


def write_markdown() -> None:
    lines = []
    for title, paragraphs in SECTIONS:
        lines.append(f"# {title}\n")
        for paragraph in paragraphs:
            lines.append(f"{paragraph}\n")
    MD_OUTPUT.write_text("\n".join(lines), encoding="utf-8")


def write_pdf() -> None:
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        "SectionTitle",
        parent=styles["Heading1"],
        fontName="Helvetica-Bold",
        fontSize=15,
        leading=18,
        textColor=colors.HexColor("#172033"),
        spaceAfter=10,
    )
    body_style = ParagraphStyle(
        "Body",
        parent=styles["BodyText"],
        fontName="Helvetica",
        fontSize=9.2,
        leading=12.3,
        textColor=colors.HexColor("#243044"),
        spaceAfter=7,
    )

    story = []
    for title, paragraphs in SECTIONS:
        story.append(Paragraph(title, title_style))
        for paragraph in paragraphs:
            story.append(Paragraph(paragraph, body_style))
        story.append(Spacer(1, 0.12 * inch))

    doc = SimpleDocTemplate(
        str(PDF_OUTPUT),
        pagesize=A4,
        rightMargin=0.55 * inch,
        leftMargin=0.55 * inch,
        topMargin=0.55 * inch,
        bottomMargin=0.55 * inch,
        title="SHL Assessment Recommendation System Approach",
        author="Chokkanath",
    )
    doc.build(story)


def main() -> None:
    DOCS_DIR.mkdir(exist_ok=True)
    write_markdown()
    write_pdf()
    print(PDF_OUTPUT)
    print(MD_OUTPUT)


if __name__ == "__main__":
    main()
