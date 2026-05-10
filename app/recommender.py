import json
import math
import re
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from html import unescape
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple
from urllib.parse import urlparse

import requests
from bs4 import BeautifulSoup


CATALOG_URL = "https://tcp-us-prod-rnd.shl.com/voiceRater/shl-ai-hiring/shl_product_catalog.json"
CACHE_DIR = Path(__file__).resolve().parent.parent / "data"
CACHE_FILE = CACHE_DIR / "shl_product_catalog.json"

STOPWORDS = {
    "a", "about", "also", "an", "and", "are", "as", "at", "be", "by", "can",
    "candidate", "candidates", "completed", "each", "for", "from", "give", "hire",
    "hiring", "i", "in", "is", "it", "job", "looking", "me", "my", "new", "of",
    "on", "or", "our", "role", "roles", "should", "that", "the", "their", "this",
    "to", "want", "we", "who", "with", "within", "you",
}

SKILL_ALIASES = {
    "js": "javascript",
    "node": "node.js",
    "nodejs": "node.js",
    "reactjs": "react",
    "ml": "machine learning",
    "ai": "artificial intelligence",
    "qa": "quality assurance testing",
    "sde": "software developer engineer",
    "dev": "developer",
    "csr": "customer service representative",
    "bpo": "contact center customer service",
}


@dataclass
class Product:
    name: str
    link: str
    description: str
    duration: int
    remote: str
    adaptive: str
    test_type: List[str]
    job_levels: List[str]
    languages: List[str]
    text: str


def _clean_text(value: object) -> str:
    if value is None:
        return ""
    text = unescape(str(value)).replace("\u00a0", " ")
    text = text.encode("latin1", errors="ignore").decode("utf-8", errors="ignore") if "â" in text else text
    return re.sub(r"\s+", " ", text).strip()


def _duration_to_minutes(value: object) -> int:
    text = _clean_text(value)
    if not text or text.lower() in {"n/a", "untimed"}:
        return 0
    match = re.search(r"\d+", text)
    return int(match.group(0)) if match else 0


def _yes_no(value: object) -> str:
    text = _clean_text(value).lower()
    return "Yes" if text in {"yes", "y", "true", "1"} else "No"


def _tokenize(text: str) -> List[str]:
    expanded = text.lower()
    for source, target in SKILL_ALIASES.items():
        expanded = re.sub(rf"\b{re.escape(source)}\b", target, expanded)
    tokens = re.findall(r"[a-z0-9+#.]+", expanded)
    return [token for token in tokens if len(token) > 1 and token not in STOPWORDS]


def _load_fallback_catalog() -> List[Dict[str, object]]:
    return [
        {
            "name": "Core Java (Entry Level) (New)",
            "link": "https://www.shl.com/products/product-catalog/view/core-java-entry-level-new/",
            "duration": "16 minutes",
            "remote": "yes",
            "adaptive": "no",
            "description": "Multi-choice test that measures entry-level Core Java programming knowledge.",
            "keys": ["Knowledge & Skills"],
            "job_levels": ["Entry-Level", "Graduate"],
            "languages": ["English (USA)"],
        },
        {
            "name": "Java 8 (New)",
            "link": "https://www.shl.com/products/product-catalog/view/java-8-new/",
            "duration": "9 minutes",
            "remote": "yes",
            "adaptive": "no",
            "description": "Multi-choice test for Java 8 language features and programming concepts.",
            "keys": ["Knowledge & Skills"],
            "job_levels": ["Mid-Professional"],
            "languages": ["English (USA)"],
        },
        {
            "name": "Entry Level Sales Solution",
            "link": "https://www.shl.com/products/product-catalog/view/entry-level-sales-solution/",
            "duration": "50 minutes",
            "remote": "yes",
            "adaptive": "no",
            "description": "Assessment solution for entry-level sales hiring.",
            "keys": ["Ability & Aptitude", "Personality & Behavior"],
            "job_levels": ["Entry-Level", "Graduate"],
            "languages": ["English (USA)"],
        },
    ]


class SHLRecommender:
    def __init__(self) -> None:
        self.products: List[Product] = []
        self.doc_vectors: List[Dict[str, float]] = []
        self.idf: Dict[str, float] = {}
        self.ready = False
        self.loaded_at = 0.0

    def load(self) -> None:
        raw_items = self._fetch_catalog()
        self.products = [self._normalize(item) for item in raw_items if item.get("status", "ok") == "ok"]
        self._build_index()
        self.ready = bool(self.products)
        self.loaded_at = time.time()

    def recommend(self, query: str, limit: int = 10) -> List[Product]:
        if not self.ready:
            self.load()
        query_text = self._query_text(query)
        max_minutes = self._extract_time_limit(query_text)
        query_tokens = _tokenize(query_text)
        query_vec = self._vectorize(query_tokens)

        scored: List[Tuple[float, Product]] = []
        for product, doc_vec in zip(self.products, self.doc_vectors):
            score = self._cosine(query_vec, doc_vec)
            score += self._business_boost(query_text, query_tokens, product, max_minutes)
            if max_minutes and product.duration and product.duration > max_minutes:
                score -= min(0.35, (product.duration - max_minutes) / 120)
            scored.append((score, product))

        scored.sort(key=lambda item: item[0], reverse=True)
        return [product for _, product in scored[: max(1, limit)]]

    def _fetch_catalog(self) -> List[Dict[str, object]]:
        try:
            response = requests.get(CATALOG_URL, timeout=20)
            response.raise_for_status()
            payload = json.loads(response.text, strict=False)
            try:
                CACHE_DIR.mkdir(exist_ok=True)
                CACHE_FILE.write_text(json.dumps(payload), encoding="utf-8")
            except OSError:
                pass
            return payload
        except Exception:
            if CACHE_FILE.exists():
                return json.loads(CACHE_FILE.read_text(encoding="utf-8"), strict=False)
            return _load_fallback_catalog()

    def _normalize(self, item: Dict[str, object]) -> Product:
        name = _clean_text(item.get("name"))
        description = _clean_text(item.get("description"))
        test_type = [_clean_text(value) for value in item.get("keys") or [] if _clean_text(value)]
        job_levels = [_clean_text(value) for value in item.get("job_levels") or [] if _clean_text(value)]
        languages = [_clean_text(value) for value in item.get("languages") or [] if _clean_text(value)]
        duration = _duration_to_minutes(item.get("duration") or item.get("duration_raw"))
        remote = _yes_no(item.get("remote"))
        adaptive = _yes_no(item.get("adaptive"))
        fields = [name, description, " ".join(test_type), " ".join(job_levels), " ".join(languages)]
        return Product(
            name=name,
            link=_clean_text(item.get("link")),
            description=description,
            duration=duration,
            remote=remote,
            adaptive=adaptive,
            test_type=test_type or ["General Assessment"],
            job_levels=job_levels,
            languages=languages,
            text=" ".join(fields),
        )

    def _build_index(self) -> None:
        tokenized_docs = [_tokenize(product.text) for product in self.products]
        doc_frequency: Dict[str, int] = defaultdict(int)
        for tokens in tokenized_docs:
            for token in set(tokens):
                doc_frequency[token] += 1

        doc_count = max(1, len(tokenized_docs))
        self.idf = {
            token: math.log((1 + doc_count) / (1 + frequency)) + 1
            for token, frequency in doc_frequency.items()
        }
        self.doc_vectors = [self._vectorize(tokens) for tokens in tokenized_docs]

    def _vectorize(self, tokens: Iterable[str]) -> Dict[str, float]:
        counts = Counter(tokens)
        if not counts:
            return {}
        total = sum(counts.values())
        return {
            token: (count / total) * self.idf.get(token, 1.0)
            for token, count in counts.items()
        }

    def _cosine(self, left: Dict[str, float], right: Dict[str, float]) -> float:
        if not left or not right:
            return 0.0
        numerator = sum(value * right.get(token, 0.0) for token, value in left.items())
        left_norm = math.sqrt(sum(value * value for value in left.values()))
        right_norm = math.sqrt(sum(value * value for value in right.values()))
        return numerator / (left_norm * right_norm) if left_norm and right_norm else 0.0

    def _query_text(self, query: str) -> str:
        parsed = urlparse(query)
        if parsed.scheme in {"http", "https"} and parsed.netloc:
            try:
                response = requests.get(query, timeout=10, headers={"User-Agent": "Mozilla/5.0"})
                response.raise_for_status()
                soup = BeautifulSoup(response.text, "html.parser")
                for tag in soup(["script", "style", "nav", "footer"]):
                    tag.decompose()
                text = soup.get_text(" ")
                return _clean_text(text)[:12000] or query
            except Exception:
                return query
        return query

    def _extract_time_limit(self, text: str) -> Optional[int]:
        lowered = text.lower()
        hour_match = re.search(r"(?:about|under|within|less than|<=?)?\s*(\d+(?:\.\d+)?)\s*hours?", lowered)
        minute_match = re.search(r"(?:about|under|within|less than|<=?)?\s*(\d+)\s*(?:minutes?|mins?)", lowered)
        if hour_match:
            return int(float(hour_match.group(1)) * 60)
        if "an hour" in lowered or "one hour" in lowered:
            return 60
        if minute_match:
            return int(minute_match.group(1))
        return None

    def _business_boost(
        self,
        query_text: str,
        query_tokens: List[str],
        product: Product,
        max_minutes: Optional[int],
    ) -> float:
        name = product.name.lower()
        description = product.description.lower()
        corpus = f"{name} {description}"
        score = 0.0

        for token in set(query_tokens):
            if token in name:
                score += 0.08
            elif token in description:
                score += 0.03

        if max_minutes and product.duration and product.duration <= max_minutes:
            score += 0.12
        if re.search(r"\b(entry|graduate|new graduates?|junior)\b", query_text.lower()):
            if any(level in {"Entry-Level", "Graduate"} for level in product.job_levels):
                score += 0.14
        if re.search(r"\b(manager|leadership|supervisor)\b", query_text.lower()):
            if any(level in {"Manager", "Supervisor", "Front Line Manager"} for level in product.job_levels):
                score += 0.12
        if re.search(r"\b(remote|online|virtual)\b", query_text.lower()) and product.remote == "Yes":
            score += 0.06
        if "sales" in query_text.lower() and "sales" in corpus:
            score += 0.20
        if any(word in query_text.lower() for word in ("java", "developer", "software", "programmer", "coding")):
            if any(word in corpus for word in ("java", "software", "programming", "coding", "computer")):
                score += 0.16
        if any(word in query_text.lower() for word in ("customer", "contact center", "call center", "service")):
            if any(word in corpus for word in ("customer", "contact center", "call center", "service")):
                score += 0.18
        return score
