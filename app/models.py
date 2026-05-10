from typing import List, Optional

from pydantic import BaseModel, Field, field_validator


class RecommendRequest(BaseModel):
    query: str = Field(..., min_length=1, description="Natural language query, job description, or URL")
    max_recommendations: Optional[int] = Field(default=10, ge=1, le=10)

    @field_validator("query")
    @classmethod
    def strip_query(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("query cannot be blank")
        return value


class Assessment(BaseModel):
    url: str
    adaptive_support: str
    description: str
    duration: int
    remote_support: str
    test_type: List[str]


class RecommendationResponse(BaseModel):
    recommended_assessments: List[Assessment]


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    message: Optional[str] = Field(default=None, description="Latest user message")
    query: Optional[str] = Field(default=None, description="Alias for message")
    history: List[ChatMessage] = Field(default_factory=list)
    max_recommendations: Optional[int] = Field(default=10, ge=1, le=10)

    @field_validator("message", "query")
    @classmethod
    def strip_optional_text(cls, value: Optional[str]) -> Optional[str]:
        if value is None:
            return value
        value = value.strip()
        return value or None


class ChatAssessment(Assessment):
    name: str
    evidence: List[str]


class ChatResponse(BaseModel):
    answer: str
    clarifying_questions: List[str]
    recommended_assessments: List[ChatAssessment]
    comparison: List[dict]
    groundedness: str
