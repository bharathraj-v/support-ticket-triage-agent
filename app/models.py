from typing import Literal

from pydantic import BaseModel, Field

Language = Literal["nl", "en", "de"]
Category = Literal["bug", "billing", "feature_request", "question", "other"]
Priority = Literal["P1", "P2", "P3", "P4"]
Action = Literal["reply", "handoff", "close"]


class Decision(BaseModel):
    """What the model must return for each ticket."""
    language: Language = Field(description="Language the customer wrote in")
    category: Category
    priority: Priority
    action: Action
    reason: str = Field(description="One or two sentences in English: why this category, priority and action")
    faq_ids: list[str] = Field(default_factory=list, description="Help centre entries (F01..F12) the reply relies on")
    draft_reply: str | None = Field(default=None, description="Reply in the customer's language; only for action=reply")
    handoff_note: str | None = Field(default=None, description="English note for the support agent; only for action=handoff")
    confidence: float = Field(ge=0, le=1, description="How sure you are the action is right, 0 to 1")


class ReplyCheck(BaseModel):
    """Independent review of a drafted reply."""
    written_in_expected_language: bool
    english_back_translation: str
    supported_by_faq: bool = Field(description="Every claim in the reply is stated in the cited help centre entries")
    problems: list[str] = Field(default_factory=list)


class HandoffNote(BaseModel):
    note: str = Field(description="English note for the human support agent")
