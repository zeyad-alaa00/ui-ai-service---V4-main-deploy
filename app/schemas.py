# schemas.py
from __future__ import annotations

from typing import Literal, List, Optional
from pydantic import BaseModel, Field, HttpUrl, field_validator


class AnalyzeUIRequest(BaseModel):
    imageUrl: HttpUrl
    


class BBoxNorm(BaseModel):
    x_norm: float
    y_norm: float
    w_norm: float
    h_norm: float

    @field_validator("x_norm", "y_norm", "w_norm", "h_norm", mode="before")
    @classmethod
    def validate_range(cls, v):
        try:
            v = float(v)
        except Exception:
            return 0.0

        return max(0.0, min(1.0, v))


class UIElement(BaseModel):
    id: str
    type: str

    bbox_norm: BBoxNorm

    is_clickable: bool = False
    text: Optional[str] = None

    # optional advanced fields (من الموديل)
    font_weight: Optional[str] = "regular"
    font_size_role: Optional[str] = "body"

    @field_validator("type")
    @classmethod
    def normalize_type(cls, v: str):
        return v.strip().lower()

    @field_validator("text")
    @classmethod
    def clean_text(cls, v: Optional[str]):
        if v is None:
            return None
        v = v.strip()
        return v if v else None


class AnalyzeUIResponse(BaseModel):
    screen_id: Optional[str] = None
    elements: List[UIElement]