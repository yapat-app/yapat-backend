"""
Annotation schemas
"""

from pydantic import BaseModel
from datetime import datetime
from typing import Optional, Dict, Any


class AnnotationBase(BaseModel):
    species_name: str
    confidence: Optional[float] = None
    notes: Optional[str] = None
    extra_metadata: Optional[Dict[str, Any]] = None


class AnnotationCreate(AnnotationBase):
    snippet_id: int


class Annotation(AnnotationBase):
    id: int
    snippet_id: int
    user_id: int
    created_at: datetime
    updated_at: Optional[datetime] = None

    class Config:
        from_attributes = True

