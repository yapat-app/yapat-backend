"""
Classifier schemas
"""

from pydantic import BaseModel
from datetime import datetime
from typing import Optional, Dict, Any


class ClassifierBase(BaseModel):
    name: str
    description: Optional[str] = None
    model_type: Optional[str] = None


class ClassifierCreate(ClassifierBase):
    team_id: int
    extra_metadata: Optional[Dict[str, Any]] = None


class ClassifierUpdate(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None
    model_type: Optional[str] = None
    model_path: Optional[str] = None
    accuracy: Optional[float] = None
    status: Optional[str] = None
    extra_metadata: Optional[Dict[str, Any]] = None


class Classifier(ClassifierBase):
    id: int
    team_id: int
    model_path: Optional[str] = None
    accuracy: Optional[float] = None
    status: str
    extra_metadata: Optional[Dict[str, Any]] = None
    created_at: datetime
    updated_at: Optional[datetime] = None

    class Config:
        from_attributes = True

