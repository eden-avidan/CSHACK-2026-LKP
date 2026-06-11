from __future__ import annotations

from pydantic import BaseModel, Field


class PersonalityProfile(BaseModel):
    """Subject traits used by the personality layer mobility heuristic."""

    age: int = Field(default=35, ge=1, le=120)
    fitness: int = Field(default=3, ge=1, le=5, description="1=low … 5=high")
    injured: bool = False
