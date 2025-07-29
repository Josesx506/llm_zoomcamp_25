from datetime import datetime, timezone
from typing import Optional

from fastapi import HTTPException
from sqlmodel import Field, Session, SQLModel, create_engine, select


class CustomModel(SQLModel):
    """
    Base model that other models can inherit from.
        In SQLModel, any model class that has `table=True` is a table model.
        And any model class that doesn't have `table=True` is a data model, these 
        ones are actually just Pydantic models (with a couple of small extra features). 
    """
    id: int | None = Field(default=None, primary_key=True) # auto-increment

    def insert(self, session: Session):
        """Adds and commits this object to the database."""
        try:
            session.add(self)
            session.commit()
            session.refresh(self)
        except Exception as e:
            session.rollback()
            print(f"Error inserting Sample: {e}")
            raise HTTPException(status_code=500, detail=f"Database insert error: {e}")

    def update(self, session: Session):
        """Commits changes to this object in the database."""
        try:
            session.add(self)
            session.commit()
            session.refresh(self)
        except Exception as e:
            session.rollback()
            print(f"Error updating Sample: {e}")
            raise HTTPException(status_code=500, detail=f"Database update error: {e}")

    def delete(self, session: Session):
        """Deletes this object from the database."""
        try:
            session.delete(self)
            session.commit()
        except Exception as e:
            session.rollback()
            print(f"Error deleting Sample: {e}")
            raise HTTPException(status_code=500, detail=f"Database delete error: {e}")



class Conversations(CustomModel, table=True):
    question: str = Field(index=False)
    answer: str = Field(index=False)
    model_used: str = Field(index=False)
    response_time: float = Field(index=False)
    prompt_tokens: int | None = Field(default=0, index=False)
    completion_tokens: int | None = Field(default=0, index=False)
    total_tokens: int | None = Field(default=0, index=False)
    eval_prompt_tokens: int | None = Field(default=0, index=False)
    eval_completion_tokens: int | None = Field(default=0, index=False)
    eval_total_tokens: int | None = Field(default=0, index=False)
    openai_cost: float | None = Field(default=0, index=False)
    timestamp: Optional[datetime] = Field(default_factory=lambda: datetime.now(timezone.utc))


class Sample(CustomModel, table=True):
    answer: str = Field(index=False)