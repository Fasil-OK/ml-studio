import uuid
from datetime import datetime, timezone

from sqlalchemy import String, DateTime, Text, ForeignKey
from sqlalchemy.orm import Mapped, mapped_column, relationship

from database import Base


class Evaluation(Base):
    __tablename__ = "evaluations"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    experiment_id: Mapped[str] = mapped_column(String, ForeignKey("experiments.id", ondelete="CASCADE"), nullable=False)
    metrics: Mapped[str] = mapped_column(Text, nullable=False)  # JSON
    confusion_matrix: Mapped[str | None] = mapped_column(Text, nullable=True)  # JSON
    per_class_metrics: Mapped[str | None] = mapped_column(Text, nullable=True)  # JSON
    best_checkpoint: Mapped[str | None] = mapped_column(String, nullable=True)
    insights: Mapped[str | None] = mapped_column(Text, nullable=True)  # JSON
    created_at: Mapped[datetime] = mapped_column(DateTime, default=lambda: datetime.now(timezone.utc))

    experiment = relationship("Experiment", back_populates="evaluations")
