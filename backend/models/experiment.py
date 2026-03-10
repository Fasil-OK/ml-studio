import uuid
from datetime import datetime, timezone

from sqlalchemy import String, Boolean, DateTime, Text, ForeignKey
from sqlalchemy.orm import Mapped, mapped_column, relationship

from database import Base


class Experiment(Base):
    __tablename__ = "experiments"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    project_id: Mapped[str] = mapped_column(String, ForeignKey("projects.id", ondelete="CASCADE"), nullable=False)
    dataset_id: Mapped[str] = mapped_column(String, ForeignKey("datasets.id"), nullable=False)
    architecture: Mapped[str] = mapped_column(String, nullable=False)
    pretrained: Mapped[bool] = mapped_column(Boolean, default=True)
    hyperparameters: Mapped[str] = mapped_column(Text, nullable=False)  # JSON
    resource_config: Mapped[str | None] = mapped_column(Text, nullable=True)  # JSON
    status: Mapped[str] = mapped_column(String, default="created")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=lambda: datetime.now(timezone.utc))

    project = relationship("Project", back_populates="experiments")
    training_metrics = relationship("TrainingMetric", back_populates="experiment", cascade="all, delete-orphan")
    evaluations = relationship("Evaluation", back_populates="experiment", cascade="all, delete-orphan")
