import uuid
from datetime import datetime, timezone

from sqlalchemy import String, Integer, DateTime, Text, ForeignKey
from sqlalchemy.orm import Mapped, mapped_column, relationship

from database import Base


class Dataset(Base):
    __tablename__ = "datasets"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    project_id: Mapped[str] = mapped_column(String, ForeignKey("projects.id", ondelete="CASCADE"), nullable=False)
    name: Mapped[str] = mapped_column(String, nullable=False)
    path: Mapped[str] = mapped_column(String, nullable=False)
    total_images: Mapped[int | None] = mapped_column(Integer, nullable=True)
    num_classes: Mapped[int | None] = mapped_column(Integer, nullable=True)
    class_names: Mapped[str | None] = mapped_column(Text, nullable=True)  # JSON
    class_counts: Mapped[str | None] = mapped_column(Text, nullable=True)  # JSON
    image_stats: Mapped[str | None] = mapped_column(Text, nullable=True)  # JSON
    annotation_format: Mapped[str | None] = mapped_column(String, nullable=True)
    quality_issues: Mapped[str | None] = mapped_column(Text, nullable=True)  # JSON
    split_info: Mapped[str | None] = mapped_column(Text, nullable=True)  # JSON
    created_at: Mapped[datetime] = mapped_column(DateTime, default=lambda: datetime.now(timezone.utc))

    project = relationship("Project", back_populates="datasets")
