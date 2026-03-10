import asyncio
import json
from typing import AsyncGenerator

from sqlalchemy import select

from config import settings
from database import async_session
from models.project import Project
from models.dataset import Dataset
from models.experiment import Experiment
from models.training_run import TrainingMetric
from models.evaluation import Evaluation
from models.chat_message import ChatMessage


class ChatService:
    async def stream_response(
        self, project_id: str, user_message: str, context: dict,
        cancel_event: asyncio.Event | None = None,
    ) -> AsyncGenerator[str, None]:
        # Save user message
        async with async_session() as db:
            db.add(ChatMessage(
                project_id=project_id, role="user",
                content=user_message, context=json.dumps(context),
            ))
            await db.commit()

        # Build system prompt with project context
        system_prompt = await self._build_system_prompt(project_id, context)

        # Get chat history
        history = await self._get_history(project_id)

        try:
            from openai import AsyncOpenAI

            client = AsyncOpenAI(
                base_url=settings.llm_base_url,
                api_key=settings.llm_api_key,
                timeout=60.0,
            )

            messages = [{"role": "system", "content": system_prompt}]
            for msg in history[-20:]:  # Last 20 messages for context
                messages.append({"role": msg["role"], "content": msg["content"]})
            messages.append({"role": "user", "content": user_message})

            full_response = ""
            stream = await client.chat.completions.create(
                model=settings.llm_model,
                messages=messages,
                stream=True,
            )
            async for chunk in stream:
                if cancel_event and cancel_event.is_set():
                    await stream.close()
                    break
                if chunk.choices and chunk.choices[0].delta.content:
                    token = chunk.choices[0].delta.content
                    full_response += token
                    yield token

            # Save assistant response (even partial if cancelled)
            if full_response:
                async with async_session() as db:
                    db.add(ChatMessage(
                        project_id=project_id, role="assistant",
                        content=full_response,
                    ))
                    await db.commit()

        except Exception as e:
            error_msg = f"Chat error: {str(e)}. Please check your LLM configuration in settings."
            yield error_msg
            async with async_session() as db:
                db.add(ChatMessage(
                    project_id=project_id, role="assistant", content=error_msg,
                ))
                await db.commit()

    async def _build_system_prompt(self, project_id: str, context: dict) -> str:
        async with async_session() as db:
            project = await db.get(Project, project_id)
            if not project:
                return "You are an ML assistant."

            prompt_parts = [
                "You are an expert ML assistant helping build an image model. "
                "Format your responses using markdown (headings, bullet points, code blocks, tables) for readability.",
                f"\n## Project\n- Name: {project.name}\n- Task: {project.task_type}\n- Status: {project.status}",
            ]

            # --- Dataset analysis ---
            result = await db.execute(
                select(Dataset).where(Dataset.project_id == project_id)
            )
            dataset = result.scalars().first()
            if dataset:
                ds_parts = [
                    f"\n## Dataset",
                    f"- Total images: {dataset.total_images}",
                    f"- Classes: {dataset.num_classes}",
                    f"- Format: {dataset.annotation_format}",
                ]

                if dataset.class_names:
                    names = json.loads(dataset.class_names)
                    ds_parts.append(f"- Class names: {', '.join(names[:30])}")

                if dataset.class_counts:
                    counts = json.loads(dataset.class_counts)
                    ds_parts.append(f"- Class distribution: {json.dumps(counts)}")

                if dataset.image_stats:
                    stats = json.loads(dataset.image_stats)
                    ds_parts.append(
                        f"- Image stats: avg {stats.get('avg_width', '?')}x{stats.get('avg_height', '?')}px, "
                        f"formats: {stats.get('formats', [])}"
                    )

                if dataset.quality_issues:
                    issues = json.loads(dataset.quality_issues)
                    if issues:
                        issue_strs = []
                        for iss in issues[:10]:
                            issue_strs.append(f"  - {iss.get('type', 'unknown')}: {iss.get('message', iss.get('count', ''))}")
                        ds_parts.append("- Quality issues:\n" + "\n".join(issue_strs))

                prompt_parts.extend(ds_parts)

            # --- Experiment (auto-find latest for this project) ---
            exp = None
            if context.get("experiment_id"):
                exp = await db.get(Experiment, context["experiment_id"])
            if not exp:
                result = await db.execute(
                    select(Experiment)
                    .where(Experiment.project_id == project_id)
                    .order_by(Experiment.created_at.desc())
                    .limit(1)
                )
                exp = result.scalars().first()

            if exp:
                hp = json.loads(exp.hyperparameters)
                prompt_parts.append(
                    f"\n## Model\n- Architecture: {exp.architecture}\n- Pretrained: {exp.pretrained}\n- Status: {exp.status}"
                    f"\n- Hyperparameters: lr={hp.get('lr')}, batch_size={hp.get('batch_size')}, "
                    f"epochs={hp.get('epochs')}, optimizer={hp.get('optimizer')}, "
                    f"augmentation={hp.get('augmentation')}, scheduler={hp.get('scheduler')}"
                )

                # --- Training history (last 10 epochs from DB) ---
                result = await db.execute(
                    select(TrainingMetric)
                    .where(TrainingMetric.experiment_id == exp.id)
                    .order_by(TrainingMetric.epoch.desc())
                    .limit(10)
                )
                db_metrics = list(reversed(result.scalars().all()))
                if db_metrics:
                    prompt_parts.append("\n## Training History (last epochs)")
                    prompt_parts.append("| Epoch | Train Loss | Val Loss | Val Acc | LR |")
                    prompt_parts.append("|-------|-----------|---------|---------|-----|")
                    for m in db_metrics:
                        if m.train_loss is not None:
                            prompt_parts.append(
                                f"| {m.epoch} | {m.train_loss:.4f} | "
                                f"{m.val_loss:.4f if m.val_loss is not None else '-'} | "
                                f"{m.val_accuracy:.4f if m.val_accuracy is not None else '-'} | "
                                f"{m.learning_rate:.6f if m.learning_rate is not None else '-'} |"
                            )
                        else:
                            prompt_parts.append(f"| {m.epoch} | - | - | - | - |")

                # --- Evaluation results ---
                result = await db.execute(
                    select(Evaluation)
                    .where(Evaluation.experiment_id == exp.id)
                    .order_by(Evaluation.created_at.desc())
                    .limit(1)
                )
                evaluation = result.scalars().first()
                if evaluation:
                    metrics = json.loads(evaluation.metrics) if isinstance(evaluation.metrics, str) else evaluation.metrics
                    prompt_parts.append(f"\n## Evaluation Results\n- Overall metrics: {json.dumps(metrics)}")

                    if evaluation.per_class_metrics:
                        pcm = json.loads(evaluation.per_class_metrics) if isinstance(evaluation.per_class_metrics, str) else evaluation.per_class_metrics
                        if pcm:
                            prompt_parts.append("- Per-class performance:")
                            for cls_metric in pcm[:20]:
                                prompt_parts.append(
                                    f"  - {cls_metric.get('class', '?')}: "
                                    f"precision={cls_metric.get('precision', '?')}, "
                                    f"recall={cls_metric.get('recall', '?')}, "
                                    f"f1={cls_metric.get('f1', '?')}"
                                )

                    if evaluation.confusion_matrix:
                        prompt_parts.append("- Confusion matrix is available (user can view in Evaluate page)")

            # --- Live training context from frontend ---
            live_metrics = context.get("trainingMetrics")
            if live_metrics and len(live_metrics) > 0:
                prompt_parts.append("\n## Live Training Progress (real-time)")
                ts = context.get("trainingStatus", "unknown")
                ce = context.get("currentEpoch")
                te = context.get("totalEpochs")
                if ce is not None and te is not None:
                    prompt_parts.append(f"- Status: {ts}, Epoch {ce}/{te}")
                else:
                    prompt_parts.append(f"- Status: {ts}")
                for lm in live_metrics:
                    prompt_parts.append(
                        f"- Epoch {lm.get('epoch', '?')}: train_loss={lm.get('train_loss', '?')}, "
                        f"val_loss={lm.get('val_loss', '?')}, val_acc={lm.get('val_accuracy', '?')}"
                    )

            prompt_parts.append(
                "\nHelp the user understand their model's performance, suggest improvements, "
                "explain ML concepts, and guide them through the AutoML process. "
                "Be concise and actionable. Use the data above to give specific, data-driven advice."
            )
            return "\n".join(prompt_parts)

    async def _get_history(self, project_id: str) -> list[dict]:
        async with async_session() as db:
            result = await db.execute(
                select(ChatMessage)
                .where(ChatMessage.project_id == project_id)
                .order_by(ChatMessage.created_at)
            )
            return [{"role": m.role, "content": m.content} for m in result.scalars().all()]
