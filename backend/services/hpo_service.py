import json
import asyncio
import logging

from sqlalchemy.ext.asyncio import AsyncSession

from models.experiment import Experiment
from config import settings
from ws.manager import ws_manager
from database import async_session

logger = logging.getLogger(__name__)


class HPOService:
    _active_studies: dict[str, bool] = {}  # experiment_id -> stop flag

    async def start_hpo(self, experiment: Experiment, n_trials: int, db: AsyncSession) -> dict:
        hp = json.loads(experiment.hyperparameters)
        resource_config = json.loads(experiment.resource_config) if experiment.resource_config else {}

        from models.dataset import Dataset
        from models.project import Project
        dataset = await db.get(Dataset, experiment.dataset_id)
        project = await db.get(Project, experiment.project_id)

        self._active_studies[experiment.id] = False

        asyncio.create_task(
            self._run_hpo(
                experiment.id, project.task_type, experiment.architecture,
                experiment.pretrained, hp, resource_config,
                dataset.path, dataset.num_classes, n_trials,
            )
        )
        return {"status": "started", "n_trials": n_trials}

    def stop_hpo(self, experiment_id: str):
        self._active_studies[experiment_id] = True

    async def _run_hpo(
        self, experiment_id, task_type, architecture, pretrained,
        base_hp, resource_config, dataset_path, num_classes, n_trials,
    ):
        import optuna

        loop = asyncio.get_event_loop()

        def objective(trial):
            if self._active_studies.get(experiment_id, True):
                raise optuna.TrialPruned("Stopped by user")

            lr = trial.suggest_float("lr", 1e-5, 1e-1, log=True)
            batch_size = trial.suggest_categorical("batch_size", [8, 16, 32])
            optimizer_name = trial.suggest_categorical("optimizer", ["adam", "adamw", "sgd"])
            weight_decay = trial.suggest_float("weight_decay", 1e-5, 1e-2, log=True)

            hp = {**base_hp}
            hp.update({
                "lr": lr,
                "batch_size": batch_size,
                "optimizer": optimizer_name,
                "weight_decay": weight_decay,
                "epochs": min(base_hp.get("epochs", 50), 10),  # Short runs for HPO
            })

            if task_type == "classification":
                from ml.trainers.classification_trainer import ClassificationTrainer
                trainer = ClassificationTrainer(
                    experiment_id=f"{experiment_id}_hpo_{trial.number}",
                    architecture=architecture,
                    pretrained=pretrained,
                    hyperparameters=hp,
                    resource_config=resource_config,
                    dataset_path=dataset_path,
                    num_classes=num_classes,
                    checkpoint_dir=str(settings.checkpoint_dir / f"{experiment_id}_hpo"),
                    stop_flag=lambda: self._active_studies.get(experiment_id, True),
                )
            else:
                return 0.0

            best_val_acc = 0.0
            for update in trainer.train():
                if update.get("type") == "epoch_end":
                    val_acc = update.get("val_accuracy", 0.0)
                    best_val_acc = max(best_val_acc, val_acc)
                    trial.report(val_acc, update["epoch"])
                    if trial.should_prune():
                        raise optuna.TrialPruned()

            # Broadcast trial result
            asyncio.run_coroutine_threadsafe(
                ws_manager.broadcast(f"hpo:{experiment_id}", {
                    "type": "trial_complete",
                    "trial": trial.number,
                    "total_trials": n_trials,
                    "params": {"lr": lr, "batch_size": batch_size, "optimizer": optimizer_name, "weight_decay": weight_decay},
                    "value": best_val_acc,
                }),
                loop,
            ).result(timeout=5)

            return best_val_acc

        def _run_study():
            study = optuna.create_study(
                direction="maximize",
                sampler=optuna.samplers.TPESampler(),
                pruner=optuna.pruners.MedianPruner(n_startup_trials=3, n_warmup_steps=2),
            )
            study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
            return study

        try:
            study = await asyncio.to_thread(_run_study)
            best = study.best_trial
            await ws_manager.broadcast(f"hpo:{experiment_id}", {
                "type": "study_complete",
                "best_params": best.params,
                "best_value": best.value,
                "n_trials": len(study.trials),
            })
        except Exception as e:
            logger.exception(f"HPO failed: {e}")
            await ws_manager.broadcast(f"hpo:{experiment_id}", {
                "type": "hpo_failed",
                "error": str(e),
            })
        finally:
            self._active_studies.pop(experiment_id, None)
