from __future__ import annotations
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
import shutil
import yaml

from deep_belief_betting.agent_training.training_config import TrainingConfig


def create_run_id(run_name: str) -> str:
    return f"{run_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"


def create_run_dir(log_dir: str, run_name: str, training_config: TrainingConfig) -> Path:
    base = Path(log_dir).expanduser()
    run_dir = base / create_run_id(run_name)
    run_dir.mkdir(parents=True, exist_ok=False)

    (run_dir / "checkpoints").mkdir()
    (run_dir / "logs").mkdir()

    resolved = run_dir / "training_config.resolved.yaml"
    with resolved.open("w", encoding="utf-8") as f:
        yaml.safe_dump(asdict(training_config), f, sort_keys=True)
    
    shutil.copy2(training_config.world_yaml_base_path, run_dir / "world_snapshot.yaml")

    return run_dir

