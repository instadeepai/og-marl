from datetime import datetime
import wandb
import time

class WandbLogger:
    def __init__(
        self,
        config={},
        project="default_project",
        notes="",
        tags=["default"],
        entity="arkalim",
        log_every=2 # seconds
    ):
        wandb.init(project=project, notes=notes, tags=tags, entity=entity, config=config)

        self._log_every = log_every
        self._ctr = 0
        self._last_log = time.time()

    def write(self, logs, force=False):
        
        
        if time.time() - self._last_log > self._log_every or force:
            wandb.log(logs)

            for key, log in logs.items():
                print(f"{key}: {float(log)} |", end=" ")
            print()

            if not force:
                self._last_log = time.time()

        self._ctr += 1

    def close(self):
        wandb.finish()