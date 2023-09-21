# Copyright 2023 InstaDeep Ltd. All rights reserved.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


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