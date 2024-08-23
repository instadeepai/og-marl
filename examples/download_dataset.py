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
from absl import app, flags

from og_marl.vault_utils.download_vault import download_and_unzip_vault

FLAGS = flags.FLAGS
flags.DEFINE_string("source", "og_marl", "Source of dataset.")
flags.DEFINE_string("env", "smac_v1", "Environment name.")
flags.DEFINE_string("scenario", "3m", "Environment scenario name.")


def main(_):
    # Download vault
    download_and_unzip_vault(FLAGS.source, FLAGS.env, FLAGS.scenario)

    # NEXT STEPS: See `examples/dataset_api_demo.ipynb`


if __name__ == "__main__":
    app.run(main)
