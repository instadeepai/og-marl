from absl import flags, app

from og_marl.environments.utils import get_environment
from og_marl.tf2.utils import get_system
from og_marl.loggers import WandbLogger
from og_marl.offline_dataset import OfflineMARLDataset

FLAGS = flags.FLAGS
flags.DEFINE_string("env", "smac_v1", "Environment name.")
flags.DEFINE_string("scenario", "3m", "Environment scenario name.")
flags.DEFINE_string("dataset", "", "Dataset type. 'Good', 'Medium', 'Poor' or '' for combined. ")
flags.DEFINE_string("system", "idrqn", "System name.")

def main(_):
    config = {
        "env": FLAGS.env,
        "scenario": FLAGS.scenario,
        "dataset": FLAGS.dataset if FLAGS.dataset != "" else "All",
        "system": FLAGS.system
    }

    env = get_environment(FLAGS.env, FLAGS.scenario)

    logger = WandbLogger(entity="claude_formanek", config=config)

    system_kwargs = {
        "add_agent_id_to_obs": True
    }
    system = get_system(FLAGS.system, env, logger, system_kwargs)

    dataset = OfflineMARLDataset(env, f"datasets/{FLAGS.env}/{FLAGS.scenario}/{FLAGS.dataset}")

    system.train_offline(
        dataset, 
        max_trainer_steps=1e6, 
        evaluate_every=1000
    )

if __name__ == "__main__":
    app.run(main)