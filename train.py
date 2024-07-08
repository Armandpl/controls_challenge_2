from typing import Any, Dict

import hydra
from omegaconf import DictConfig, OmegaConf
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.logger import Image
from stable_baselines3.common.monitor import Monitor
import wandb

from utils import seed_everything, upload_file_to_artifacts
from env import TinyPhysicsEnv


class EvalCallbackLogPlot(EvalCallback):
  # TODO this will probably break for multiple env or even multiple eval rollouts
  def _log_success_callback(
    self, locals_: Dict[str, Any], globals_: Dict[str, Any]
  ) -> None:
    """Callback passed to the  ``evaluate_policy`` function in order to log the success rate (when
    applicable), for instance when using HER.

    :param locals_:
    :param globals_:
    """
    info = locals_["info"]

    if locals_["done"]:
      # TODO make sure we're actually logging the mean and not just the last value
      self.logger.record_mean("eval/lataccel_cost", float(info["lataccel_cost"]))
      self.logger.record_mean("eval/jerk_cost", float(info["jerk_cost"]))
      self.logger.record_mean("eval/total_cost", float(info["total_cost"]))
      self.logger.record("eval/plot", Image(info["plot"], "HWC"), exclude="stdout")

      self.logger.record(
        "time/total_timesteps", self.num_timesteps, exclude="tensorboard"
      )
      self.logger.dump(self.num_timesteps)


@hydra.main(version_base="1.3", config_path="configs", config_name="train.yaml")
def main(cfg: DictConfig):
  run = wandb.init(
    config=OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True),
    project=cfg.wandb.project,
    sync_tensorboard=True,
  )
  seed_everything(cfg.seed)
  env = Monitor(hydra.utils.instantiate(cfg.env, _recursive_=True))
  eval_env = Monitor(
    hydra.utils.instantiate(
      cfg.env, eval=True, _recursive_=True
    )
  )
  check_env(env)

  # setup algo/model
  verbose = 2 if cfg.debug else 0
  model = hydra.utils.instantiate(
    cfg.algo,
    env=env,
    tensorboard_log=f"runs/{run.id}",
    verbose=verbose,
    _convert_="all",
    _recursive_=True,
  )

  try:
    model.learn(
      total_timesteps=cfg.total_timesteps,
      callback=None
      if cfg.evaluation.eval_freq is None
      else EvalCallbackLogPlot(
        eval_env=eval_env,
        n_eval_episodes=cfg.evaluation.n_eval_episodes,
        eval_freq=cfg.evaluation.eval_freq,
        deterministic=cfg.evaluation.deterministic,
      ),
      progress_bar=cfg.progress_bar,
    )
  except KeyboardInterrupt:
    print("Training interrupted, saving model...")

  model.save("../models/rl_controller")
  if cfg.wandb.save_model:
    upload_file_to_artifacts("../models/rl_controller.zip", "model", "model")

  run.finish()


if __name__ == "__main__":
  main()
