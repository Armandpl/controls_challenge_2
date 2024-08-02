from typing import Any, Dict
import warnings

import hydra
from omegaconf import DictConfig, OmegaConf
from stable_baselines3.common.callbacks import EventCallback
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.logger import Image
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, sync_envs_normalization, VecEnv
from typing import Union
import gymnasium as gym
import numpy as np
import wandb

from utils import seed_everything, upload_file_to_artifacts
from env import TinyPhysicsEnv

class EvalCallback(EventCallback):
  def __init__(
    self,
    eval_env: Union[gym.Env, VecEnv],
    n_eval_episodes: int = 8,
    eval_freq: int = 10000,
    deterministic: bool = True,
  ):
    super().__init__()

    self.n_eval_episodes = n_eval_episodes
    self.eval_freq = eval_freq
    self.deterministic = deterministic

    # Convert to VecEnv for consistency
    if not isinstance(eval_env, VecEnv):
      eval_env = DummyVecEnv([lambda: eval_env])  # type: ignore[list-item, return-value]

    self.eval_env = eval_env

  def _init_callback(self) -> None:
    # Does not work in some corner cases, where the wrapper is not the same
    if not isinstance(self.training_env, type(self.eval_env)):
      warnings.warn("Training and eval env are not of the same type" f"{self.training_env} != {self.eval_env}")

  def _on_step(self) -> bool:
    if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
      # Sync training and eval env if there is VecNormalize
      if self.model.get_vec_normalize_env() is not None:
        try:
          sync_envs_normalization(self.training_env, self.eval_env)
        except AttributeError as e:
          raise AssertionError(
            "Training and eval env are not wrapped the same way, "
            "see https://stable-baselines3.readthedocs.io/en/master/guide/callbacks.html#evalcallback "
            "and warning above."
          ) from e

      stats, plots = {}, []

      for _ in range(self.n_eval_episodes):
        obs = self.eval_env.reset()
        done = False
        return_ = 0
        len_ = 0
        while not done:
          action, _states = self.model.predict(obs)
          obs, rewards, dones, info = self.eval_env.step(action)
          return_ += rewards[0] # single eval env
          done = dones[0]
          len_ += 1

        stats.setdefault("eval/mean_reward", []).append(return_)
        stats.setdefault("eval/mean_ep_len", []).append(len_)
        for k,v in info[0].items():
          if k == 'plot':
            plots.append(info[0]['plot'])
          elif isinstance(v, (float)):
            stats.setdefault(f"eval/mean_{k}", []).append(v)

      for k,v in stats.items():
        self.logger.record(k,float(np.mean(v)))

      # stack the plots
      # TODO clean that up, make func
      plots = plots[:8] # only show first 8
      leftcol = np.concatenate(plots[:4], axis=0)
      rightcol = np.concatenate(plots[4:], axis=0)
      plot_to_log = np.concatenate([leftcol, rightcol], axis=1)

      self.logger.record("eval/plot", Image(plot_to_log, "HWC"), exclude="stdout")

      self.logger.record(
        "time/total_timesteps", self.num_timesteps, exclude="tensorboard"
      )
      self.logger.dump(self.num_timesteps)

    return True


@hydra.main(version_base="1.3", config_path="configs", config_name="train.yaml")
def main(cfg: DictConfig):
  run = wandb.init(
    config=OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True),
    project=cfg.wandb.project,
    sync_tensorboard=True,
  )
  seed_everything(cfg.seed)
  def make_env():
    return Monitor(hydra.utils.instantiate(cfg.env, _recursive_=True))
  env = SubprocVecEnv([make_env for _ in range(cfg.n_envs)])
  eval_env = Monitor(
    hydra.utils.instantiate(
      cfg.env, eval=True, _recursive_=True
    )
  )
  check_env(eval_env)

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
      else EvalCallback(
        eval_env=eval_env,
        n_eval_episodes=cfg.evaluation.n_eval_episodes,
        eval_freq=cfg.evaluation.eval_freq // cfg.n_envs,
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
