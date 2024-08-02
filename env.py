from pathlib import Path
import random
from typing import Optional

import gymnasium as gym
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from tinyphysics import (
  TinyPhysicsSimulator,
  TinyPhysicsModel,
  STEER_RANGE,
  CONTROL_START_IDX,
  LAT_ACCEL_COST_MULTIPLIER,
  DEL_T,
  COST_END_IDX,
  CONTEXT_LENGTH,
  LATACCEL_RANGE
)
from controllers.zero import Controller as ZeroController
from controllers.pid import Controller as PIDController
from utils import canvas_to_img

DATA_PATH = Path("./data/")
TEST_SET_SIZE = 5000


def symlog(x):
  return np.sign(x) * np.log(np.abs(x)+1)


class TinyPhysicsEnv(gym.Env):
  def __init__(self, eval=False, max_lataccel_err:float=1.0, symlog_obs:bool=True):
    self.eval = eval
    self.max_lataccel_err = max_lataccel_err
    self.symlog_obs = symlog_obs

    self.action_space = gym.spaces.Box(STEER_RANGE[0], STEER_RANGE[1])
    self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(169,))

    self.tinyphysicsmodel = None

    data = sorted(DATA_PATH.iterdir())[TEST_SET_SIZE:]
    if self.eval:
      self.files = data[:TEST_SET_SIZE]
    else: # make sure we don't train on test set
      self.files = data[TEST_SET_SIZE:]

  def _get_obs(self):
    raw_states = [list(x) for x in self.sim.state_history[-CONTEXT_LENGTH:]]
    actions = self.sim.action_history[-CONTEXT_LENGTH:]
    lataccels = self.sim.current_lataccel_history[-CONTEXT_LENGTH:]
    target_lataccels = self.sim.target_lataccel_history[-CONTEXT_LENGTH:]

    # only give plan, not future states as we can't get those at inference time
    plan = self.sim.futureplan.lataccel

    obs = np.concatenate([
      np.array(lataccels), # (20,)
      np.array(target_lataccels), # (20,)
      np.array(actions), # (20,)
      np.array(raw_states).flatten(), # (20*3,) roll_lataccel, v_ego, a_ego
      np.array(plan) # (49,)
    ], dtype=np.float32) # (169,)

    if self.symlog_obs:
      obs = symlog(obs)

    return obs

  def step(self, action: float):
    # control step:
    action = np.clip(action, self.action_space.low, self.action_space.high)
    action = float(action[0])
    self.sim.action_history.append(action)

    self.sim.sim_step(self.sim.step_idx)
    self.sim.step_idx += 1

    target = self.sim.target_lataccel_history[-1]
    current = self.sim.current_lataccel_history[-1]
    prev = self.sim.current_lataccel_history[-2]

    lat_accel_cost = ((target - current)**2)
    jerk_cost = (((current - prev) / DEL_T)**2)
    total_cost = lat_accel_cost + jerk_cost/LAT_ACCEL_COST_MULTIPLIER

    reward  = -symlog(total_cost)

    state, target, futureplan = self.sim.get_state_target_futureplan(self.sim.step_idx)
    self.sim.state_history.append(state)
    self.sim.target_lataccel_history.append(target)
    self.sim.futureplan = futureplan
    obs = self._get_obs()

    truncated = False
    info = {}
    terminated = self.sim.step_idx == COST_END_IDX
    # if abs(target-current) > self.max_lataccel_err and self.sim.step_idx>120: # give it at least 2s
    #   terminated = True

    if terminated:
      info = self.sim.compute_cost()

      # make plot of episode
      if self.eval:
        fig, ax = plt.subplots(2, figsize=(12, 14), constrained_layout=True)
        canvas = fig.canvas
        self.sim.plot_history(ax, full=False)
        canvas.draw()  # draw the canvas, cache the renderer
        image = canvas_to_img(canvas)
        info.update({"plot": image})
        plt.close("all")

    return obs, reward, terminated, truncated, info

  def reset(
    self,
    seed: Optional[int] = None,
    options: Optional[dict] = None,
  ):
    super().reset(seed=seed, options=options)

    if self.tinyphysicsmodel is None:
      self.tinyphysicsmodel = TinyPhysicsModel("models/tinyphysics.onnx", debug=False)

    self.sim = TinyPhysicsSimulator(
      model=self.tinyphysicsmodel,
      data_path=str(random.choice(self.files)),
      controller=ZeroController(),
      debug=False
    )
    self.sim.reset()
    while self.sim.step_idx < CONTROL_START_IDX:
      self.sim.step()

    return self._get_obs(), {}


if __name__ == "__main__":
  env = TinyPhysicsEnv(eval=True, symlog_obs=False)
  controller = PIDController()

  obs, _ = env.reset()
  done = False
  return_ = 0
  while not done:
    action = controller.update(
      target_lataccel=obs[39],
      current_lataccel=obs[19],
      state=None,
      future_plan=None
    )
    obs, reward, terminated, truncated, _ = env.step(action)
    return_ += reward
    done = terminated or truncated

  print("mean reward:", return_/COST_END_IDX)

  fig, ax = plt.subplots(4, figsize=(12, 14), constrained_layout=True)
  env.sim.plot_history(ax)
  plt.show()

# TODO make sure to log lataccel and jerk cost separately
# as well as plots for the same N (8?) rollouts on the test set

# add option for mini obs to match the observation the PID uses
# so we can start by trying to match the PID performances
