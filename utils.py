import random

import numpy as np
import torch

import wandb


def seed_everything(seed):
  random.seed(seed)
  np.random.seed(seed)
  torch.manual_seed(seed)


def upload_file_to_artifacts(pth, artifact_name, artifact_type):
  artifact = wandb.Artifact(artifact_name, type=artifact_type)
  artifact.add_file(pth)
  wandb.log_artifact(artifact)


def canvas_to_img(canvas):
  image_flat = np.frombuffer(canvas.tostring_rgb(), dtype="uint8")
  image = image_flat.reshape(*reversed(canvas.get_width_height()), 3)
  return image
