"""Portable policy inference: plain torch MLPs + the two obs normalizers.

Nothing here imports SB3 or safety_sb3. The actor MLPs were extracted at build
time (``build/extract_weights.py``) into plain ``nn.Sequential`` state_dicts;
the normalizers are plain arrays. Runtime deps: torch + numpy only.
"""

from __future__ import annotations

import os

import numpy as np
import torch
import torch.nn as nn

_ASSETS = os.path.join(os.path.dirname(__file__), "assets", "extracted")


def _actor_mlp(in_dim: int) -> nn.Sequential:
  """The architecture both trained policies share:
  policy_net [in->512->256->128] with Tanh, then the linear action head ->12.
  Matches SB3 ``_predict(deterministic=True)`` (Gaussian mean, no squashing)."""
  return nn.Sequential(
    nn.Linear(in_dim, 512), nn.Tanh(),
    nn.Linear(512, 256), nn.Tanh(),
    nn.Linear(256, 128), nn.Tanh(),
    nn.Linear(128, 12),
  )


def _torch_load(path, device):
  """torch.load with weights_only=True where supported (torch >= 1.13); older
  torch (e.g. 1.10 in the robot's go2_sdk env) rejects the keyword."""
  return torch.load(path, map_location=device, weights_only=True)


def load_actor(name: str, in_dim: int, device: str = "cpu") -> nn.Sequential:
  net = _actor_mlp(in_dim)
  sd = _torch_load(os.path.join(_ASSETS, name), device)
  net.load_state_dict(sd)
  net.eval().to(device)
  for p in net.parameters():
    p.requires_grad_(False)
  return net


def _critic_mlp(in_dim: int) -> nn.Sequential:
  """The reach-avoid twin's state-value head: value_net [in->512->256->128]
  with Tanh, then the linear scalar head ->1. Matches SB3
  ``policy.predict_values`` (a FlattenExtractor makes the features extractor a
  no-op for the 1-D obs)."""
  return nn.Sequential(
    nn.Linear(in_dim, 512), nn.Tanh(),
    nn.Linear(512, 256), nn.Tanh(),
    nn.Linear(256, 128), nn.Tanh(),
    nn.Linear(128, 1),
  )


def load_critic(name: str, in_dim: int, device: str = "cpu") -> nn.Sequential:
  net = _critic_mlp(in_dim)
  sd = _torch_load(os.path.join(_ASSETS, name), device)
  net.load_state_dict(sd)
  net.eval().to(device)
  for p in net.parameters():
    p.requires_grad_(False)
  return net


class WalkerNorm:
  """SB3 VecNormalize statistics as a plain callable.

      clip((o - mean) / sqrt(var + epsilon), -clip, clip)
  """

  def __init__(self, device: str = "cpu"):
    d = np.load(os.path.join(_ASSETS, "walker_norm.npz"))
    self.mean = torch.as_tensor(d["mean"], dtype=torch.float32, device=device)
    self.var = torch.as_tensor(d["var"], dtype=torch.float32, device=device)
    self.eps = float(d["epsilon"])
    self.clip = float(d["clip"])

  def __call__(self, obs: torch.Tensor) -> torch.Tensor:
    return torch.clamp((obs - self.mean) / torch.sqrt(self.var + self.eps),
                       -self.clip, self.clip)


class JumpNorm:
  """safety_sb3 TensorNormalize statistics (see eval/policies.load_twin).

      clip((o - obs_mean) / sqrt(obs_var + 1e-8), -10, 10)
  """

  def __init__(self, width_tag: str = "w30", device: str = "cpu"):
    st = _torch_load(os.path.join(_ASSETS, f"jump_norm_{width_tag}.pt"), device)
    self.mean = st["obs_mean"].to(device).float()
    self.var = st["obs_var"].to(device).float()
    self.eps = float(st.get("epsilon", torch.tensor(1e-8)))
    self.clip = float(st.get("clip", torch.tensor(10.0)))

  def __call__(self, obs: torch.Tensor) -> torch.Tensor:
    return torch.clamp((obs - self.mean) / torch.sqrt(self.var + self.eps),
                       -self.clip, self.clip)
