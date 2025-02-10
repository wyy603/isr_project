from isaacgym.torch_utils import *
from isaacgym import gymtorch, gymapi
import torch

from legged_gym.envs.h1_2.h1_2_env import H1_2Robot
from collections import deque


class H12RMAEnv(H1_2Robot):

    def __init__(self, cfg, sim_params, physics_engine, sim_device, headless):
        super().__init__(cfg, sim_params, physics_engine, sim_device, headless)
        self.num_rma_obs = cfg.env.num_rma_obs

    def _init_buffers(self):
        super()._init_buffers()

    def compute_observations(self):
        super().compute_observations()
        self.rma_obs_buf = self.privileged_obs_buf.clone()

    def step(self, actions):
        """ Apply actions, simulate, call self.post_physics_step()

        Args:
            actions (torch.Tensor): Tensor of shape (num_envs, num_actions_per_env)
        """
        super().step(actions)
        clip_obs = self.cfg.normalization.clip_observations
        self.rma_obs_buf = torch.clip(self.rma_obs_buf, -clip_obs, clip_obs)
        return self.obs_buf, self.privileged_obs_buf, self.rma_obs_buf, self.rew_buf, self.reset_buf, self.extras

    def reset_idx(self, env_ids):
        super().reset_idx(env_ids)

    def reset(self):
        """ Reset all robots"""
        self.reset_idx(torch.arange(self.num_envs, device=self.device))
        obs, privileged_obs, _, _, _, _ = self.step(torch.zeros(self.num_envs, self.num_actions, device=self.device, requires_grad=False))
        return obs, privileged_obs

    def get_rma_observations(self):
        return self.rma_obs_buf