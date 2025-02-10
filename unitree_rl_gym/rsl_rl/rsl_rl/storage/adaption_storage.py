import torch

from .rollout_storage import RolloutStorage
from rsl_rl.utils import split_and_pad_trajectories

class AdaptStorage(RolloutStorage):
    class Transition(RolloutStorage.Transition):
        def __init__(self):
            super().__init__()
            self.rma_observations = None

    def __init__(self, num_envs, num_transitions_per_env, obs_shape, privileged_obs_shape, rma_obs_shape, actions_shape, device='cpu'):
        super().__init__(num_envs, num_transitions_per_env, obs_shape, privileged_obs_shape, actions_shape, device)

        self.saved_hidden_states_r = None

        self.rma_obs_shape = rma_obs_shape
        self.rma_observations = torch.zeros(num_transitions_per_env, num_envs, *rma_obs_shape, device=self.device)

    def add_transitions(self, transition: Transition):
        self.rma_observations[self.step].copy_(transition.rma_observations)
        super().add_transitions(transition)

    def _save_hidden_states(self, hidden_states):
        if hidden_states is None or hidden_states==(None, None, None, None):
            return
        # make a tuple out of GRU hidden state sto match the LSTM format
        hid_a = hidden_states[0] if isinstance(hidden_states[0], tuple) else (hidden_states[0],)
        hid_r = hidden_states[1] if isinstance(hidden_states[1], tuple) else (hidden_states[1],)
        hid_a1 = hidden_states[2] if isinstance(hidden_states[2], tuple) else (hidden_states[2],)
        hid_r1 = hidden_states[3] if isinstance(hidden_states[3], tuple) else (hidden_states[3],)

        # initialize if needed 
        if self.saved_hidden_states_a is None:
            self.saved_hidden_states_a = [torch.zeros(self.observations.shape[0], *hid_a[i].shape, device=self.device) for i in range(len(hid_a))]
            self.saved_hidden_states_r = [torch.zeros(self.observations.shape[0], *hid_r[i].shape, device=self.device) for i in range(len(hid_r))]
            self.saved_hidden_states_a1 = [torch.zeros(self.observations.shape[0], *hid_a1[i].shape, device=self.device) for i in range(len(hid_a1))]
            self.saved_hidden_states_r1 = [torch.zeros(self.observations.shape[0], *hid_r1[i].shape, device=self.device) for i in range(len(hid_r1))]
        # copy the states
        for i in range(len(hid_a)):
            self.saved_hidden_states_a[i][self.step].copy_(hid_a[i])
            self.saved_hidden_states_r[i][self.step].copy_(hid_r[i])
            self.saved_hidden_states_a1[i][self.step].copy_(hid_a1[i])
            self.saved_hidden_states_r1[i][self.step].copy_(hid_r1[i])
                
    # for RNNs only
    def reccurent_mini_batch_generator(self, num_mini_batches, num_epochs=8):

        padded_obs_trajectories, trajectory_masks = split_and_pad_trajectories(self.observations, self.dones)
        if self.privileged_observations is not None: 
            padded_critic_obs_trajectories, _ = split_and_pad_trajectories(self.privileged_observations, self.dones)
        else: 
            padded_critic_obs_trajectories = padded_obs_trajectories

        if self.rma_observations is not None:
            padded_rma_obs_trajectories, _ = split_and_pad_trajectories(self.rma_observations, self.dones)
        else:
            padded_rma_obs_trajectories = padded_obs_trajectories

        mini_batch_size = self.num_envs // num_mini_batches
        for ep in range(num_epochs):
            first_traj = 0
            for i in range(num_mini_batches):
                start = i*mini_batch_size
                stop = (i+1)*mini_batch_size

                dones = self.dones.squeeze(-1)
                last_was_done = torch.zeros_like(dones, dtype=torch.bool)
                last_was_done[1:] = dones[:-1]
                last_was_done[0] = True
                trajectories_batch_size = torch.sum(last_was_done[:, start:stop])
                last_traj = first_traj + trajectories_batch_size
                
                masks_batch = trajectory_masks[:, first_traj:last_traj]
                obs_batch = padded_obs_trajectories[:, first_traj:last_traj]
                rma_obs_batch = padded_rma_obs_trajectories[:, first_traj:last_traj]
                critic_obs_batch = padded_critic_obs_trajectories[:, first_traj:last_traj]

                actions_batch = self.actions[:, start:stop]
                old_mu_batch = self.mu[:, start:stop]
                old_sigma_batch = self.sigma[:, start:stop]
                returns_batch = self.returns[:, start:stop]
                advantages_batch = self.advantages[:, start:stop]
                values_batch = self.values[:, start:stop]
                old_actions_log_prob_batch = self.actions_log_prob[:, start:stop]

                # reshape to [num_envs, time, num layers, hidden dim] (original shape: [time, num_layers, num_envs, hidden_dim])
                # then take only time steps after dones (flattens num envs and time dimensions),
                # take a batch of trajectories and finally reshape back to [num_layers, batch, hidden_dim]
                last_was_done = last_was_done.permute(1, 0)
                hid_a_batch = [ saved_hidden_states.permute(2, 0, 1, 3)[last_was_done][first_traj:last_traj].transpose(1, 0).contiguous()
                                for saved_hidden_states in self.saved_hidden_states_a ]
                hid_r_batch = [ saved_hidden_states.permute(2, 0, 1, 3)[last_was_done][first_traj:last_traj].transpose(1, 0).contiguous()
                                for saved_hidden_states in self.saved_hidden_states_r ]
                hid_a1_batch = [ saved_hidden_states.permute(2, 0, 1, 3)[last_was_done][first_traj:last_traj].transpose(1, 0).contiguous()
                                for saved_hidden_states in self.saved_hidden_states_a1 ]
                hid_r1_batch = [ saved_hidden_states.permute(2, 0, 1, 3)[last_was_done][first_traj:last_traj].transpose(1, 0).contiguous()
                                for saved_hidden_states in self.saved_hidden_states_r1 ]

                # remove the tuple for GRU
                hid_a_batch = hid_a_batch[0] if len(hid_a_batch)==1 else hid_a_batch
                hid_r_batch = hid_r_batch[0] if len(hid_r_batch)==1 else hid_r_batch
                hid_a1_batch = hid_a1_batch[0] if len(hid_a1_batch)==1 else hid_a1_batch
                hid_r1_batch = hid_r1_batch[0] if len(hid_r1_batch)==1 else hid_r1_batch

                yield obs_batch, critic_obs_batch, rma_obs_batch, actions_batch, values_batch, advantages_batch, returns_batch, \
                       old_actions_log_prob_batch, old_mu_batch, old_sigma_batch, (hid_a_batch,hid_r_batch,hid_a1_batch,hid_r1_batch), masks_batch
                
                first_traj = last_traj