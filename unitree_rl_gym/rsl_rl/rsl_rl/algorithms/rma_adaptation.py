import torch
import torch.nn as nn
import torch.optim as optim

from rsl_rl.modules.actor_critic_recurrent_rma import ActorCriticRecurrentRMA, ActorCriticRecurrentAdaption
from rsl_rl.storage.adaption_storage import AdaptStorage
from rsl_rl.algorithms.ppo import PPO

class RMAAdaptation(PPO):
    actor_critic: ActorCriticRecurrentAdaption
    teacher_actor_critic: ActorCriticRecurrentRMA
    def __init__(self,
                 actor_critic,
                 teacher_actor_critic,
                 num_learning_epochs=1,
                 num_mini_batches=1,
                 clip_param=0.2,
                 gamma=0.998,
                 lam=0.95,
                 value_loss_coef=1.0,
                 entropy_coef=0.0,
                 learning_rate=1e-3,
                 max_grad_norm=1.0,
                 use_clipped_value_loss=True,
                 schedule="fixed",
                 desired_kl=0.01,
                 device='cpu',
                 ):
        super().__init__(actor_critic, num_learning_epochs, num_mini_batches, clip_param, gamma, lam, value_loss_coef, entropy_coef, learning_rate, max_grad_norm, use_clipped_value_loss, schedule, desired_kl, device)

        self.transition = AdaptStorage.Transition()

        self.teacher_actor_critic = teacher_actor_critic
        self.teacher_actor_critic.to(self.device)
        self.teacher_actor_critic.eval()

        self.optimizer = optim.Adam(self.actor_critic.actor.parameters(), lr=learning_rate)

    def init_storage(self, num_envs, num_transitions_per_env, actor_obs_shape, critic_obs_shape, rma_obs_shape, action_shape):
        self.storage = AdaptStorage(num_envs, num_transitions_per_env, actor_obs_shape, critic_obs_shape, rma_obs_shape, action_shape, self.device)

    def act(self, obs, critic_obs, rma_obs):
        if self.actor_critic.is_recurrent:
            tmp1 = self.actor_critic.get_hidden_states()
            tmp2 = self.teacher_actor_critic.get_hidden_states()
            self.transition.hidden_states = (tmp1[0],tmp1[1],tmp2[0],tmp2[1])
        self.transition.actions = self.actor_critic.act(obs, obs).detach()
        _ = self.teacher_actor_critic.act(obs, rma_obs).detach()
        
        self.transition.values = self.actor_critic.evaluate(critic_obs).detach()
        self.transition.actions_log_prob = self.actor_critic.get_actions_log_prob(self.transition.actions).detach()
        self.transition.action_mean = self.actor_critic.action_mean.detach()
        self.transition.action_sigma = self.actor_critic.action_std.detach()
        # need to record obs and critic_obs before env.step()
        self.transition.observations = obs
        self.transition.critic_observations = critic_obs
        self.transition.rma_observations = rma_obs
        return self.transition.actions

    def update(self):
        mean_encoder_loss = 0
        mean_action_loss = 0

        generator = self.storage.reccurent_mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)
        for obs_batch, critic_obs_batch, rma_obs_batch, actions_batch, target_values_batch, advantages_batch, returns_batch, old_actions_log_prob_batch, old_mu_batch, old_sigma_batch, hid_states_batch, masks_batch in generator:

                # TODO -------------------------------------------------------
                # Please calculate the encoder loss and action loss
                # The student module is self.actor_critic
                # The teacher module is self.teacher_actor_critic
                # You can get student_latent and teacher_latent via ActorCriticEncoder.encode
                # You can get student_action and teacher_action via ActorCriticEncoder.act_inference
                # The encoder loss is set to the mean square loss of student and teacher latents.
                # The action loss is set to the mean square loss of student and teacher actions.
                student_latent = self.actor_critic.encode(obs_batch, masks=masks_batch, hidden_states=hid_states_batch[1])
                student_action = self.actor_critic.act_inference(obs_batch, obs_batch, masks=masks_batch, hidden_states_obs=hid_states_batch[0], hidden_states_rma=hid_states_batch[1])
                with torch.no_grad():
                    teacher_latent = self.teacher_actor_critic.encode(rma_obs_batch, masks=masks_batch, hidden_states=hid_states_batch[3])
                    teacher_action = self.teacher_actor_critic.act_inference(obs_batch, rma_obs_batch, masks=masks_batch, hidden_states_obs=hid_states_batch[2], hidden_states_rma=hid_states_batch[3])

                encoder_loss = torch.mean((student_latent - teacher_latent)**2)
                action_loss = torch.mean((student_action - teacher_action)**2)
                # ------------------------------------------------------------

                loss = encoder_loss*0.1 + action_loss

                # Gradient step
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.actor_critic.actor.parameters(), self.max_grad_norm)
                self.optimizer.step()

                mean_encoder_loss += encoder_loss.item()
                mean_action_loss += action_loss.item()

        num_updates = self.num_learning_epochs * self.num_mini_batches
        mean_encoder_loss /= num_updates
        mean_action_loss /= num_updates
        self.storage.clear()

        return mean_encoder_loss, mean_action_loss