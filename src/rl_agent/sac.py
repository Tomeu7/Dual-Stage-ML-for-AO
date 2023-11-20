# sac.py
import torch
import torch.nn as nn
import numpy as np
import time
from src.rl_agent.utils import soft_update, initialise, ReplayMemory

class Sac:
    def __init__(self,
                 config_agent,
                 state_channels,
                 action_shape,
                 device,
                 mask_valid_actuators,
                 two_output_critic,
                 two_output_actor):

        self.device = "cuda:" + str(device)
        # Parameters state-action
        self.state_channels = state_channels
        self.action_shape = action_shape

        # Parameters agent
        self.capacity, self.batch_size, self.lr = config_agent['replay_capacity'], \
                                                  config_agent['batch_size'], \
                                                  config_agent['lr']
        self.lr_alpha = config_agent['lr_alpha']
        self.alpha = config_agent['alpha']
        self.target_update_interval = config_agent['target_update_interval']
        self.tau = config_agent['tau']
        self.automatic_entropy_tuning = config_agent['automatic_entropy_tuning']
        self.gamma = torch.tensor([config_agent['gamma']], dtype=torch.float32, requires_grad=False).to(self.device)

        # Other parameters
        self.update_simplified = config_agent['update_simplified']
        self.train_for_steps = config_agent['train_for_steps']
        self.train_every_steps = config_agent['train_every_steps']

        # Initializing SAC
        self.target_entropy, self.log_alpha, self.alpha_optim, \
            self.policy, _, self.critic, self.critic_target, \
            self.policy_optim, self.critic_optim, self.feature_extractor =\
            initialise(self.device, self.lr_alpha, self.automatic_entropy_tuning,
                       self.alpha, self.state_channels, self.lr, mask_valid_actuators,
                       config_agent['initialise_last_layer_near_zero'],
                       config_agent['initialize_last_layer_zero'],
                       config_agent['num_layers_actor'],
                       config_agent['num_layers_critic'],
                       two_output_critic=two_output_critic,
                       two_output_actor=two_output_actor,
                       agent_type=config_agent['agent_type'],
                       entropy_factor=config_agent['entroy_factor']
                       )

        # Replay buffer
        self.replay_buffer = ReplayMemory(capacity=self.capacity)
        # Counters
        self.num_updates = 0
        self.total_step = 0
        self.list_times_single_cycle = []
        self.print_every = config_agent['print_every']
        # Loss MSE
        self.mse_loss = nn.MSELoss()
        # For TT training
        self.two_output_critic = two_output_critic
        self.two_output_actor = two_output_actor

        print("----------------------------------------------")
        if self.feature_extractor is not None:
            print(self.feature_extractor)
        print(self.critic)
        print(self.policy)
        print("----------------------------------------------")

    @torch.no_grad()
    def get_tensors_from_memory(self):
        state_batch, action_batch, reward_batch, next_state_batch = \
            self.replay_buffer.sample(batch_size=self.batch_size)
        state_batch = state_batch.to(self.device)
        next_state_batch = next_state_batch.to(self.device)
        if self.two_output_actor:
            action_batch = action_batch.to(self.device)
        else:
            action_batch = action_batch.to(self.device).unsqueeze(1)
        if self.two_output_critic:
            reward_batch = reward_batch.to(self.device)
        else:
            reward_batch = reward_batch.to(self.device).unsqueeze(1)
        return state_batch, action_batch, reward_batch, next_state_batch

    ###################################################################################################################
    @torch.no_grad()
    def get_bellman_backup(self,
                           reward_batch,
                           next_state_batch,
                           next_state_action=None,
                           next_state_log_pi=None
                           ):

        if next_state_action is None and next_state_log_pi is None:
            next_state_action, next_state_log_pi, _ = self.policy(next_state_batch, False, False)

        qf1_next_target, qf2_next_target = self.critic_target(next_state_batch, next_state_action)

        if self.two_output_actor and not self.two_output_critic:
            next_state_log_pi = next_state_log_pi.mean(1, keepdim=True)
        elif self.two_output_critic and not self.two_output_actor:
            qf1_next_target = qf1_next_target.mean(1, keepdim=True)
            qf2_next_target = qf2_next_target.mean(1, keepdim=True)

        min_qf_next_target = torch.min(qf1_next_target,
                                       qf2_next_target) - self.alpha * next_state_log_pi

        next_q_value = reward_batch + self.gamma * min_qf_next_target

        return next_q_value

    def calculate_q_loss(self, qf1, qf2, next_q_value):
        """
        Calculates loss with Soft Bellman equation
        :param qf1: Q_1(s,a)
        :param qf2: Q_2(s,a)
        :param next_q_value: Bellman error
        :return: float total loss, float q1 loss, float q2 loss
        """
        qf1_loss = self.mse_loss(qf1, next_q_value)
        qf2_loss = self.mse_loss(qf2, next_q_value)
        qf_loss = qf1_loss + qf2_loss
        return qf_loss, qf1_loss, qf2_loss

    def update_critic(self,
                      state_batch,
                      action_batch,
                      reward_batch,
                      next_state_batch,
                      next_state_action=None,
                      next_state_log_pi=None
                      ):
        """
        Updates critic with Soft Bellman equation
        :param state_batch: the state batch extracted from memory
        :param action_batch: the action batch extracted from memory
        :param reward_batch: the reward batch extracted from memory
        :param next_state_batch: the next state batch batch extracted from memory
        :param next_state_action: if we sample it from outside the function
        :param next_state_log_pi: if we sample it from outside the function
        :return: float loss for Q1, float loss for Q2
        """

        next_q_value = self.get_bellman_backup(reward_batch,
                                               next_state_batch,
                                               next_state_action,
                                               next_state_log_pi
                                               )
        qf1, qf2 = self.critic(state_batch, action_batch)

        if self.two_output_critic and not self.two_output_actor:
            qf1 = qf1.mean(1, keepdim=True)
            qf2 = qf2.mean(1, keepdim=True)

        qf_loss, qf1_loss, qf2_loss = self.calculate_q_loss(qf1,
                                                            qf2,
                                                            next_q_value)
        # Default
        self.critic_optim.zero_grad()
        qf_loss.backward()
        self.critic_optim.step()

        return qf1_loss, qf2_loss

    ###################################################################################################################
    def calculate_policy_loss(self, log_pi, qf1_pi, qf2_pi):
        min_qf_pi = torch.min(qf1_pi, qf2_pi)
        policy_loss = torch.mean((self.alpha * log_pi) - min_qf_pi)
        # TODO why no mean
        return policy_loss

    def update_actor(self, state_batch, pi=None, log_pi=None):

        # policy accepts two bools, first if we are in evaluation mode, second if we are
        if pi is None or log_pi is None:
            pi, log_pi, _ = self.policy(state_batch, False, False)

        qf1_pi, qf2_pi = self.critic(state_batch, pi)

        if self.two_output_actor and not self.two_output_critic:
            log_pi = log_pi.mean(1, keepdim=True)
        elif self.two_output_critic and not self.two_output_actor:
            qf1_pi = qf1_pi.mean(1, keepdim=True)
            qf2_pi = qf2_pi.mean(1, keepdim=True)

        policy_loss = self.calculate_policy_loss(log_pi, qf1_pi, qf2_pi)

        self.policy_optim.zero_grad()
        policy_loss.backward()
        self.policy_optim.step()

        af_loss = self.update_alpha(log_pi)

        return af_loss, policy_loss

    ###################################################################################################################

    def update_alpha(self, log_pi):
        """
        Updates alpha
        :param log_pi: array, log probabilities of the current action
        + Notes:
        log_pi with CNN model -> (Bsz, 1, Nact, Nact)
        :return: float alpha loss, float alpha value
        """

        if self.automatic_entropy_tuning:
            log_pi = log_pi.reshape(self.batch_size, -1).sum(dim=-1, keepdim=True)
            multiplier = (log_pi + self.target_entropy).detach()
            alpha_loss = -(self.log_alpha * multiplier).mean()

            self.alpha_optim.zero_grad()
            alpha_loss.backward()
            self.alpha_optim.step()

            self.alpha = self.log_alpha.exp()
        else:
            alpha_loss = 0

        return alpha_loss

    def update_less_operations(self):
        state_batch, action_batch, reward_batch, next_state_batch = self.get_tensors_from_memory()

        next_state_action, next_state_log_pi, _ = self.policy(next_state_batch, False, False)

        # Policy
        af_loss, pf_loss = self.update_actor(next_state_batch, next_state_action, next_state_log_pi)

        # Critic
        qf1_loss, qf2_loss = self.update_critic(state_batch, action_batch, reward_batch, next_state_batch, next_state_action, next_state_log_pi)

        if self.num_updates % self.target_update_interval == 0:
            soft_update(self.critic_target, self.critic, self.tau)

        if (self.total_step + 1) % self.print_every == 0:
            if self.automatic_entropy_tuning:
                af_tlogs = self.alpha.clone().item()
            else:
                af_tlogs = torch.tensor(self.alpha).item()
            return qf1_loss.detach().cpu().numpy(),\
                   qf2_loss.detach().cpu().numpy(),\
                   pf_loss.detach().cpu().item(),\
                   af_loss,\
                   af_tlogs
        else:
            return 0, 0, 0, 0, 0

    def update(self):
        # Extract tensors
        state_batch, action_batch, reward_batch, next_state_batch = self.get_tensors_from_memory()

        qf1_loss, qf2_loss = self.update_critic(state_batch, action_batch, reward_batch, next_state_batch)

        # Update actor
        af_loss, pf_loss = self.update_actor(state_batch)
        if self.num_updates % self.target_update_interval == 0:
            soft_update(self.critic_target, self.critic, self.tau)

        if (self.total_step + 1) % self.print_every == 0:
            if self.automatic_entropy_tuning:
                af_tlogs = self.alpha.clone().item()
            else:
                af_tlogs = torch.tensor(self.alpha).item()
            return qf1_loss.detach().cpu().numpy(), \
                   qf2_loss.detach().cpu().numpy(), \
                   pf_loss.detach().cpu().item(), \
                   af_loss, \
                   af_tlogs
        else:
            return 0, 0, 0, 0, 0

    def select_action(self, s, evaluation):

        state = torch.FloatTensor(s).unsqueeze(0).to(device=self.device)
        action = self.policy.forward(state, evaluation, True)
        return action

    def update_replay_buffer(self, s, a, r, s_next):
        self.replay_buffer.push(s, a, r, s_next)

    def train(self, step):
        self.total_step = step
        qf1_loss, qf2_loss, pf_loss, af_tlogs, time_update = 0, 0, 0, 0, 0
        if len(self.replay_buffer) > self.batch_size:
            for i in range(self.train_for_steps):
                time_start_iter = time.time()

                if self.update_simplified:
                    qf1_loss, qf2_loss, pf_loss, af_loss, af_tlogs = self.update_less_operations()
                else:
                    qf1_loss, qf2_loss, pf_loss, af_loss, af_tlogs = self.update()

                # Record metrics
                self.list_times_single_cycle.append(time.time() - time_start_iter)

                self.num_updates += 1

            if (self.total_step + 1) % self.print_every == 0:
                array_times = np.array(self.list_times_single_cycle)
                time_update = array_times.mean()
                # Print metrics
                print("SacUpdater: send policy weights,"
                       " number of updates: {} - L_Q1: {:2f} - L_Q2: {:2f} - L_pi: {:2f} alpha {:2f} -"
                       " Max(iter_time) {:4f} - <iter_time> {:4f} - Min(iter_time) {:4f} - Len replay {}"
                       .format(str(self.num_updates), qf1_loss, qf2_loss, pf_loss, af_tlogs,
                               array_times.max(), time_update, array_times.min(), len(self.replay_buffer)))
                self.list_times_single_cycle.clear()

        return qf1_loss, pf_loss, af_tlogs, time_update

    def save_policy(self, file_name='policy_model_weights.pth'):
        torch.save(self.policy.state_dict(), file_name)

    def load_policy(self, file_name='policy_model_weights.pth'):
        self.policy.load_state_dict(torch.load(file_name))

    def reset_optimizer(self):
        from torch.optim.adam import Adam
        self.policy_optim = Adam(self.policy.parameters(),
                             lr=self.lr)

