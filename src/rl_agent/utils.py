# utils.py
import torch
import numpy as np
import random
from src.rl_agent.models import GaussianPolicyCNNActuators, QNetworkCNNActuators
from torch.optim.adam import Adam
from collections import deque


class DelayedMDP:
    def __init__(self, delay):
        self.delay = delay

        # for reward assignment
        self.action_list = deque(maxlen=delay+1)
        self.state_list = deque(maxlen=delay+1)
        self.next_state_list = deque(maxlen=delay+1)
        self.next_action_list = deque(maxlen=delay+1)
        # Reward list only used for model or for sum_rewards
        self.reward_list = deque(maxlen=delay+1)

    def check_update_possibility(self):
        """
        Checks that action list (and all the lists by the same rule)
        have enough information to take into account the delay
        """
        return len(self.action_list) >= (self.delay + 1)

    def save(self, s, a, s_next, r=None):

        self.action_list.append(a)
        self.state_list.append(s)
        self.next_state_list.append(s_next)
        if r is not None:
            self.reward_list.append(r)

    def credit_assignment(self):
        return self.state_list[0], self.action_list[0], self.next_state_list[-1]


class ReplayMemory:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.position = 0
        self.precision = torch.float32

    def push(self, state, action, reward, next_state):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)

        self.buffer[self.position] = (torch.tensor(state,  dtype=self.precision),
                                      torch.tensor(action,  dtype=self.precision),
                                      torch.tensor(reward,  dtype=self.precision),
                                      torch.tensor(next_state, dtype=self.precision))
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state = map(torch.stack, zip(*batch))
        return state, action, reward, next_state

    def __len__(self):
        return len(self.buffer)

    def reset(self):
        self.buffer = []
        self.position = 0


def soft_update(target, source, tau):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)


def hard_update(target, source):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(param.data)


def initialise(device,
               lr_alpha,
               automatic_entropy_tuning,
               alpha,
               state_channels,
               lr,
               mask_valid_actuators,
               initialise_last_layer_near_zero,
               initialize_last_layer_zero,
               num_layers_actor,
               num_layers_critic,
               two_output_critic,
               two_output_actor,
               agent_type,
               entropy_factor
               ):

    policy_target = None
    feature_extractor = None
    if agent_type == "td3":
        raise NotImplementedError
    elif agent_type == "sac":
        # Models
        policy = GaussianPolicyCNNActuators(num_state_channels=state_channels,
                                            mask_valid_actuators=mask_valid_actuators,
                                            num_layers=num_layers_actor,
                                            initialise_last_layer_near_zero=initialise_last_layer_near_zero,
                                            initialize_last_layer_zero=initialize_last_layer_zero,
                                            two_output_actor=two_output_actor).to(device)
        critic = QNetworkCNNActuators(num_state_channels=state_channels,
                                      num_layers=num_layers_critic,
                                      mask_valid_actuators=mask_valid_actuators,
                                      two_output_critic=two_output_critic,
                                      two_output_actor=two_output_actor).to(device)
        critic_target = QNetworkCNNActuators(num_state_channels=state_channels,
                                             num_layers=num_layers_critic,
                                             mask_valid_actuators=mask_valid_actuators,
                                             two_output_critic=two_output_critic,
                                             two_output_actor=two_output_actor).to(device)
        hard_update(critic, critic_target)
        optim = Adam

        # actor optim
        policy_optim = optim(policy.parameters(),
                             lr=lr)

        critic_optim = optim(critic.parameters(),
                             lr=lr)

        # Alpha
        if automatic_entropy_tuning:
            target_entropy = -torch.prod(
                torch.FloatTensor(mask_valid_actuators[mask_valid_actuators == 1].shape).to(device)).item() * entropy_factor
            log_alpha = torch.tensor(np.log(alpha), requires_grad=True, device=device, dtype=torch.float32)

            alpha_optim = Adam([log_alpha], lr=lr_alpha)
        else:
            target_entropy, log_alpha, alpha_optim = None, None, None
    else:
        raise NotImplementedError

    return target_entropy, log_alpha, alpha_optim,\
           policy, policy_target, critic, critic_target,\
           policy_optim, critic_optim, feature_extractor
