# models.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal


def weights_init_last_layer_policy_(mean_out,
                                    log_std_out=None,
                                    initialize_last_layer_zero=False,
                                    initialise_last_layer_near_zero=False):
    """
    If we initalize last layer(s) to 0 or close to 0.
    This is useful if we do a correction on top of the linear integrator.
    """

    if initialize_last_layer_zero:

        with torch.no_grad():
            mean_out.weight = torch.nn.Parameter(torch.zeros_like(mean_out.weight),
                                                      requires_grad=True)
            if log_std_out is not None:
                log_std_out.weight = torch.nn.Parameter(torch.zeros_like(log_std_out.weight),
                                                             requires_grad=True)
                torch.nn.init.constant_(log_std_out.bias, -1)
    elif initialise_last_layer_near_zero:

        with torch.no_grad():
            mean_out.weight = torch.nn.Parameter(torch.zeros_like(mean_out.weight),
                                                      requires_grad=True)
            torch.nn.init.xavier_uniform_(mean_out.weight,
                                          gain=1e-4)
            if log_std_out is not None:
                log_std_out.weight = torch.nn.Parameter(torch.zeros_like(log_std_out.weight),
                                                        requires_grad=True)

                torch.nn.init.xavier_uniform_(log_std_out.weight,
                                              gain=1e-4)
                torch.nn.init.constant_(log_std_out.bias, -1)

    return mean_out, log_std_out


def weights_init_(m, gain=1):
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        torch.nn.init.xavier_uniform_(m.weight, gain=gain)
        if hasattr(m, 'bias') and m.bias is not None:
            torch.nn.init.constant_(m.bias, 0)


class QNetworkCNNActuators(nn.Module):
    """
    SAC or TD3 (Q-nets)
    """
    def __init__(self,
                 num_state_channels=4,
                 mask_valid_actuators=None,
                 hidden_dim=64,
                 num_layers=2,
                 kernel_size=3,
                 activation="relu",
                 two_output_critic=False,
                 two_output_actor=False):
        super(QNetworkCNNActuators, self).__init__()

        output_channels = 2 if two_output_critic else 1
        actor_channels = 2 if two_output_actor else 1
        # Layers
        self.q1_list = nn.ModuleList()
        self.q2_list = nn.ModuleList()
        self.q1_list.append(nn.Conv2d(num_state_channels + actor_channels, hidden_dim, kernel_size=kernel_size, stride=1, padding=1))
        self.q2_list.append(nn.Conv2d(num_state_channels + actor_channels, hidden_dim, kernel_size=kernel_size, stride=1, padding=1))

        for i in range(num_layers - 1):
            self.q1_list.append(nn.Conv2d(hidden_dim, hidden_dim, kernel_size=kernel_size, stride=1, padding=1))
            self.q2_list.append(nn.Conv2d(hidden_dim, hidden_dim, kernel_size=kernel_size, stride=1, padding=1))

        self.q1_list.append(nn.Conv2d(hidden_dim, output_channels, kernel_size=kernel_size, stride=1, padding=1))
        self.q2_list.append(nn.Conv2d(hidden_dim, output_channels, kernel_size=kernel_size, stride=1, padding=1))
        # Activations
        if activation == "relu":
            self.activation = F.relu
        elif activation == "leaky_relu":
            self.activation = F.leaky_relu
        else:
            raise NotImplementedError

        # Weight initialization
        self.apply(weights_init_)

        # Mask valid actuators
        self.mask_valid_actuators = nn.Parameter(
            torch.tensor(mask_valid_actuators.reshape(-1, mask_valid_actuators.shape[0], mask_valid_actuators.shape[1]),
                         dtype=torch.float32), requires_grad=False)

    def forward(self, state, action):
        """
        Recieve as input state and action in 2D format.
        :param state: (Bsz, N, Na, Na).
        :param action: (Bsz, 1, Na, Na).
        :return: Q_1(s,a):(Bsz, 1, Na, Na), Q_2(s,a): (Bsz, 1, Na, Na).
        """

        x = torch.cat([state, action], 1)

        x1 = self.q1_list[0](x)
        x2 = self.q2_list[0](x)

        x1 = self.activation(x1)
        x2 = self.activation(x2)

        # Layers
        for i in range(1, len(self.q1_list)-1):
            x1 = self.q1_list[i](x1)
            x2 = self.q2_list[i](x2)

            x1 = self.activation(x1)
            x2 = self.activation(x2)

        x1 = self.q1_list[-1](x1)
        x2 = self.q2_list[-1](x2)

        x1 = torch.mul(x1, self.mask_valid_actuators)
        x2 = torch.mul(x2, self.mask_valid_actuators)

        return x1, x2

    def to(self, device):
        return super(QNetworkCNNActuators, self).to(device)


class GaussianPolicyCNNActuators(nn.Module):
    """
    Soft Actor Critic (policy)
    """
    def __init__(self,
                 num_state_channels=4,
                 mask_valid_actuators=None,
                 hidden_dim=64,
                 num_layers=2,
                 activation="relu",
                 kernel_size=3,
                 initialize_last_layer_zero=False,
                 initialise_last_layer_near_zero=False,
                 eps=1e-5,
                 log_sig_max=2,
                 log_sig_min=-20,
                 two_output_actor=False
                 ):

        super(GaussianPolicyCNNActuators, self).__init__()

        # Layers
        self.pi_list = nn.ModuleList()
        self.pi_list.append(
            torch.nn.Conv2d(num_state_channels, hidden_dim, kernel_size=kernel_size, stride=1, padding=1))

        for i in range(num_layers - 1):
            self.pi_list.append(nn.Conv2d(hidden_dim, hidden_dim, kernel_size=kernel_size, stride=1, padding=1))

        output_channels = 2 if two_output_actor else 1

        self.mean_out = torch.nn.Conv2d(hidden_dim, output_channels, kernel_size=kernel_size, stride=1, padding=1)
        self.log_std_out = torch.nn.Conv2d(hidden_dim, output_channels, kernel_size=kernel_size, stride=1, padding=1)

        # Activations
        if activation == "relu":
            self.activation = F.relu
        elif activation == "leaky_relu":
            self.activation = F.leaky_relu
        else:
            raise NotImplementedError

        # Weight initialization
        self.apply(weights_init_)

        self.mean_out, self.log_std_out = weights_init_last_layer_policy_(self.mean_out, self.log_std_out,
                                                                          initialize_last_layer_zero,
                                                                          initialise_last_layer_near_zero)

        # Mask valid actuators
        self.mask_valid_actuators = nn.Parameter(
            torch.tensor(mask_valid_actuators.reshape(-1, mask_valid_actuators.shape[0], mask_valid_actuators.shape[1]),
                         dtype=torch.float32), requires_grad=False)
        self.LOG_SIG_MAX = log_sig_max
        self.LOG_SIG_MIN = log_sig_min
        self.eps = eps

    def forward(self, state, evaluation=False, sample_for_loop=False):
        """
        Recieve as input state in 2D format.
        :param state: (Bsz, N, Na, Na).
        :param evaluation: if we are exploring or exploiting.
        :param sample_for_loop: if we are in the loop or updating.
        :return: mean (Bsz, 1, Na, Na), log_std (Bsz, 1, Na, Na).
        """

        if sample_for_loop:

            # For the loop we do less operation and we do not track the gradients.
            with torch.no_grad():
                # Layers
                for i in range(len(self.pi_list)):
                    # print(i, state.dtype)
                    state = self.activation(self.pi_list[i](state))

                mean, log_std = self.mean_out(state), self.log_std_out(state)
                # log_std = (self.LOG_SIG_MIN + 0.5 * (self.LOG_SIG_MAX - self.LOG_SIG_MIN) * (log_std + 1))
                log_std = torch.clamp(log_std, min=self.LOG_SIG_MIN, max=self.LOG_SIG_MAX)

                if evaluation:
                    mean = torch.tanh(mean)
                    mean = torch.mul(mean, self.mask_valid_actuators)
                    return mean.squeeze(0)
                else:
                    std = log_std.exp()
                    normal = Normal(mean, std)
                    x_t = normal.rsample()
                    y_t = torch.tanh(x_t)
                    a = torch.mul(y_t, self.mask_valid_actuators)
                    return a.squeeze(0)
        else:
            for i in range(len(self.pi_list)):
                state = self.activation(self.pi_list[i](state))

            mean, log_std = self.mean_out(state), self.log_std_out(state)
            log_std = torch.clamp(log_std, min=self.LOG_SIG_MIN, max=self.LOG_SIG_MAX)
            std = log_std.exp()

            normal = Normal(mean, std)

            x_t = normal.rsample()
            y_t = torch.tanh(x_t)
            mean = torch.tanh(mean)
            log_prob = normal.log_prob(x_t)

            log_prob -= torch.log((1 - y_t.pow(2).clamp(min=0, max=1)) + self.eps)

            log_prob = torch.mul(log_prob, self.mask_valid_actuators)
            action = torch.mul(y_t, self.mask_valid_actuators)

            return action, log_prob, mean

    def to(self, device):
        return super(GaussianPolicyCNNActuators, self).to(device)


class Td3PolicyCNNActuators(nn.Module):
    """
    Twin delayed deep deterministic policy gradient (policy)
    """
    def __init__(self,
                 num_state_channels=4,
                 mask_valid_actuators=None,
                 hidden_dim=64,
                 num_layers=2,
                 activation="relu",
                 kernel_size=3,
                 initialize_last_layer_zero=False,
                 initialise_last_layer_near_zero=False,
                 eps=1e-5,
                 std=0.3,
                 two_output_actor=False
                 ):

        super(Td3PolicyCNNActuators, self).__init__()

        # Layers
        self.pi_list = nn.ModuleList()
        self.pi_list.append(
            torch.nn.Conv2d(num_state_channels, hidden_dim, kernel_size=kernel_size, stride=1, padding=1))

        for i in range(num_layers - 1):
            self.pi_list.append(nn.Conv2d(hidden_dim, hidden_dim, kernel_size=kernel_size, stride=1, padding=1))

        output_channels = 2 if two_output_actor else 1

        self.mean_out = torch.nn.Conv2d(hidden_dim, output_channels, kernel_size=kernel_size, stride=1, padding=1)

        # Activations
        if activation == "relu":
            self.activation = F.relu
        elif activation == "leaky_relu":
            self.activation = F.leaky_relu
        else:
            raise NotImplementedError

        # Weight initialization
        self.apply(weights_init_)

        self.mean_out, _ = weights_init_last_layer_policy_(self.mean_out,
                                                           initialize_last_layer_zero=initialize_last_layer_zero,
                                                           initialise_last_layer_near_zero=initialise_last_layer_near_zero)

        # Mask valid actuators
        self.mask_valid_actuators = nn.Parameter(
            torch.tensor(mask_valid_actuators.reshape(-1, mask_valid_actuators.shape[0], mask_valid_actuators.shape[1]),
                         dtype=torch.float32), requires_grad=False)
        self.ACTION_MIN = -1.0
        self.ACTION_MAX = 1.0
        self.STD = std
        self.eps = eps

    def forward(self, state, evaluation=False, sample_for_loop=False):
        """
        Recieve as input state in 2D format.
        :param state: (Bsz, N, Na, Na).
        :param evaluation: if we are exploring or exploiting.
        :param sample_for_loop: if we are in the loop or updating.
        :return: mean (Bsz, 1, Na, Na), log_std (Bsz, 1, Na, Na).
        """

        if sample_for_loop:

            # For the loop we do less operation and we do not track the gradients.
            with torch.no_grad():
                # Layers
                for i in range(len(self.pi_list)):
                    state = self.activation(self.pi_list[i](state))
                mean = self.mean_out(state)

                if evaluation:
                    # TODO check
                    mean = torch.clamp(mean, min=self.ACTION_MIN, max=self.ACTION_MAX)
                    mean = torch.mul(mean, self.mask_valid_actuators)
                    return mean.squeeze(0)
                else:

                    normal = Normal(mean, self.STD)

                    x_t = normal.rsample()
                    y_t = torch.clamp(x_t, min=self.ACTION_MIN, max=self.ACTION_MAX)
                    a = torch.mul(y_t, self.mask_valid_actuators)
                    return a.squeeze(0)
        else:
            for i in range(len(self.pi_list)):
                state = self.activation(self.pi_list[i](state))
            mean = self.mean_out(state)

            normal = Normal(mean, self.STD)
            x_t = normal.rsample()
            y_t = torch.clamp(x_t, min=self.ACTION_MIN, max=self.ACTION_MAX)
            action = torch.mul(y_t, self.mask_valid_actuators)

            return action, mean

    def to(self, device):
        return super(Td3PolicyCNNActuators, self).to(device)