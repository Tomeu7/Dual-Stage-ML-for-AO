from abc import ABC
import gym
from collections import deque, OrderedDict
import numpy as np
from shesha.supervisor.rlSupervisor import RlSupervisor as Supervisor
from shesha.util.utilities import load_config_from_file
import math
from gym import spaces
from src.env_methods.projectors import ProjectorCreator
import torch

class AoEnv(gym.Env, ABC):

    #            Initialization
    #
    #  __  __      _   _               _
    # |  \/  | ___| |_| |__   ___   __| |___
    # | |\/| |/ _ \ __| '_ \ / _ \ / _` / __|
    # | |  | |  __/ |_| | | | (_) | (_| \__ \
    # |_|  |_|\___|\__|_| |_|\___/ \__,_|___/

    def __init__(self,
                 config_env_rl,
                 parameter_file,
                 seed,
                 device,
                 pyr_gpu_ids=[0],
                 override_generate_phase_projectors=False):
        super(AoEnv, self).__init__()

        self.config_env_rl = config_env_rl
        self.device = device

        # For control TT
        self.control_tt = config_env_rl['control_tt']
        self.two_output_actor = config_env_rl['separate_tt_into_two_actions'] and config_env_rl['control_tt']
        self.a_tt_from_hddm = None
        self.a_hddm_from_tt = None
        self.a_hddm = None
        # For metrics of TT/HDDM
        self.reconstruction_for_reward = None

        # Loading Compass config
        fd_parameters = "data/parameter_files/"
        config_compass = load_config_from_file(fd_parameters + parameter_file)
        config_compass.p_loop.set_devices(pyr_gpu_ids)

        self.supervisor = Supervisor(config=config_compass,
                                     n_reverse_filtered_from_cmat=self.config_env_rl['n_reverse_filtered_from_cmat'],
                                     filter_commands=config_env_rl['filter_commands'],
                                     command_clip_value=config_env_rl['command_clip_value'],
                                     initial_seed=seed,
                                     which_modal_basis="Btt",
                                     mode=self.config_env_rl['mode'],
                                     device=device,
                                     control_tt=self.control_tt)

        # From supervisor for easy access
        self.command_shape = self.supervisor.command_shape  # always in 1D
        self.action_1d_shape = self.supervisor.action_1d_shape
        self.action_2d_shape = self.supervisor.action_2d_shape
        self.hddm_shape = self.supervisor.hddm_shape  # always in 1D
        self.tt_shape = self.supervisor.tt_shape  # always in 1D
        self.mask_valid_actuators = self.supervisor.mask_valid_actuators
        self.num_dm = self.supervisor.num_dm
        # Initializy the history
        self.s_dm_history = deque(maxlen=self.config_env_rl['number_of_previous_s_dm'])
        self.s_dm_residual_history = deque(maxlen=self.config_env_rl['number_of_previous_s_dm_residual'])
        self.s_dm_residual_rl_history = deque(maxlen=self.config_env_rl['number_of_previous_s_dm_residual_rl'])
        # tt state
        self.s_dm_residual_tt_history = deque(maxlen=self.config_env_rl['number_of_previous_s_dm_residual_tt'])
        self.s_dm_tt_history = deque(maxlen=self.config_env_rl['number_of_previous_s_dm_tt'])
        # for a reward
        self.a_for_reward_history = deque(maxlen=self.config_env_rl['number_of_previous_a_for_reward'])
        # keys
        self.list_of_keys_of_state = ['s_dm', 's_dm_residual', 's_dm_residual_rl', 'a_for_reward', 's_dm_residual_tt', 's_dm_tt']
        # Observation/action space
        self.state_size_channel_0, self.observation_shape,\
            self.observation_space, self.action_space = self.define_state_action_space()
        # Normalization
        self.norm_parameters = {"dm":
                                    {"mean": 0.0,
                                     "std": self.config_env_rl['dm_std']},
                                "dm_residual":
                                            {"mean": 0.0,
                                             "std": self.config_env_rl['dm_residual_std']}
                                }
        # Delayed assignment
        self.delayed_assignment = self.config_env_rl['delayed_assignment']
        # Defined for the state
        self.s_next_main = None
        # Some values for Unet
        self.override_generate_phase_projectors = override_generate_phase_projectors

        print("Parameter file {} Observation space {} Action space {} Compass device {}".format(
            parameter_file, self.observation_space.shape, self.action_space.shape, self.device))
        self.actuator2phase, self.phase2actuator = None, None
        self.actuatorhddm2phase, self.phase2actuatorhddm = None, None
        self.actuatortt2phase, self.phase2actuatortt = None, None
        self.actuatorhddm2phase_torch, self.phase2actuatorhddm_torch = None, None
        self.actuatortt2phase_torch, self.phase2actuatortt_torch = None, None
        if self.control_tt or self.override_generate_phase_projectors:
            self.actuator2phase, self.phase2actuator, \
            self.actuatorhddm2phase, self.phase2actuatorhddm, \
            self.actuatortt2phase, self.phase2actuatortt = self.create_projectors_of_phase(parameter_file)

            self.supervisor.actuatortt2phase = self.actuatortt2phase
            self.supervisor.phase2actuatorhddm = self.phase2actuatorhddm

            if self.device != -1:
                # add matrices to GPU
                self.actuator2phase, self.phase2actuator, \
                self.actuatorhddm2phase, self.phase2actuatorhddm, \
                self.actuatortt2phase, self.phase2actuatortt = self.create_projectors_of_phase(parameter_file)

                self.actuatorhddm2phase_torch, self.phase2actuatorhddm_torch, \
                self.actuatortt2phase_torch, self.phase2actuatortt_torch = \
                    torch.FloatTensor(self.actuatorhddm2phase).to(self.device),\
                    torch.FloatTensor(self.phase2actuatorhddm).to(self.device), \
                    torch.FloatTensor(self.actuatortt2phase).to(self.device), \
                    torch.FloatTensor(self.phase2actuatortt).to(self.device)

        if self.control_tt:
            assert self.num_dm > 1, "You must have a TT DM if you want to control it"

    #           Basic environment
    #
    #  __  __      _   _               _
    # |  \/  | ___| |_| |__   ___   __| |___
    # | |\/| |/ _ \ __| '_ \ / _ \ / _` / __|
    # | |  | |  __/ |_| | | | (_) | (_| \__ \
    # |_|  |_|\___|\__|_| |_|\___/ \__,_|___/

    def create_projectors_of_phase(self, parameter_file):
        """
        We create the projector matrices needed to go from phase to hddm, hddm to phase, tt to phase, phase to tt
        Args:
            parameter_file: current parameter_file

        Returns: different projector matrices

        """
        projector_creator = ProjectorCreator(parameter_file,
                                             self.supervisor,
                                             second_dm_index=1)
        actuator2phase, phase2actuator, actuatorhddm2phase, phase2actuatorhddm, actuatortt2phase, phase2actuatortt = \
            projector_creator.get_projector_targetphase2actuator()
        del projector_creator

        return actuator2phase, phase2actuator, actuatorhddm2phase, phase2actuatorhddm, actuatortt2phase, phase2actuatortt

    def define_state_action_space(self):
        """
        Define state-action space for RL. It might be different than the of the linear environment
        Returns: number of channels of the state, observation shape, observation space, action space
        """
        state_size_channel_0 = int(self.config_env_rl['number_of_previous_s_dm']) +\
                               int(self.config_env_rl['number_of_previous_s_dm_residual']) + \
                               int(self.config_env_rl['number_of_previous_s_dm_residual_rl']) + \
                               int(self.config_env_rl['number_of_previous_s_dm_residual_tt']) + \
                               int(self.config_env_rl['number_of_previous_s_dm_tt']) + \
                               int(self.config_env_rl['s_dm_residual_rl']) + \
                               int(self.config_env_rl['s_dm_residual']) + \
                               int(self.config_env_rl['s_dm']) + \
                               int(self.config_env_rl['s_dm_tt']) + \
                               int(self.config_env_rl['s_dm_residual_tt'])

        observation_shape = (state_size_channel_0,) + self.action_2d_shape
        observation_space = spaces.Box(low=-math.inf, high=math.inf, shape=observation_shape, dtype=np.float32)
        action_space = spaces.Box(low=-math.inf, high=math.inf, shape=self.action_2d_shape, dtype=np.float32)

        return state_size_channel_0, observation_shape, observation_space, action_space

    def append_to_attr(self, attr, value):
        self.__dict__[attr].append(value)

    def clear_attr(self, attr):
        self.__dict__[attr].clear()

    def reset_history(self, attributes_list):
        """
        Reset lists of history for a certain attribute
        Args:
            attributes_list: the attributes list we want to reset

        Returns: None

        """
        idx = 0
        for attr in attributes_list:
            attr_history = attr + "_history"
            shape_of_history = np.zeros(self.action_2d_shape, dtype="float32")
            self.clear_attr(attr_history)
            rang = self.config_env_rl["number_of_previous_" + attr]
            for _ in range(rang):
                self.append_to_attr(attr_history, shape_of_history)
            idx += 1

    def reset(self,
              only_reset_dm: bool = False,
              return_dict: bool = False,
              add_one_to_seed: bool = True):

        """
        Reset the environment and some parameters for the RL
        Args:
            only_reset_dm: only if we want to reset the DM and not the atmosphere
            return_dict: if we return the state as a dictionary
            add_one_to_seed: if we add one to the current seed

        Returns:

        """
        if add_one_to_seed:
            self.supervisor.add_one_to_seed()
            print("Resetting add one to seed", self.supervisor.current_seed)
        self.supervisor.reset(only_reset_dm)
        # We reset state history and a_for_reward history
        self.reset_history(self.list_of_keys_of_state + ["a_for_reward"])
        if not only_reset_dm:
            self.supervisor.move_atmos_compute_wfs_reconstruction()

        self.build_state()
        s = self.get_next_state(return_dict=return_dict)
        return s

    def reset_without_rl(self, only_reset_dm=False, add_one_to_seed=False):
        """
        Reset the environment without resetting and parameter of RL
        Args:
            only_reset_dm: only if we want to reset the DM and not the atmosphere
            add_one_to_seed: if we add one to the current seed

        Returns: None
        """
        if add_one_to_seed:
            self.supervisor.add_one_to_seed()
            print("Resetting add one to seed", self.supervisor.current_seed)
        self.supervisor.reset(only_reset_dm)
        if not only_reset_dm:
            self.supervisor.move_atmos_compute_wfs_reconstruction()

    def standardise(self, inpt, key):
        """
        standardises
        :param inpt: state to be normalized
        :param key: "wfs" or "dm"
        :return: input normalized
        """

        mean = self.norm_parameters[key]['mean']
        std = self.norm_parameters[key]['std']
        return (inpt - mean) / std

    def filter_actions(self, a, exploratory=False):
        # TODO
        if exploratory:
            # assert self.num_dm > 1
            if self.control_tt:
                self.a_tt_from_hddm = self.hddm2tt(a[:-2]) + a[-2:]
                self.a_hddm_from_tt = self.tt2hddm(self.a_tt_from_hddm)

            a = a[:-2]
            if self.config_env_rl['filter_state_actuator_space_with_btt']:
                a = self.supervisor.filter_dm_info_actuator_space_with_modal_basis(a,
                                                                                   add_tip_tilt_to_not_break=True)
            if self.control_tt:
                a = a + self.a_hddm_from_tt

            a = self.supervisor.apply_projector_volts1d_to_volts2d(a)
        elif self.control_tt and self.two_output_actor:
            a_hddm = a[0]
            self.a_hddm_from_tt = self.supervisor.apply_projector_volts2d_to_volts1d(a[1])
            self.a_tt_from_hddm = self.hddm2tt(self.a_hddm_from_tt)

            if self.config_env_rl['filter_state_actuator_space_with_btt']:
                a_hddm = self.supervisor.apply_projector_volts2d_to_volts1d(a_hddm)
                a_hddm = self.supervisor.filter_dm_info_actuator_space_with_modal_basis(a_hddm,
                                                                                       add_tip_tilt_to_not_break=True if self.num_dm > 1 else False)
                a = a.cpu().numpy()
                a[0] = self.supervisor.apply_projector_volts1d_to_volts2d(a_hddm)
            self.a_hddm_from_tt = self.a_hddm_from_tt.cpu().numpy()
        else:
            # 0) To 1D
            a = self.supervisor.apply_projector_volts2d_to_volts1d(a)
            if self.control_tt:
                self.a_tt_from_hddm = self.hddm2tt(a)
                self.a_hddm_from_tt = self.tt2hddm(self.a_tt_from_hddm)

            # 1) In case of actuator space filter with Btt if necessary
            if self.config_env_rl['filter_state_actuator_space_with_btt']:
                a = self.supervisor.filter_dm_info_actuator_space_with_modal_basis(a,
                                                                                   add_tip_tilt_to_not_break=True if self.num_dm > 1 else False)
                if self.control_tt:
                    a = a + self.a_hddm_from_tt

            a = self.supervisor.apply_projector_volts1d_to_volts2d(a)

        return a

    def add_s_dm_info(self, s_next, s_dm_info, key_attr, key_norm):
        """
        Adds dm info to state
        Args:
            s_next: current state dictionary
            s_dm_info: info we want to add to the state
            key_attr: name of the info we want to add
            key_norm: name of the normalization we want to apply

        Returns: s_next with added info

        """
        key_attribute_history = key_attr + "_history"
        current_history = getattr(self, key_attribute_history)
        for idx in range(len(current_history)):
            past_s_dm_info = getattr(self, key_attribute_history)[idx]
            past_s_dm_info = self.process_dm_state(past_s_dm_info, key=key_norm)
            s_next[key_attribute_history + "_" + str(len(current_history) - idx)] = \
                past_s_dm_info

        if self.config_env_rl["number_of_previous_" + key_attr] > 0:
            self.append_to_attr(key_attribute_history, s_dm_info)

        # 2 Add current residual to the state
        if self.config_env_rl[key_attr]:
            s_dm = self.process_dm_state(s_dm_info, key=key_norm)
            s_next[key_attr] = s_dm.copy()
        return s_next

    def process_dm_state(self, s_dm, key):
        if self.config_env_rl['normalization_bool']:
            s_dm = self.standardise(s_dm, key=key)
        return s_dm

    def calculate_linear_residual(self):
        """
        Calculates linear reconstruction
        Returns: linear reconstruction as a vector in 1D
        """
        c_linear = self.supervisor.rtc.get_err(0)
        if self.config_env_rl['filter_state_actuator_space_with_btt']:
            c_linear = self.supervisor.filter_dm_info_actuator_space_with_modal_basis(c_linear)

        return c_linear

    def penalize_action_in_reward(self, r2d, a):
        """
        Penalizes reward with magnitude of actions
        Args:
            r2d: the raw reward
            a: current actions

        Returns: Reward with actions penalization
        """
        if self.config_env_rl['value_action_penalizer'] > 0 and a is not None:
            if len(self.a_for_reward_history) > 0:
                action_penalizer = self.a_for_reward_history[0]
            else:
                action_penalizer = a
            r2d = r2d - \
                  self.config_env_rl['value_action_penalizer'] * np.square(action_penalizer)
            self.a_for_reward_history.append(a)

        return r2d

    def calculate_reward(self, a):
        """
        Calculates the reward for RL
        Args:
            a: current action

        Returns: reward in 2D
        """
        linear_reconstruction = self.calculate_linear_residual()
        # For metrics
        self.reconstruction_for_reward = linear_reconstruction
        # Build reward
        r2d_linear_rec = -np.square(self.preprocess_dm_info(linear_reconstruction,
                                                            sum_tt_projection=self.config_env_rl['joint_tt_into_reward']))

        # TODO
        r2d_linear_rec /= r2d_linear_rec.reshape(-1).shape[0]
        if not self.config_env_rl['joint_tt_into_reward'] and self.control_tt:
            tt_reconstruction = self.config_env_rl['scaling_for_residual_tt'] * linear_reconstruction[-2:]
            r2d_hddm_reconstruction_from_tt =\
                -np.square(self.supervisor.apply_projector_volts1d_to_volts2d(self.tt2hddm(tt_reconstruction)))
            r2d_linear_rec = np.stack([r2d_linear_rec, r2d_hddm_reconstruction_from_tt])

        r2d_linear_rec = self.penalize_action_in_reward(r2d_linear_rec, a)

        if self.config_env_rl['reward_type'] == "scalar_actuators":
            return r2d_linear_rec.mean()
        elif self.config_env_rl['reward_type'] == "2d_actuators":
            return r2d_linear_rec
        else:
            raise NotImplementedError

    #
    #         Step methods
    # This works for both level = Correction and level = Gain
    #  __  __      _   _               _
    # |  \/  | ___| |_| |__   ___   __| |___
    # | |\/| |/ _ \ __| '_ \ / _ \ / _` / __|
    # | |  | |  __/ |_| | | | (_) | (_| \__ \
    # |_|  |_|\___|\__|_| |_|\___/ \__,_|___/

    @torch.no_grad()
    def hddm2tt(self, dm_hddm):
        """
        From HDDM project to TT
        Args:
            dm_hddm: shape (hddm.shape,)

        Returns: dm_tt_from_hddm shape (2.shape,)
        """
        if self.device > -1:
            if not torch.is_tensor(dm_hddm):
                dm_hddm = torch.FloatTensor(dm_hddm).to(self.device)
            phase = self.actuatorhddm2phase_torch @ dm_hddm
        else:
            phase = self.actuatorhddm2phase.dot(dm_hddm)
        # remove piston
        phase = phase - phase.mean()
        if self.device > -1:
            dm_tt_from_hddm = (self.phase2actuatortt_torch @ phase).cpu().numpy()
        else:
            dm_tt_from_hddm = self.phase2actuatortt.dot(phase)
        return dm_tt_from_hddm

    @torch.no_grad()
    def tt2hddm(self, dm_tt):
        """
        From TT project to HDDM
        Args:
            dm_tt: shape (2,)

        Returns: dm_hddm_from_tt shape (hddm.shape,)
        """
        if self.device > -1:
            dm_tt_torch = torch.FloatTensor(dm_tt).to(self.device)
            phase = self.actuatortt2phase_torch@dm_tt_torch
        else:
            phase = self.actuatortt2phase.dot(dm_tt)
        # remove piston
        phase = phase - phase.mean()
        if self.device > -1:
            dm_hddm_from_tt = (self.phase2actuatorhddm_torch @ phase).cpu().numpy()
        else:
            dm_hddm_from_tt = self.phase2actuatorhddm.dot(phase)
        return dm_hddm_from_tt

    def preprocess_dm_info(self, s_dm_info, sum_tt_projection=False):
        """
        Preprocess information about the DM either command or reconstruction
        Args:
            s_dm_info: current s_dm_info in vector form
            sum_tt_projection: Do we sum tt projection even if we are controlling tt

        Returns: s_dm_info after processing matrix form
        """
        if self.num_dm == 2:
            s_hddm_info = s_dm_info[:-2]
            if self.control_tt and sum_tt_projection:
                s_tt = self.config_env_rl['scaling_for_residual_tt'] * s_dm_info[-2:]
                s_hddm_from_tt = self.tt2hddm(s_tt)
                s_hddm_info += s_hddm_from_tt
            s_dm = self.supervisor.apply_projector_volts1d_to_volts2d(s_hddm_info)
        else:
            raise NotImplementedError

        return s_dm

    def build_state(self):
        self.build_state_simple()

    def build_state_simple(self):
        """
        Creating the state with values only coming from linear Rec. and commands
        Returns: new state

        """
        s_next = OrderedDict()

        # TT state
        if self.config_env_rl['s_dm_residual_tt']:
            linear_rec = self.calculate_linear_residual()
            s_tt = self.config_env_rl['scaling_for_residual_tt'] * linear_rec[-2:]
            s_hddm_from_tt = self.supervisor.apply_projector_volts1d_to_volts2d(self.tt2hddm(s_tt))
            s_next = self.add_s_dm_info(s_next, s_hddm_from_tt, key_attr="s_dm_residual_tt", key_norm="dm_residual")

        if self.config_env_rl['number_of_previous_s_dm_tt'] > 0 or self.config_env_rl['s_dm_tt']:
            s_tt = self.supervisor.past_command[-2:]
            s_hddm_from_tt = self.supervisor.apply_projector_volts1d_to_volts2d(self.tt2hddm(s_tt))
            s_next = self.add_s_dm_info(s_next, s_hddm_from_tt, key_attr="s_dm_tt", key_norm="dm")

        # HDDM state or TT + HDDM state
        if self.config_env_rl['number_of_previous_s_dm'] > 0 or self.config_env_rl['s_dm']:
            s_dm = self.preprocess_dm_info(self.supervisor.past_command,
                                           sum_tt_projection=self.config_env_rl['joint_tt_into_s_dm'])
            s_next = self.add_s_dm_info(s_next, s_dm, key_attr="s_dm", key_norm="dm")

        if self.config_env_rl['s_dm_residual_rl']:
            # Actions work a bit different that other dm info
            if self.control_tt:
                past_a = self.preprocess_dm_info(self.supervisor.past_action_rl)
            else:
                past_a = self.supervisor.apply_projector_volts1d_to_volts2d(self.supervisor.past_action_rl)
            s_next = self.add_s_dm_info(s_next, past_a, key_attr="s_dm_residual_rl", key_norm="dm_residual")

        if self.config_env_rl['s_dm_residual']:
            s_dm_residual = self.preprocess_dm_info(self.calculate_linear_residual(),
                                                    sum_tt_projection=self.config_env_rl['joint_tt_into_s_dm_residual'])
            s_next = self.add_s_dm_info(s_next, s_dm_residual, key_attr="s_dm_residual", key_norm="dm_residual")

        self.s_next_main = s_next

    def get_next_state(self, return_dict):
        if return_dict:
            return self.s_next_main
        else:
            return np.stack(np.array(list(self.s_next_main.values())))

    def step_process_rl_action(self, a):
        if self.two_output_actor:
            a = a[0]
        a_2d = a * self.config_env_rl['action_scale']
        a_1d = self.supervisor.apply_projector_volts2d_to_volts1d(a_2d)
        if self.control_tt:
            self.a_hddm_from_tt *= self.config_env_rl['action_scale']
            self.a_tt_from_hddm *= self.config_env_rl['action_scale']

            # TODO filter is not needed here?
            self.a_hddm = self.supervisor.filter_dm_info_actuator_space_with_modal_basis(a_1d - self.a_hddm_from_tt,
                                                                                         add_tip_tilt_to_not_break=True)

            a_1d = np.concatenate([self.a_hddm, self.a_tt_from_hddm])

        return a_1d, a_2d

    def step(self, a, controller_type):
        """
        A step in the environment.
        We send an action and we get a new state and reward
        Args:
            a: action
            controller_type: "RL", "UNet+Linear" or TODO

        Returns:
            New state
            New reward
            Some information
        """
        a_2d = None
        if controller_type == "RL":
            # 1) Rescale actions and transform them into 1D
            a, a_2d = self.step_process_rl_action(a)
        # 2) Move DM and compute strehl
        self.supervisor.move_dm_and_compute_strehl(a,
                                                   controller_type=controller_type)

        r = self.calculate_reward(a_2d)
        sr_se, sr_le, _, _ = self.supervisor.target.get_strehl(0)
        info = {"sr_se": sr_se,
                "sr_le": sr_le}
        # 3) Move atmos, compute WFS and reconstruction
        self.supervisor.move_atmos_compute_wfs_reconstruction()
        # 4) Build state
        self.build_state()
        s = self.get_next_state(return_dict=False)

        if self.supervisor.iter % (self.config_env_rl['reset_strehl_every_and_print'] + 1) == 0 and\
                self.supervisor.iter > 1:
            self.supervisor.target.reset_strehl(0)

        return s, r, False, info

    def step_only_linear(self):
        """
        A step on the environment only with the linear controller
        Returns: None

        """
        self.supervisor.move_dm_and_compute_strehl(None, controller_type="Linear")
        self.supervisor.move_atmos_compute_wfs_reconstruction()

