import os
from src.env_methods.env import AoEnv
from src.unet.unet import UnetGenerator
import numpy as np
import pandas as pd
import torch
from collections import OrderedDict
from gym import spaces
from collections import deque
import math


class AoEnvNonLinear(AoEnv):
    def __init__(self, unet_dir, unet_name, unet_type, device_unet,
                 gain_factor_unet,
                 gain_factor_linear=0,
                 normalization_095_005=True,
                 config_env_rl=None,
                 parameter_file=None,
                 seed=None,
                 device=None,
                 normalization_noise_unet=False,
                 normalization_noise_value_unet=3):

        self.device_unet = device_unet
        self.normalization_noise_unet = normalization_noise_unet
        # We save the non-linear rec as a variable
        self.non_linear_reconstruction = None
        self.normalization_095_005 = normalization_095_005
        if normalization_noise_unet:
            self.readout_noise = normalization_noise_value_unet
        else:
            self.readout_noise = 0
        self.gain_factor_unet = gain_factor_unet
        self.gain_factor_linear = gain_factor_linear
        self.no_subtract_mean_from_phase = config_env_rl['no_subtract_mean_from_phase']
        print("----------------")
        print("-- Unet Model --")
        print("U-Net name: ", unet_name)
        print("U-Net dir: ", unet_dir)
        print("U-Net type: ", unet_type)
        print("U-Net no_subtract_mean_from_phase", self.no_subtract_mean_from_phase)

        super(AoEnvNonLinear, self).__init__(config_env_rl, parameter_file, seed, device,
                                             override_generate_phase_projectors=True)

        self.unet_type = None
        self.unet_name = None
        self.unet_dir = None
        self.min_wfs_image, self.scale_wfs, \
            self.value_left_up_1, self.value_left_up_2, self.value_right_down_1, self.value_right_down_2, \
            self.min_phase, self.scale_phase, self.out_mask, self.wfs_image_pad_value, self.out_unpad_value = \
            [None] * 11
        self.model = None

        self.update_unet_params(unet_dir, unet_name, unet_type)

        self.s_dm_residual_non_linear_history = \
            deque(maxlen=self.config_env_rl['number_of_previous_s_dm_residual_non_linear'])
        self.s_dm_residual_non_linear_tt_history = \
            deque(maxlen=self.config_env_rl['number_of_previous_s_dm_residual_non_linear_tt'])

        self.list_of_keys_of_state = ['s_dm', 's_dm_residual', 's_dm_residual_rl', 'a_for_reward', 's_dm_residual_tt',
                                      's_dm_tt', 's_dm_residual_non_linear', 's_dm_residual_non_linear_tt']

        self.linear_reconstructor = self.supervisor.rtc.get_command_matrix(0)

    def update_unet_params(self, unet_dir, unet_name, unet_type):
        """
        Updates U-Nets parameters
        Args:
            unet_dir: path to directory where U-Net weights are saved
            unet_name: name of the current U-nET
            unet_type: "volts" or "phase"

        Returns: None
        """
        self.unet_type = unet_type
        self.unet_name = unet_name
        self.unet_dir = unet_dir

        self.model = self.load_unet(unet_dir, unet_name, unet_type)

        self.min_wfs_image, self.scale_wfs, \
            self.value_left_up_1, self.value_left_up_2, self.value_right_down_1, self.value_right_down_2, \
            self.min_phase, self.scale_phase, self.out_mask, self.wfs_image_pad_value, self.out_unpad_value = \
            self.prepare_values_norm_pad(unet_dir, unet_type, self.normalization_095_005)

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
                               int(self.config_env_rl['number_of_previous_s_dm_residual_non_linear']) + \
                               int(self.config_env_rl['number_of_previous_s_dm_residual_non_linear_tt']) + \
                               int(self.config_env_rl['s_dm_residual_rl']) + \
                               int(self.config_env_rl['s_dm_residual']) + \
                               int(self.config_env_rl['s_dm']) + \
                               int(self.config_env_rl['s_dm_tt']) + \
                               int(self.config_env_rl['s_dm_residual_tt']) + \
                               int(self.config_env_rl['s_dm_residual_non_linear']) + \
                               int(self.config_env_rl['s_dm_residual_non_linear_tt'])

        observation_shape = (state_size_channel_0,) + self.action_2d_shape
        observation_space = spaces.Box(low=-math.inf, high=math.inf, shape=observation_shape, dtype=np.float32)
        action_space = spaces.Box(low=-math.inf, high=math.inf, shape=self.action_2d_shape, dtype=np.float32)

        return state_size_channel_0, observation_shape, observation_space, action_space

    def load_unet(self, unet_dir, unet_name, unet_type):
        """
        Loads U-Net trained weights
        Args:
            unet_dir: path to directory where U-Net weights are saved
            unet_name: name of the current U-nET
            unet_type: "volts" or "phase"

        Returns: U-Net with loaded weights
        """
        print("Loading UNET: UNet dir:", unet_dir, ", Name: ", unet_name)
        unet_full_path = os.path.join(unet_dir, unet_name)
        if unet_type == "phase":
            model = UnetGenerator(input_nc=4, output_nc=1, num_downs=9, ngf=64).to(self.device_unet)
        elif unet_type == "volts":
            model = UnetGenerator(input_nc=4, output_nc=1, num_downs=6, ngf=64).to(self.device_unet)
        else:
            raise NotImplementedError
        state_dict = torch.load(unet_full_path, map_location=self.device_unet)
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:]  # remove 'module.' from key
            new_state_dict[name] = v
        model.load_state_dict(new_state_dict)
        model.eval()

        return model

    def prepare_values_norm_pad(self, unet_full_dir, unet_type, normalization_095_005):
        """
        Prepares values for normalization and padding for UNet
        TODO: Generalise some hardcoded values for current parameter file
        Args:
            unet_full_dir: path to UNet dir
            unet_type: "volts" or "phase"
            normalization_095_005: if we do not exactly normalize with bounds 0-1 but between 0.05 and 0.95
        Returns: values for normalization and padding
        """
        num_pix_wfs = 56  # full size 256, divide by 2 128, we have 56 pix in the par file
        # for padding wfs
        edge_offset_wfs = 44
        center_offset_wfs = 56
        value_left_up_1 = edge_offset_wfs - 4
        value_left_up_2 = edge_offset_wfs + num_pix_wfs + 4
        value_right_down_1 = edge_offset_wfs + num_pix_wfs + center_offset_wfs - 4
        value_right_down_2 = -edge_offset_wfs + 4
        df_norm = pd.read_csv(unet_full_dir + "/info.csv")
        if unet_type == "volts":
            min_phase = df_norm['Min voltage'].values[0]
            max_phase = df_norm['Max voltage'].values[0]
            num_pix_phase = 40  # 40 because its commands
            unet_size = 64  # 128 max(wfs,phase)
            out_unpad_value = int((unet_size - num_pix_phase) / 2.0)
            out_mask = self.supervisor.mask_valid_actuators
            wfs_image_pad_value = 0
        elif unet_type == "phase":
            min_phase = df_norm['Min phase'].values[0]
            max_phase = df_norm['Max phase'].values[0]
            num_pix_phase = 448  # 448
            unet_size = 512  # 512
            out_mask = self.supervisor.get_s_pupil()
            wfs_image_pad_value = int(unet_size / 2) - 32  # 192
            out_unpad_value = int((unet_size - num_pix_phase)/2.0)
        else:
            raise NotImplementedError

        if normalization_095_005:
            scale_phase = (max_phase - min_phase) / 0.9
            min_phase = min_phase - 0.05 * scale_phase
        min_wfs_image = df_norm['Min wfs'].values[0]
        max_wfs_image = df_norm['Max wfs'].values[0]
        if self.normalization_noise_unet:
            min_wfs_image = 0
        scale_wfs = (max_wfs_image - min_wfs_image)

        scale_phase = (max_phase - min_phase)
        return min_wfs_image, scale_wfs, value_left_up_1, value_left_up_2, value_right_down_1, value_right_down_2,\
               min_phase, scale_phase, out_mask, wfs_image_pad_value, out_unpad_value

    def pad_expand(self, wfs_image):
        """
        Pads WFS image
        """
        if self.wfs_image_pad_value > 0:
            wfs_image = np.pad(wfs_image, ((self.wfs_image_pad_value, self.wfs_image_pad_value),
                                           (self.wfs_image_pad_value, self.wfs_image_pad_value)), 'constant')
        wfs_image = np.expand_dims(wfs_image, axis=0)
        return wfs_image

    def prepare_wfs_image(self, wfs_image):
        """
        Preprocessing of WFS image for U-Net
        Args:
            wfs_image: current WFS image
        Returns: processed WFS image
        """

        if self.normalization_noise_unet:
            # Things below readout_noise... maybe forget about them?
            wfs_image = wfs_image - self.readout_noise
            wfs_image[wfs_image < 0] = 0

        # Remove extra dimensions
        wfs_image = np.squeeze(wfs_image)
        wfs_channel_1 = wfs_image[self.value_left_up_1:self.value_left_up_2,
                                  self.value_left_up_1:self.value_left_up_2]
        # Lower left
        wfs_channel_2 = wfs_image[self.value_left_up_1:self.value_left_up_2,
                                  self.value_right_down_1:self.value_right_down_2]
        # Upper right
        wfs_channel_3 = wfs_image[self.value_right_down_1:self.value_right_down_2,
                                  self.value_left_up_1:self.value_left_up_2]
        # Lower right
        wfs_channel_4 = wfs_image[self.value_right_down_1:self.value_right_down_2,
                                  self.value_right_down_1:self.value_right_down_2]

        # Pad, important before normalization because we are not centered at 0, we are centered at min_wfs_image
        wfs_channel_1 = self.pad_expand(wfs_channel_1)
        wfs_channel_2 = self.pad_expand(wfs_channel_2)
        wfs_channel_3 = self.pad_expand(wfs_channel_3)
        wfs_channel_4 = self.pad_expand(wfs_channel_4)
        # Concatenate
        wfs_image_multiple_channels = np.concatenate([wfs_channel_1, wfs_channel_2, wfs_channel_3, wfs_channel_4],
                                                     axis=0)
        wfs_image_multiple_channels_norm = (wfs_image_multiple_channels - self.min_wfs_image) / self.scale_wfs

        wfs_image_multiple_channels_norm = np.expand_dims(wfs_image_multiple_channels_norm, axis=0)

        # To torch
        wfs_image_multiple_channels_norm_torch =\
            torch.FloatTensor(wfs_image_multiple_channels_norm).to(self.device_unet)

        return wfs_image_multiple_channels_norm_torch

    def process_output(self, out):
        """
        Denormalises output of U-Net and does extra preprocessing
        Args:
            out: unprocessed output of U-Net

        Returns: processed output of U-Net

        """
        # 1) To numpy
        out = out.cpu().numpy()

        # 2) Unpad
        out = out[0, 0, self.out_unpad_value:-self.out_unpad_value, self.out_unpad_value:-self.out_unpad_value]

        # 3) Denormalize
        out = out * self.scale_phase  # + self.min_phase - We remove the min from the concept

        # 4) Multiply mask
        out = np.multiply(out, self.out_mask)

        # 5) Remove mean if necessary
        if self.no_subtract_mean_from_phase:
            pass
        else:
            out[self.out_mask == 1] -= out[self.out_mask == 1].mean()

        return out

    def process_with_projectors(self, out, no_filter):
        """
        Processes output of U-Net to separate HDDM and TT correction stages
        Args:
            out: output of U-Net in 2D
            no_filter: if we force not to filter with Btt

        Returns: output in 1D where the last two elements correspond to TT and first elements correspond to the HDDM

        """

        # 7) Project to volts (only phase) or to 1D (only volts)
        if self.unet_type == "phase":
            out_final = self.phase2actuator.dot(out[self.out_mask == 1])
            if no_filter:
                pass
            else:
                out_final = self.supervisor.filter_dm_info_actuator_space_with_modal_basis(out_final)
        elif self.unet_type == "volts":
            out_1d = self.supervisor.apply_projector_volts2d_to_volts1d(out)
            if no_filter:
                out_final = np.concatenate([out_1d, [0, 0]])
            else:
                out_hddm = \
                    self.supervisor.filter_dm_info_actuator_space_with_modal_basis(out_1d,
                                                                                   add_tip_tilt_to_not_break=True)
                out_tt = self.hddm2tt(out_1d)
                out_final = np.concatenate([out_hddm, out_tt])
        else:
            raise NotImplementedError

        return out_final

    @torch.no_grad()
    def infer(self, wfs_image, no_filter=False):
        """
        Map WFS image to DM
        Args:
            wfs_image: current wfs_image
            no_filter: if we force not to filter with Btt modes

        Returns: output of inference processed. Output will be in 1D.

        """
        wfs_image_processed = self.prepare_wfs_image(wfs_image)
        out = self.model(wfs_image_processed)
        processed_output = self.process_output(out)
        processed_output_final = self.process_with_projectors(processed_output, no_filter)

        return processed_output_final

    def calculate_linear_reconstruction(self, slopes):
        """
        Calculates linear reconstruction from slopes
        Returns: linear reconstruction as a vector in 1D
        """
        linear_reconstruction = self.linear_reconstructor.dot(slopes)
        linear_reconstruction = self.supervisor.filter_dm_info_actuator_space_with_modal_basis(linear_reconstruction)
        return linear_reconstruction

    def add_linear_reconstruction(self, out, slopes):
        linear_reconstruction = self.calculate_linear_reconstruction(slopes)
        return out + linear_reconstruction

    def step_only_unet(self, tt_linear=False):

        # 1)  Non-linear
        wfs_image = self.supervisor.wfs.get_wfs_image(0)
        non_linear_reconstruction = - self.infer(wfs_image)  # negative because we need inverse prediction

        non_linear_reconstruction_with_gain = self.gain_factor_unet * non_linear_reconstruction
        if tt_linear:
            linear_reconstruction = self.supervisor.rtc.get_err(0)
            non_linear_reconstruction_with_gain[-2:] = linear_reconstruction[-2:]
        # 2) Move DM and compute strehl
        self.supervisor.move_dm_and_compute_strehl(action=non_linear_reconstruction_with_gain, controller_type="UNet")

        # 3) Compute new wfs image
        self.supervisor.move_atmos_compute_wfs_reconstruction()

    def step_only_combined_with_linear(self):
        """
        Step in the environment using U-Net + Linear Rec.
        Returns: None
        """
        assert self.gain_factor_linear is not None

        reconstruction = self.calculate_non_linear_residual()

        # 1) Move DM and compute strehl
        self.supervisor.move_dm_and_compute_strehl(action=reconstruction, controller_type="UNet+Linear")

        # 2) Compute new wfs image and linear reconstruction
        self.supervisor.move_atmos_compute_wfs_reconstruction()

    def calculate_non_linear_residual(self, update_non_linear_reconstruction_variable=False):
        """
        Calculates non-linear reconstruction
        Returns: non-linear reconstruction as a vector in 1D
        """
        # 1) Non-linear
        wfs_image = self.supervisor.wfs.get_wfs_image(0)
        # negative because we need inverse prediction
        non_linear_reconstruction = - self.infer(wfs_image)

        # 2) linear
        slopes = self.supervisor.rtc.get_slopes(0)
        # negative because we need inverse prediction
        linear_reconstruction = - self.calculate_linear_reconstruction(slopes)

        # 3) Sum
        non_linear_reconstruction_with_gain = self.gain_factor_unet * non_linear_reconstruction
        linear_reconstruction_with_gain = self.gain_factor_linear * linear_reconstruction
        reconstruction = non_linear_reconstruction_with_gain + linear_reconstruction_with_gain
        if update_non_linear_reconstruction_variable:
            self.non_linear_reconstruction = reconstruction

        return reconstruction

    def build_state(self):
        """
        Creating the state with values coming from linear Rec., non-linear Rec. and commands
        Returns: new state

        """
        self.build_state_simple()
        if self.config_env_rl['s_dm_residual_non_linear'] or self.config_env_rl['s_dm_residual_non_linear_tt']:
            non_linear_rec = self.calculate_non_linear_residual(update_non_linear_reconstruction_variable=True)
            if self.config_env_rl['s_dm_residual_non_linear']:
                s_non_linear_tt = self.preprocess_dm_info(non_linear_rec, sum_tt_projection=self.config_env_rl['joint_tt_into_s_dm_residual_non_linear'])
                self.s_next_main = self.add_s_dm_info(self.s_next_main, s_non_linear_tt,
                                                      key_attr="s_dm_residual_non_linear", key_norm="dm_residual")

            if self.config_env_rl['s_dm_residual_non_linear_tt']:
                s_non_linear_tt = self.config_env_rl['scaling_for_residual_tt'] * non_linear_rec[-2:]
                s_non_linear_hddm_from_tt = \
                    self.supervisor.apply_projector_volts1d_to_volts2d(self.tt2hddm(s_non_linear_tt))
                self.s_next_main = self.add_s_dm_info(self.s_next_main, s_non_linear_hddm_from_tt,
                                                      key_attr="s_dm_residual_non_linear_tt", key_norm="dm_residual")

    def step(self, a, controller_type):
        """
        A step in the environment.
        We send an action and we get a new state and reward
        Args:
            a: action
            controller_type: "RL", "UNet+Linear" or "Linear"

        Returns:
            New state
            New reward
            Some information
        """
        a_2d = None
        if controller_type == "RL":
            # 1) Rescale actions and transform them into 1D
            a, a_2d = self.step_process_rl_action(a)
        elif controller_type == "UNet+Linear" or controller_type == "UNet+Linearv2":
            a = self.non_linear_reconstruction

        unet_command = self.non_linear_reconstruction
        if self.config_env_rl['correction_hddm_only_rl_tt']:
            unet_command[-2:] = 0

        # 2) Move DM and compute strehl
        self.supervisor.move_dm_and_compute_strehl(a,
                                                   controller_type=controller_type,
                                                   unet_or_phase_command=unet_command)

        r = self.calculate_reward(a_2d)
        sr_se, sr_le, _, _ = self.supervisor.target.get_strehl(0)
        info = {"sr_se": sr_se,
                "sr_le": sr_le}
        # 3) Move atmos, compute WFS and reconstruction
        self.supervisor.move_atmos_compute_wfs_reconstruction()
        # 4) Build state
        self.build_state()
        s = self.get_next_state(return_dict=False)

        if self.supervisor.iter % (self.config_env_rl['reset_strehl_every_and_print'] + 1) == 0 and \
                self.supervisor.iter > 1:
            self.supervisor.target.reset_strehl(0)

        return s, r, False, info

    def calculate_reward(self, a):
        """
        Calculates the reward for RL
        Args:
            a: current action

        Returns: reward in 2D
        """

        non_linear_reconstruction = self.non_linear_reconstruction
        # For metrics
        self.reconstruction_for_reward = non_linear_reconstruction
        # Build reward
        r2d_non_linear_rec = -np.square(
            self.preprocess_dm_info(non_linear_reconstruction,
                                    sum_tt_projection=self.config_env_rl['joint_tt_into_reward']))

        r2d_non_linear_rec /= r2d_non_linear_rec.reshape(-1).shape[0]
        if not self.config_env_rl['joint_tt_into_reward'] and self.control_tt:
            tt_reconstruction = self.config_env_rl['scaling_for_residual_tt'] * non_linear_reconstruction[-2:]
            r2d_hddm_reconstruction_from_tt = \
                -np.square(self.supervisor.apply_projector_volts1d_to_volts2d(self.tt2hddm(tt_reconstruction)))
            r2d_non_linear_rec = np.stack([r2d_non_linear_rec, r2d_hddm_reconstruction_from_tt])

        r2d_non_linear_rec = self.penalize_action_in_reward(r2d_non_linear_rec, a)

        if self.config_env_rl['reward_type'] == "scalar_actuators":
            return r2d_non_linear_rec.mean()
        elif self.config_env_rl['reward_type'] == "2d_actuators":
            return r2d_non_linear_rec
        else:
            raise NotImplementedError
