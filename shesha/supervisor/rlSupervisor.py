from shesha.supervisor.genericSupervisor import GenericSupervisor
from shesha.supervisor.compassSupervisor import CompassSupervisor
from shesha.supervisor.components import AtmosCompass, DmCompass, RtcCompass, TargetCompass, TelescopeCompass, \
    WfsCompass
from shesha.supervisor.optimizers import ModalBasis, Calibration
import shesha.ao.basis as bas
import numpy as np
import torch


class RlSupervisor(CompassSupervisor):
    def __init__(self, config, n_reverse_filtered_from_cmat, *,
                 filter_commands=True,
                 command_clip_value=1000,
                 initial_seed=1234,
                 which_modal_basis="Btt",
                 mode="only_rl",
                 device=-1,
                 control_tt=False,
                 reset_when_clip=False,
                 reduce_gain_tt_to=-1):
        """ Instantiates a RlSupervisor object

        Args:
            config: (config module) : Configuration module
            n_reverse_filtered_from_cmat :

        Kwargs:
            initial_seed: seed to start experiments
        """

        GenericSupervisor.__init__(self, config)
        # Generic Compass
        self.basis = ModalBasis(self.config, self.dms, self.target)
        self.calibration = Calibration(self.config, self.tel, self.atmos, self.dms,
                                       self.target, self.rtc, self.wfs)
        # Initial values
        self.initial_seed, self.current_seed = initial_seed, initial_seed
        self.filter_commands = filter_commands
        self.command_clip_value = command_clip_value
        self.num_dm = len(self.config.p_dms)
        self.mode = mode
        assert self.mode in ["correction", "only_rl"]
        self.reduce_gain_tt_to = reduce_gain_tt_to

        # Modes
        dms = []
        p_dms = []
        if type(self.config.p_controllers) == list:
            for config_p_controller in self.config.p_controllers:
                if config_p_controller.get_type() != "geo":
                    for dm_idx in config_p_controller.get_ndm():
                        dms.append(self.dms._dms.d_dms[dm_idx])
                        p_dms.append(self.config.p_dms[dm_idx])

        else:
            dms = self.dms._dms.d_dms
            p_dms = self.config.p_dms

        self.modes2volts, self.volts2modes =\
            self.basis.compute_modes_to_volts_basis(dms, p_dms, modal_basis_type=which_modal_basis)

        self.modes_filtered = self.manage_filtered_modes(self.modes2volts, n_reverse_filtered_from_cmat)
        self.obtain_and_set_cmat_filtered(self.modes2volts, n_reverse_filtered_from_cmat)¡
        self.projector_volts1d_to_volts2d, self.projector_volts2d_to_volts1d =\
            self.create_projector_volts1d_to_volts2d()

        # dm/action shapes
        self.command_shape = self.rtc.get_command(0).shape  # always in 1D
        if self.num_dm > 1:
            self.hddm_shape = self.rtc.get_command(0)[:-2].shape
            self.tt_shape = self.rtc.get_command(0)[-2:].shape
            if control_tt:
                self.action_1d_shape = self.command_shape
            else:
                self.action_1d_shape = self.hddm_shape
        else:
            self.hddm_shape = self.rtc.get_command(0).shape
            self.tt_shape = None
            self.action_1d_shape = self.hddm_shape

        self.mask_valid_actuators = self.apply_projector_volts1d_to_volts2d(np.ones(self.hddm_shape))
        self.action_2d_shape = self.mask_valid_actuators.shape

        # Device and if we do some calculations with torch on GPU
        self.device = device
        if device >= 0:
            self.modes2volts_torch, self.volts2modes_torch = \
                torch.FloatTensor(self.modes2volts).to(device), torch.FloatTensor(self.volts2modes).to(device)
            self.projector_volts2d_to_volts1d_torch = torch.FloatTensor(self.projector_volts2d_to_volts1d).to(device)
            # self.projector_volts1d_to_volts2d_torch = torch.FloatTensor(self.projector_volts1d_to_volts2d).to(device)
            self.zeros_tt = torch.zeros(2).to(self.device)
        else:
            self.modes2volts_torch, self.volts2modes_torch = None, None
            self.projector_volts2d_to_volts1d_torch = None
            # self.projector_volts1d_to_volts2d_torch = None
            self.zeros_tt = None

        # Control TT
        self.control_tt = control_tt
        if self.control_tt:
            assert self.num_dm > 1, "To control the TT you need a HDDM and TT mirrors"
        # Save past commands and past actions
        self.past_command = None
        self.past_action_rl = None
        # Projectors
        self.actuatortt2phase = None # We will set it from the environment
        self.phase2actuatorhddm = None # We will set it from the environment
        # Reset DM
        self.reset_when_clip = reset_when_clip
        if self.reset_when_clip:
            self.look_up_table = self.compute_distances_for_reset_actuators()
        else:
            self.look_up_table = None
    #
    #   __             __  __     _   _            _
    #  |   \          |  \/  |___| |_| |_  ___  __| |___
    #  |   |          | |\/| / -_)  _| ' \/ _ \/ _` (_-<
    #  |__/ efault    |_|  |_\___|\__|_||_\___/\__,_/__/
    #

    def set_gain(self, g):
        """
        Sets a certain gain in the system for the linear approach with integrator
        :param g: gain to be set
        """
        # g = self.supervisor.rtc.d_control[0].gain
        if np.isscalar(g):
            self.rtc._rtc.d_control[0].set_gain(g)
        else:
            raise ValueError("Cannot set array gain w/ generic + integrator law")

    def filter_modal_basis(self, modes2volts_unfiltered, volts2modes_unfiltered, nfilt):
        """
        Filters modal basis
        Args:
            modes2volts_unfiltered: modes2volts without removing any modes
            volts2modes_unfiltered: volts2modes without removing any modes
            nfilt: number of modes filtered

        Returns: filtered projection modes2volts and volts2modes

        """

        if self.num_dm == 1:
            raise NotImplementedError
        else:
            # Filtering on Btt modes
            modes2volts = \
                np.zeros((modes2volts_unfiltered.shape[0], modes2volts_unfiltered.shape[1] - nfilt))
            modes2volts[:, :modes2volts.shape[1] - 2] = \
                modes2volts_unfiltered[:, :modes2volts_unfiltered.shape[1] - (nfilt + 2)]
            # TT part
            modes2volts[:, modes2volts.shape[1] - 2:] = \
                modes2volts_unfiltered[:, modes2volts_unfiltered.shape[1] - 2:]

            volts2modes = \
                np.zeros((volts2modes_unfiltered.shape[0] - nfilt, volts2modes_unfiltered.shape[1]))
            volts2modes[:volts2modes.shape[0] - 2, :] = \
                volts2modes_unfiltered[:volts2modes_unfiltered.shape[0] - (nfilt + 2)]
            # TT part
            volts2modes[volts2modes.shape[0] - 2:, :] = \
                volts2modes_unfiltered[volts2modes_unfiltered.shape[0] - 2:, :]

        return modes2volts, volts2modes

    def create_projector_volts1d_to_volts2d(self):
        """
        Creates a projector for the command_vector to go into a 2D array which would take their according positions.
        :return: projector
        """

        def get_linspace_for_projection_volts1d_to_2d():
            # Required information
            x_pos_actuators_ = self.config.p_dms[0]._xpos

            y_pos_actuators_ = self.config.p_dms[0]._ypos
            x_pos_linspace_ = np.linspace(self.config.p_geom._p1 - 0.5,
                                         self.config.p_geom._p2 + 0.5,
                                         self.config.p_dms[0].nact)
            y_pos_linspace_ = np.linspace(self.config.p_geom._p1 - 0.5,
                                         self.config.p_geom._p2 + 0.5,
                                         self.config.p_dms[0].nact)

            return x_pos_linspace_, y_pos_linspace_, x_pos_actuators_, y_pos_actuators_

        x_pos_linspace, y_pos_linspace, x_pos_actuators, y_pos_actuators = \
            get_linspace_for_projection_volts1d_to_2d()

        if self.num_dm > 1:
            total_len = len(self.rtc.get_command(0)) - 2
        else:
            total_len = len(self.rtc.get_command(0))

        epsilon = 1e-3
        # Projector has the shape number of commands - 2 (which is tip tilt)
        projector_volts1d_to_volts2d = np.zeros((total_len,
                                                 self.config.p_dms[0].nact *
                                                 self.config.p_dms[0].nact))

        for i in range(len(x_pos_linspace)):
            for j in range(len(y_pos_linspace)):
                which_idx = -1
                for idx in range(len(x_pos_actuators)):
                    # adding epsilon solves bug of precission for some parameter files
                    if np.abs(x_pos_actuators[idx] - x_pos_linspace[i]) < epsilon \
                            and np.abs(y_pos_actuators[idx] - y_pos_linspace[j]) < epsilon:
                        which_idx = idx
                if which_idx != -1:
                    projector_volts1d_to_volts2d[which_idx, int(i * self.config.p_dms[0].nact) + int(j)] = 1

        projector_volts2d_to_volts1d = np.linalg.pinv(projector_volts1d_to_volts2d)

        return projector_volts1d_to_volts2d, projector_volts2d_to_volts1d

    def manage_filtered_modes(self, modes2volts, nfilt):
        """
        Get a matrix that contains the index of the modes we filter
        Args:
            modes2volts: modes2volts matrix
            nfilt:  number of modes filtered

        Returns: matrix that contains index of modes filtered

        """

        if self.num_dm > 1:
            modes_filtered = np.arange(modes2volts.shape[1] - nfilt - 2,
                                       modes2volts.shape[1] - 2)
        else:
            modes_filtered = np.arange(modes2volts.shape[1] - nfilt,
                                       modes2volts.shape[1])

        return modes_filtered

    def add_one_to_seed(self):
        self.current_seed += 1

    def compute_distances_for_reset_actuators(self):
        """
        Computes a look up table to check the closest actuator to reset if an actuator reaches the clip value
        """

        # Build distance matrix
        xpos = np.array([np.arange(40) for _ in range(40)])
        xpos[self.mask_valid_actuators == 0] = -99999
        xpos = xpos.reshape(-1)
        ypos = np.array([np.array([i]*40) for i in range(40)])
        ypos[self.mask_valid_actuators == 0] = -99999
        ypos = ypos.reshape(-1)
        distance_matrix = ((xpos[None, :] - xpos[:, None]) ** 2 + (ypos[None, :] - ypos[:, None]) ** 2) ** 0.5

        # Build lookup table
        look_up_table = {}
        idx = 0
        for i in range(len(self.rtc.get_command(0)) - 2):
            while True:
                if self.mask_valid_actuators.reshape(-1)[idx] == 1:
                    break
                else:
                    idx += 1

            closest = np.argsort(distance_matrix[idx, :].reshape(40, 40)[self.mask_valid_actuators == 1])
            look_up_table[i] = closest
            idx += 1

        return look_up_table

    def reset_clipped_actuator(self, final_command_actuator):
        """
        Reset a clipped actuator to the closest neighbour
        """

        # 1) Identify actuators that reached the clip value
        clipped_indices = np.where(np.abs(final_command_actuator[:-2]) == self.command_clip_value)[0]
        closest_to_clipped = []
        for index in clipped_indices:
            index_in_lookup = 0
            while True:
                closest_index = self.look_up_table[index][index_in_lookup]

                if closest_index not in clipped_indices:
                    break
                else:
                    index_in_lookup += 1
            closest_to_clipped.append(closest_index)

        # 2) Get the values of the two closest non-clipped actuators
        closest_values = final_command_actuator[np.array([closest_to_clipped])]

        # 3) Change its value
        final_command_actuator[clipped_indices] = closest_values

        return final_command_actuator

    def obtain_and_set_cmat_filtered(self, btt2act, n_reverse_filtered_from_cmat):
        """
        Sets Cmat build with the Btt modes
        """
        if n_reverse_filtered_from_cmat > -1:
            # print("Using Btt basis")
            # print("+ Shape Btt basis", btt2act.shape)
            assert type(n_reverse_filtered_from_cmat) == int
            cmat = bas.compute_cmat_with_Btt(rtc=self.rtc._rtc,
                                             Btt=btt2act,
                                             nfilt=n_reverse_filtered_from_cmat)
            print("+ Number of filtered modes", n_reverse_filtered_from_cmat)
        else:
            print("+ WARNING: NO CMAT BUILT FROM BTT, IT IS THE ORIGINAL WITHOUT FILTERING ANY MODES!!!")
            cmat = None

        return cmat

    @torch.no_grad()
    def apply_projector_volts2d_to_volts1d(self, command_2d):
        """
        Change commands from 2d to 1d
        """
        if torch.is_tensor(command_2d):
            command_2d_flattened = command_2d.flatten().to(self.device)
            command_1d = command_2d_flattened@self.projector_volts2d_to_volts1d_torch
        else:
            command_1d = command_2d.flatten().dot(self.projector_volts2d_to_volts1d)
        return command_1d

    def apply_projector_volts1d_to_volts2d(self, command_1d):
        """
        Change commands from 1d to 2d
        """
        command_2d = command_1d.dot(self.projector_volts1d_to_volts2d).reshape(self.config.p_dms[0].nact, self.config.p_dms[0].nact)
        return command_2d

    def reset(self, only_reset_dm=False, noise=None):
        """ Reset the simulation to return to its original state
        """
        if not only_reset_dm:
            self.atmos.reset_turbu(self.current_seed)
            self.wfs.reset_noise(self.current_seed, noise)
            for tar_index in range(len(self.config.p_targets)):
                self.target.reset_strehl(tar_index)
        self.dms.reset_dm()
        self.rtc.open_loop()
        self.rtc.close_loop()
        self.reset_past_commands()  # Only necessary for Only RL
        self.iter = 0

    def reset_past_commands(self):
        """
        Resetting past commands (i.e. command at frame t-1)
        Past command RL is used either for only_rl_integrated or with slope_space + slope_space_correction
        """
        self.past_command = np.zeros(self.command_shape)
        self.past_action_rl = np.zeros(self.action_1d_shape)

    #     ___                  _      __  __     _   _            _
    #    / __|___ _ _  ___ _ _(_)__  |  \/  |___| |_| |_  ___  __| |___
    #   | (_ / -_) ' \/ -_) '_| / _| | |\/| / -_)  _| ' \/ _ \/ _` (_-<
    #    \___\___|_||_\___|_| |_\__| |_|  |_\___|\__|_||_\___/\__,_/__/

    def _init_tel(self):
        """Initialize the telescope component of the supervisor as a TelescopeCompass
        """
        self.tel = TelescopeCompass(self.context, self.config)

    def _init_atmos(self):
        """Initialize the atmosphere component of the supervisor as a AtmosCompass
        """
        self.atmos = AtmosCompass(self.context, self.config)

    def _init_dms(self):
        """Initialize the DM component of the supervisor as a DmCompass
        """
        self.dms = DmCompass(self.context, self.config)

    def _init_target(self):
        """Initialize the target component of the supervisor as a TargetCompass
        """
        if self.tel is not None:
            self.target = TargetCompass(self.context, self.config, self.tel)
        else:
            raise ValueError("Configuration not loaded or Telescope not initilaized")

    def _init_wfs(self):
        """Initialize the wfs component of the supervisor as a WfsCompass
        """
        if self.tel is not None:
            self.wfs = WfsCompass(self.context, self.config, self.tel)
        else:
            raise ValueError("Configuration not loaded or Telescope not initilaized")

    def _init_rtc(self):
        """Initialize the rtc component of the supervisor as a RtcCompass
        """
        if self.wfs is not None:
            self.rtc = RtcCompass(self.context, self.config, self.tel, self.wfs,
                                  self.dms, self.atmos, cacao=False)
        else:
            raise ValueError("Configuration not loaded or Telescope not initilaized")

    #   _____   __        __  __     _   _            _
    #  |  _  | |  |      |  \/  |___| |_| |_  ___  __| |___
    #  | |/ /  |  |__    | |\/| / -_)  _| ' \/ _ \/ _` (_-<
    #  | |\ \  |_ _ _|   |_|  |_\___|\__|_||_\___/\__,_/__/
    #

    #                            I
    #   _____   __               __  __     _   _            _
    #  |  _  | |  |             |  \/  |___| |_| |_  ___  __| |___
    #  | |/ /  |  |__           | |\/| / -_)  _| ' \/ _ \/ _` (_-<
    #  | |\ \  |_ _ _| control  |_|  |_\___|\__|_||_\___/\__,_/__/
    #

    #
    #       I a) Methods used by only RL and correction
    #
    #
    #

    def rl_control(self,
                   action: np.ndarray,
                   ncontrol: int,
                   unet_or_phase_command: np.ndarray = None,
                   controller_type: str = "RL"):
        """
        Control for RL predictive control
        Args:
            action: action in 1D
            ncontrol: control idx
            unet_or_phase_command: if instead we use integrator with U-Net or phase projected we will use this value instead
            controller_type: controller type
        """

        if self.mode == "correction" and controller_type == "RL":
            final_command = self.correction_control(action=action,
                                                    unet_or_phase_command=unet_or_phase_command)
        else:
            final_command = self.only_rl_control(action=action)
        self.past_command = final_command
        self.past_action_rl = action

        self.rtc.set_command(ncontrol, final_command)

        #
        #       II b) Methods only used by only rl
        #
        #
        #

    def filter_dm_info_actuator_space_with_modal_basis(self,
                                                       dm_info_to_filter,
                                                       add_tip_tilt_to_not_break=False,
                                                       return_tt=False):
        """
        Filter modes
        """

        if self.device > -1:
            if isinstance(dm_info_to_filter, np.ndarray):
                dm_info_to_filter = torch.FloatTensor(dm_info_to_filter).to(self.device)
            else:
                dm_info_to_filter = dm_info_to_filter.to(self.device)
            dm_info_filtered = self.filter_dm_info_actuator_space_with_modal_basis_torch(dm_info_to_filter,
                                                                                         add_tip_tilt_to_not_break,
                                                                                         return_tt)
        else:
            dm_info_filtered = self.filter_dm_info_actuator_space_with_modal_basis_numpy(dm_info_to_filter,
                                                                                         add_tip_tilt_to_not_break,
                                                                                         return_tt)

        return dm_info_filtered

    @torch.no_grad()
    def filter_dm_info_actuator_space_with_modal_basis_torch(self,
                                                             dm_info_to_filter,
                                                             add_tip_tilt_to_not_break=False,
                                                             return_tt=False):
        """
        Filter modes in torch
        """

        add_tip_tilt_to_not_break_bool = self.num_dm > 1 and\
                                         add_tip_tilt_to_not_break

        if add_tip_tilt_to_not_break_bool:
            dm_info_to_filter = torch.cat([dm_info_to_filter, self.zeros_tt])

        dm_info_to_filter_modes = self.volts2modes_torch@dm_info_to_filter
        dm_info_to_filter_modes[self.modes_filtered] = 0
        dm_info_filtered = self.modes2volts_torch@dm_info_to_filter_modes

        if add_tip_tilt_to_not_break_bool and not return_tt:
            dm_info_filtered = dm_info_filtered[:-2]

        return dm_info_filtered.cpu().numpy()

    def filter_dm_info_actuator_space_with_modal_basis_numpy(self,
                                                            dm_info_to_filter,
                                                            add_tip_tilt_to_not_break=False,
                                                            return_tt=False):
        """
        Filter modes in numpy
        """

        add_tip_tilt_to_not_break_bool = self.num_dm > 1 and\
                                         add_tip_tilt_to_not_break

        if add_tip_tilt_to_not_break_bool:
            dm_info_to_filter = np.concatenate([dm_info_to_filter, [0.0, 0.0]])

        dm_info_to_filter_modes = self.volts2modes.dot(dm_info_to_filter)
        dm_info_to_filter_modes[self.modes_filtered] = 0
        dm_info_filtered = self.modes2volts.dot(dm_info_to_filter_modes)

        if add_tip_tilt_to_not_break_bool and not return_tt:
            dm_info_filtered = dm_info_filtered[:-2]

        return dm_info_filtered

    def correction_control(self,
                           action: np.ndarray,
                           unet_or_phase_command: np.ndarray = None):
        """
        Control for correction only
        """

        if unet_or_phase_command is None:
            err = self.rtc.get_err(0)
        else:
            err = unet_or_phase_command.copy()
        if self.reduce_gain_tt_to > -1:
            err[-2:] *= self.reduce_gain_tt_to

        final_command_actuator = self.past_command + err

        if self.num_dm > 1:
            if self.control_tt:
                # Command has shape hddm + tt. action hddm + tt
                final_command_actuator += action
            else:
                # Command has shape hddm + tt. action hddm
                final_command_actuator[:-2] += action
            # We clip for otherwise it can diverge
            final_command_actuator[:-2] = np.clip(final_command_actuator[:-2],
                                                  -self.command_clip_value,
                                                  self.command_clip_value)
            if self.reset_when_clip:
                if (np.abs(final_command_actuator[:-2]) >= self.command_clip_value).any():
                    final_command_actuator = self.reset_clipped_actuator(final_command_actuator)
        else:
            raise NotImplementedError

        if self.filter_commands:
            final_command_actuator = self.filter_dm_info_actuator_space_with_modal_basis(final_command_actuator)

        return final_command_actuator

    def only_rl_control(self,
                        action: np.ndarray):
        """
        Control for RL only
        """

        if self.num_dm > 1:
            if self.control_tt:
                # Command has shape hddm + tt. action hddm + tt
                final_command_actuator = self.past_command + action
            else:
                # Command has shape hddm + tt. action hddm
                # Only control hddm mirror
                final_command_actuator = self.past_command[:-2] + action
                # Concatenate tt command created by integrator
                commands_tip_tilt = self.rtc.get_command(0)[-2:]
                final_command_actuator = np.concatenate([final_command_actuator, commands_tip_tilt])

            # Only clip hddm mirror
            final_command_actuator[:-2] = np.clip(final_command_actuator[:-2],
                                                  -self.command_clip_value,
                                                  self.command_clip_value)
            if self.reset_when_clip:
                if (np.abs(final_command_actuator[:-2]) >= self.command_clip_value).any():
                    final_command_actuator = self.reset_clipped_actuator(final_command_actuator)
        else:
            raise NotImplementedError
        # Filter if necessary
        if self.filter_commands:
            final_command_actuator = self.filter_dm_info_actuator_space_with_modal_basis(final_command_actuator)
        return final_command_actuator

    #
    #   ___  ___         __  __     _   _            _
    #  |   \|  |        |  \/  |___| |_| |_  ___  __| |___
    #  | |\    |        | |\/| / -_)  _| ' \/ _ \/ _` (_-<
    #  |_|  \__|  ext   |_|  |_\___|\__|_||_\___/\__,_/__/
    #

    def raytrace_target(self, ncontrol: int):
        """
        Does the raytacing operation
        :param ncontrol: ncontrol that will have an associated target
        :return: None
        """
        t = ncontrol
        if self.atmos.is_enable:
            self.target.raytrace(t, tel=self.tel, atm=self.atmos, dms=self.dms)
        else:
            self.target.raytrace(t, tel=self.tel, dms=self.dms)

    #
    #           II a) PART TWO METHODS
    #

    def move_dm_and_compute_strehl(self,
                                   action: np.ndarray,
                                   controller_type: str = "RL",
                                   unet_or_phase_command: np.ndarray = None
                                   ):
        """
        Move DM and compute SR based on the controller type.
        """

        w = 0
        tar_trace = range(len(self.config.p_targets))
        if controller_type in ["RL", "UNet+Linear-withRLmethod", "Phase-withRLmethod"]:
            self.rl_control(action, w, unet_or_phase_command, controller_type)
        elif controller_type in ["UNet", "UNet+Linear", "Phase"]:
            if self.reduce_gain_tt_to > -1:
                action[-2:] *= self.reduce_gain_tt_to
            command = self.past_command + action
            self.past_command = command
            self.rtc.set_command(0, command)
        elif controller_type == "Linear":
            pass
        else:
            raise NotImplementedError

        self.rtc.apply_control(w)

        for tar_index in tar_trace:
            self.target.comp_tar_image(tar_index)
            self.target.comp_strehl(tar_index)

        self.iter += 1

    def tt2hddm(self, dm_tt):
        """
        From TT project to HDDM
        Args:
            dm_tt: shape (2,)

        Returns: dm_hddm_from_tt shape (hddm.shape,)
        """
        # self.tt_shape to phase shape
        phase = self.actuatortt2phase.dot(dm_tt)
        # remove piston
        phase = phase - phase.mean()
        # phase to self.tt_shape
        dm_hddm_from_tt = self.phase2actuatorhddm.dot(phase)
        return dm_hddm_from_tt

    def move_atmos_compute_wfs_reconstruction(self) -> None:
        """
        Move atmos and compute WFS image and linear rec.
        """
        w = 0
        self.atmos.move_atmos()

        self.raytrace_target(w)
        self.wfs.raytrace(w, tel=self.tel, atm=self.atmos)
        self.wfs.raytrace(w, dms=self.dms, ncpa=False, reset=False)
        self.wfs.compute_wfs_image(w)
        self.rtc.do_centroids(w)
        self.rtc.do_control(w)
