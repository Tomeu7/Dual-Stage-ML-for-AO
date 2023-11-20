from shesha.supervisor.genericSupervisor import GenericSupervisor
from shesha.supervisor.compassSupervisor import CompassSupervisor
from shesha.supervisor.components import AtmosCompass, DmCompass, RtcCompass, TargetCompass, TelescopeCompass, \
    WfsCompass
from shesha.supervisor.optimizers import ModalBasis, Calibration
import shesha.ao.basis as bas
import numpy as np


class RlSupervisor(CompassSupervisor):
    def __init__(self, config, n_reverse_filtered_from_cmat, *,
                 include_tip_tilt=False,
                 filter_commands=True,
                 command_filter_value=500,
                 initial_seed=1234,
                 which_modal_basis="Btt",
                 mode="only_rl"):
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
        self.command_filter_value = command_filter_value
        self.num_dm = len(self.config.p_dms)
        self.mode = mode
        assert self.mode in ["correction", "only_rl"]
        self.modes2volts, self.volts2modes = self.basis.compute_modes_to_volts_basis(modal_basis_type=which_modal_basis)
        self.modes_filtered = self.manage_filtered_modes(self.modes2volts, n_reverse_filtered_from_cmat)
        self.obtain_and_set_cmat_filtered(self.modes2volts, n_reverse_filtered_from_cmat)
        # Projector 1D->2D, 2D->1D and mask_valid_actuators
        self.projector_volts1d_to_volts2d, self.projector_volts2d_to_volts1d =\
            self.create_projector_volts1d_to_volts2d()
        # Save past commands and past actions
        self.past_command_rl = None
        self.past_action_rl = None
        # dm/action shapes
        self.command_shape = self.rtc.get_command(0).shape  # always in 1D
        if self.num_dm > 1:
            if include_tip_tilt:
                raise NotImplementedError
            else:
                self.action_1d_shape = self.rtc.get_command(0)[:-2].shape
                self.action_2d_shape = self.apply_projector_volts1d_to_volts2d(np.ones(self.action_1d_shape)).shape
        else:
            self.action_1d_shape = self.rtc.get_command(0).shape
            self.action_2d_shape = self.apply_projector_volts1d_to_volts2d(np.ones(self.action_1d_shape)).shape
        # Mask valid actuators
        self.mask_valid_actuators = self.apply_projector_volts1d_to_volts2d(np.ones(self.action_1d_shape))


    #
    #   __             __  __     _   _            _
    #  |   \          |  \/  |___| |_| |_  ___  __| |___
    #  |   |          | |\/| / -_)  _| ' \/ _ \/ _` (_-<
    #  |__/ efault    |_|  |_\___|\__|_||_\___/\__,_/__/
    #

    def create_projector_volts1d_to_volts2d(self):
        """
        Creates a projector for the command_vector to go into a 2D array which would take their according positions.
        :return: projector
        """

        def get_linspace_for_projection_volts1d_to_2d():
            # Required information
            x_pos_actuators = self.config.p_dms[0]._xpos

            y_pos_actuators = self.config.p_dms[0]._ypos
            x_pos_linspace = np.linspace(self.config.p_geom._p1 - 0.5,
                                         self.config.p_geom._p2 + 0.5,
                                         self.config.p_dms[0].nact)
            y_pos_linspace = np.linspace(self.config.p_geom._p1 - 0.5,
                                         self.config.p_geom._p2 + 0.5,
                                         self.config.p_dms[0].nact)

            return x_pos_linspace, y_pos_linspace, x_pos_actuators, y_pos_actuators

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

    def manage_filtered_modes(self, modes2volts, n_reverse_filtered_from_cmat):

        if self.num_dm > 1:
            modes_filtered = np.arange(modes2volts.shape[1] - n_reverse_filtered_from_cmat - 2,
                                       modes2volts.shape[1] - 2)
        else:
            modes_filtered = np.arange(modes2volts.shape[1] - n_reverse_filtered_from_cmat,
                                       modes2volts.shape[1])

        return modes_filtered

    def add_one_to_seed(self):
        self.current_seed += 1

    def obtain_and_set_cmat_filtered(self, btt2act, n_reverse_filtered_from_cmat):

        print("+ Setting cmat filtered...")

        if n_reverse_filtered_from_cmat > -1:
            print("Using Btt basis")
            print("+ Shape Btt basis", btt2act.shape)
            assert type(n_reverse_filtered_from_cmat) == int
            cmat = bas.compute_cmat_with_Btt(rtc=self.rtc._rtc,
                                             Btt=btt2act,
                                             nfilt=n_reverse_filtered_from_cmat)
            print("+ Number of filtered modes", n_reverse_filtered_from_cmat)
        else:
            print("+ WARNING: NO CMAT BUILT FROM BTT, IT IS THE ORIGINAL WITHOUT FILTERING ANY MODES!!!")
            cmat = None

        return cmat

    def apply_projector_volts2d_to_volts1d(self, command_2d):
        command_1d = command_2d.flatten().dot(self.projector_volts2d_to_volts1d)
        return command_1d

    def apply_projector_volts1d_to_volts2d(self, command_1d):
        command_2d = command_1d.dot(self.projector_volts1d_to_volts2d).reshape(self.config.p_dms[0].nact, self.config.p_dms[0].nact)
        return command_2d

    def reset(self, only_reset_dm):
        """ Reset the simulation to return to its original state
        """
        if not only_reset_dm:
            self.atmos.reset_turbu(self.current_seed)
            self.wfs.reset_noise(self.current_seed)
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
        self.past_command_rl = np.zeros(self.action_1d_shape)
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
                   ncontrol: int):

        if self.mode == "correction":
            final_command = self.correction_control(action=action)
        else:
            final_command = self.only_rl_control(action=action)

        if self.num_dm > 1:
            self.past_command_rl = final_command[:-2].copy()
            self.past_action_rl = action
        else:
            self.past_command_rl = final_command
            self.past_action_rl = action

        self.rtc.set_command(ncontrol, final_command)
        # print("action_in_env", action[0], final_command[0])

        #
        #       II b) Methods only used by only rl
        #
        #
        #

    def filter_dm_info_actuator_space_with_modal_basis(self,
                                                       dm_info_to_filter,
                                                       add_tip_tilt_to_not_break=False):
        """
        Filter information in respect the DM with the Btt modes
        :param dm_info_to_filter: current unfiltered DM information
        :return: dm_info_filtered
        """

        add_tip_tilt_to_not_break_bool = self.num_dm > 1 and\
                                         add_tip_tilt_to_not_break

        if add_tip_tilt_to_not_break_bool:
            dm_info_to_filter = np.concatenate([dm_info_to_filter, [0.0, 0.0]])

        dm_info_to_filter_modes = self.volts2modes.dot(dm_info_to_filter)
        dm_info_to_filter_modes[self.modes_filtered] = 0
        dm_info_filtered = self.modes2volts.dot(dm_info_to_filter_modes)

        if add_tip_tilt_to_not_break_bool:
            dm_info_filtered = dm_info_filtered[:-2]

        return dm_info_filtered

    def correction_control(self,
                           action: np.ndarray):

        final_command_actuator = self.rtc.get_command(0)  # We get the command of the integrator

        if self.num_dm > 1:
            final_command_actuator[:-2] += action
            # We clip for otherwise it can diverge
            final_command_actuator[:-2] = np.clip(final_command_actuator[:-2],
                                                  -self.command_filter_value,
                                                  self.command_filter_value)
        else:
            final_command_actuator += action
            # We clip for otherwise it can diverge
            final_command_actuator = np.clip(final_command_actuator,
                                             -self.command_filter_value,
                                             self.command_filter_value)

        if self.filter_commands:
            final_command_actuator = self.filter_dm_info_actuator_space_with_modal_basis(action)

        return final_command_actuator

    def only_rl_control(self,
                        action: np.ndarray):

        final_command_actuator = self.past_command_rl + action

        # Clip otherwise it can diverge. Also do it before concatenating TT commands because we do not control it
        final_command_actuator = np.clip(final_command_actuator,
                                         -self.command_filter_value,
                                         self.command_filter_value)

        if self.num_dm > 1:
            commands_tip_tilt = self.rtc.get_command(0)[-2:]
            final_command_actuator = np.concatenate([final_command_actuator, commands_tip_tilt])


        if self.filter_commands:
            final_command_actuator = self.filter_dm_info_actuator_space_with_modal_basis(action)

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
                                   controller_type: str = "RL"
                                   ):

        w = 0
        tar_trace = range(len(self.config.p_targets))
        if controller_type == "RL":
            self.rl_control(action, w)
        elif controller_type == "Integrator":
            pass
        else:
            raise NotImplementedError

        self.rtc.apply_control(w)

        for tar_index in tar_trace:
            self.target.comp_tar_image(tar_index)
            self.target.comp_strehl(tar_index)

        self.iter += 1

    #
    #           II b) PART ONE METHODS
    #
    #

    def move_atmos_compute_wfs_reconstruction(self) -> None:

        w = 0
        self.atmos.move_atmos()

        self.raytrace_target(w)
        self.wfs.raytrace(w, tel=self.tel, atm=self.atmos)
        self.wfs.raytrace(w, dms=self.dms, ncpa=False, reset=False)
        self.wfs.compute_wfs_image(w)
        self.rtc.do_centroids(w)
        self.rtc.do_control(w)
