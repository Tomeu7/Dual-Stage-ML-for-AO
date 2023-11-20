import numpy as np
import os

FOLDER_PROJECTORS = "data/projectors/"


def compute_phase_from_command(supervisor, send_to_dm, dm_index, pupil,
                               return_phase_original=False):
    """
    Compute the phase that a command produces
    Args:
        supervisor: supervisor object
        send_to_dm: commands we send to DM
        dm_index: dm index
        pupil: the pupiL
        return_phase_original: if we also return the full phase without removing mean and in 2D
    """
    supervisor.dms._dms.d_dms[dm_index].set_com(send_to_dm)
    supervisor.dms._dms.d_dms[dm_index].comp_shape()
    supervisor.target.raytrace(index=0, dms=supervisor.dms, ncpa=False, reset=True)
    phase_original = supervisor.target.get_tar_phase(0)

    phase = phase_original[np.where(pupil)]
    phase = phase - phase.mean()

    if return_phase_original:
        return phase, phase_original
    else:
        return phase


class ProjectorCreator:
    def __init__(self,
                 parameter_file,
                 supervisor,
                 second_dm_index=1):
        self.supervisor = supervisor
        self.modes2volts = supervisor.modes2volts
        self.pupil = supervisor.get_s_pupil()
        self.where_is_pupil = np.where(self.pupil)
        self.second_dm_index = second_dm_index
        self.path_projectors = FOLDER_PROJECTORS + parameter_file[:-3]
        self.num_dm = supervisor.num_dm
        self.command_shape = self.supervisor.command_shape

    def get_projector_targetphase2modes(self, basis="btt"):
        """
        Compute projector from target phase to modes and saves it as a np.array
        """
        print("Basis", basis)
        if basis == "btt":
            if os.path.exists(self.path_projectors + "_btt_phase2modes.npy"):
                modes2phase = np.load(self.path_projectors + "_btt_modes2phase.npy")
                phase2modes = np.load(self.path_projectors + "_btt_phase2modes.npy")
            else:
                print("Computing projector")
                if basis == "btt" and self.num_dm == 2:
                    modes2phase, phase2modes = self.get_btt2targetphase_2dm()
                elif basis == "btt" and self.num_dm == 1:
                    raise AssertionError
                else:
                    raise NotImplementedError
                np.save(self.path_projectors + "_btt_modes2phase.npy", modes2phase)
                np.save(self.path_projectors + "_btt_phase2modes.npy", phase2modes)
        else:
            raise NotImplementedError
        return modes2phase, phase2modes

    def get_btt2targetphase_2dm(self):
        """
        Compute projector btt to phase and phase to btt
        """
        self.supervisor.reset(only_reset_dm=False)
        modes2phase = np.zeros((self.where_is_pupil[0].shape[0], self.modes2volts.shape[1]), np.float32)
        for mode in range(self.modes2volts.shape[1]):
            print("Obtaining projector btt -> phase, btt mode:", mode)
            send_to_dm = self.modes2volts[:, mode]  # This has shape volts x modes
            if mode < (self.modes2volts.shape[1] - 2):
                phase = compute_phase_from_command(self.supervisor,
                                                   send_to_dm[:-2], dm_index=0,
                                                   pupil=self.pupil)
            else:
                phase = compute_phase_from_command(self.supervisor,
                                                   send_to_dm[-2:], dm_index=self.second_dm_index,
                                                   pupil=self.pupil)

            modes2phase[:, mode] = phase.copy().reshape(-1)
            self.supervisor.reset(only_reset_dm=False)

        phase2modes = np.linalg.pinv(modes2phase)

        return modes2phase, phase2modes

    def get_projector_targetphase2actuator(self):
        """
        Compute projector from target phase to actuators and saves it as a np.array
        """
        if os.path.exists(self.path_projectors + "_actuator_actuator2phase.npy"):
            actuator2phase = np.load(self.path_projectors + "_actuator_actuator2phase.npy")
            phase2actuator = np.load(self.path_projectors + "_actuator_phase2actuator.npy")
            actuatorpzt2phase = np.load(self.path_projectors + "_actuator_actuatorpzt2phase.npy")
            phase2actuatorpzt = np.load(self.path_projectors + "_actuator_phase2actuatorpzt.npy")
            actuatortt2phase = np.load(self.path_projectors + "_actuator_actuatortt2phase.npy")
            phase2actuatortt = np.load(self.path_projectors + "_actuator_phase2actuatortt.npy")
        else:
            print("Computing projector")

            actuator2phase, phase2actuator, actuatorpzt2phase, phase2actuatorpzt, actuatortt2phase, phase2actuatortt \
                = self.get_act2targetphase_2dm()

            np.save(self.path_projectors + "_actuator_actuator2phase.npy", actuator2phase)
            np.save(self.path_projectors + "_actuator_phase2actuator.npy", phase2actuator)
            np.save(self.path_projectors + "_actuator_actuatorpzt2phase.npy", actuatorpzt2phase)
            np.save(self.path_projectors + "_actuator_phase2actuatorpzt.npy", phase2actuatorpzt)
            np.save(self.path_projectors + "_actuator_actuatortt2phase.npy", actuatortt2phase)
            np.save(self.path_projectors + "_actuator_phase2actuatortt.npy", phase2actuatortt)

        return actuator2phase, phase2actuator, actuatorpzt2phase, phase2actuatorpzt, actuatortt2phase, phase2actuatortt

    def get_act2targetphase_2dm(self):
        """
        Compute projector from target phase to actuators
        """
        self.supervisor.reset(only_reset_dm=False)
        actuator2phase = np.zeros((self.where_is_pupil[0].shape[0], self.command_shape[0]), np.float32)
        actuatorpzt2phase = np.zeros((self.where_is_pupil[0].shape[0], self.command_shape[0]-2), np.float32)
        actuatortt2phase = np.zeros((self.where_is_pupil[0].shape[0], 2), np.float32)
        for actuator in range(self.command_shape[0]):
            print("Obtaining projector act -> phase, act number:", actuator)
            send_to_dm = np.zeros(self.command_shape)
            send_to_dm[actuator] += 1  # This has shape volts x modes
            if actuator < (self.command_shape[0] - 2):
                phase = compute_phase_from_command(self.supervisor,
                                                   send_to_dm[:-2], dm_index=0,
                                                   pupil=self.pupil)
                actuatorpzt2phase[:, actuator] = phase.copy().reshape(-1)
            else:
                phase = compute_phase_from_command(self.supervisor,
                                                   send_to_dm[-2:], dm_index=self.second_dm_index,
                                                   pupil=self.pupil)

                actuatortt2phase[:, actuator - (self.command_shape[0] - 2)] = phase.copy().reshape(-1)
            actuator2phase[:, actuator] = phase.copy().reshape(-1)
            self.supervisor.reset(only_reset_dm=False)

        phase2actuatorpzt = np.linalg.pinv(actuatorpzt2phase)
        phase2actuatortt = np.linalg.pinv(actuatortt2phase)
        phase2actuator = np.linalg.pinv(actuator2phase)

        return actuator2phase, phase2actuator, actuatorpzt2phase, phase2actuatorpzt, actuatortt2phase, phase2actuatortt