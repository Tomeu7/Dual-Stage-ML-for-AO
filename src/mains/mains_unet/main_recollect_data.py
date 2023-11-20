# main_recollect_data.py
import argparse
import pandas as pd
import numpy as np
import random
import os
from tqdm import tqdm
from src.config import obtain_config_env_default
from src.env_methods.env import AoEnv
import h5py

def loguniform(low=0., high=1., size=None):
    return np.exp(np.random.uniform(low, high, size))

def next_phase_from_mirror(sup_, current_voltage_):
    """
    Raytrace through the atmosphere and apply the specified mirror shape (in volts) to the DM
    """

    sup_.rtc.set_command(controller_index=0, com=current_voltage_)
    sup_.rtc.apply_control(0, comp_voltage=False)

    sup_.target.raytrace(0, tel=sup_.tel, dms=sup_.dms)

    sup_.wfs.raytrace(0, tel=sup_.tel)
    if not sup_.config.p_wfss[0].open_loop and sup_.dms is not None:
        sup_.wfs.raytrace(0, dms=sup_.dms, ncpa=False, reset=False)
    sup_.wfs.compute_wfs_image(0)

    sup_.rtc.compute_slopes(0)
    sup_.rtc.do_control(0)

    sup_.target.comp_tar_image(0)
    sup_.target.comp_strehl(0)

    sup_.iter += 1

def poke_uniform_actuators(dm_shape_, args_):
    """
    Generates random uniform aberration and sets in the DM channel.
    Otherwise we use a log_uniform distribution.
    """

    new_var = loguniform(low=args_.min_loguniform,
                         high=args_.max_loguniform,
                         size=1)

    new_commands = np.random.normal(loc=0.0, scale=new_var, size=dm_shape_)

    new_commands = np.clip(new_commands, a_min=-args_.clip, a_max=args_.clip).astype(np.float32)

    return new_commands

def initialize_folders_and_compass():
    """
    Initialization with arguments
    """
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--parameter_file', default="pyr_40x40_8m_M0_n3.py")
    parser.add_argument('--data_size', type=int, default=200000)
    parser.add_argument('--seed', type=int, default=100)
    parser.add_argument('--dataset_name', type=str)
    parser.add_argument('--train_dataset_percentage', help='Train/Evaluation split', default=0.8, type=float)
    parser.add_argument('--min_loguniform', type=float, help="e.g. 0.0001", default=0.5)
    parser.add_argument('--max_loguniform', type=float, help="e.g. 0.5", default=6)
    parser.add_argument('--clip', type=float, help="e.g. 1.0", default=1000.0)
    parser.add_argument("--dont_get_the_phase", action="store_true")
    parser.add_argument("--pyr_gpu_id", default=0, type=int)
    parser.add_argument("--n_modes_filtered", default=100, type=int)
    parser.add_argument("--path_to_data", type=str, default="/raid/pbartomeu/outputs/output_dataset_gan/test_example/")
    args_ = parser.parse_args()
    conf_env = obtain_config_env_default(args_.parameter_file, args_.n_modes_filtered)
    conf_env['control_tt'] = True  # We need projections therefore we need to set up a variable to True
    # TODO implement better
    env_ = AoEnv(config_env_rl=conf_env,
                 parameter_file=args_.parameter_file,
                 seed=args_.seed,
                 device=args_.pyr_gpu_id,
                 pyr_gpu_ids=[args_.pyr_gpu_id])
    sup_ = env_.supervisor

    # Create directories for data
    path_to_data_ = args_.path_to_data
    path_to_data_train_ = path_to_data_ + "train/"
    path_to_data_eval_ = path_to_data_ + "evaluation/"
    if not os.path.exists(path_to_data_train_):
        os.makedirs(path_to_data_train_)
        os.mkdir(path_to_data_eval_)

    np.random.seed(args_.seed)
    random.seed(args_.seed)

    dm_shape_ = sup_.rtc.get_command(0).shape
    pupil_mask_ = sup_.get_s_pupil()
    np.save(path_to_data_ + "mask_valid_commands.npy", env_.mask_valid_actuators)
    np.save(path_to_data_ + "pupil.npy", pupil_mask_)

    # For statistics
    log_ = pd.DataFrame(columns=['Max wfs', 'Min wfs', 'Max phase', 'Min phase', 'Max voltage', 'Min voltage'])
    log_normalize_ = pd.DataFrame(columns=['Max wfs', 'Min wfs', 'Max phase', 'Min phase', 'Max voltage', 'Min voltage'])
    wfs_max_, wfs_min_, phase_max_, phase_min_, voltage_max_, voltage_min_ =\
        0, float("inf"), 0, float("inf"), 0, float("inf")
    wfs_norm_max_, wfs_norm_min_ = 0, float("inf")

    print("--------------INFO----------------")
    print("S. Pupil shape ", sup_.get_s_pupil().shape)
    print("M. Pupil shape ", sup_.get_m_pupil().shape)
    print("WFS Image shape", sup_.wfs.get_wfs_image(0).shape)
    print("Phase Shape", sup_.target.get_tar_phase(0).shape)
    print("DM Shape", env_.mask_valid_actuators.shape)
    print("----------------------------------")

    return sup_, env_, args_, path_to_data_, path_to_data_train_, path_to_data_eval_, log_, log_normalize_,\
           wfs_max_, wfs_min_, phase_max_, phase_min_, voltage_max_, voltage_min_,\
           wfs_norm_max_, wfs_norm_min_, args_.dont_get_the_phase, dm_shape_, pupil_mask_

def manage_norm_params(frame_, dont_get_the_phase,
                       wfs_image_, wfs_max_, wfs_min_,
                       current_voltage_, voltage_max_, voltage_min_,
                       phase_, phase_max_, phase_min_):
    """
    Keep track of max and min of different data values
    """

    wfs_max_ = max(wfs_max_, wfs_image_.max())
    wfs_min_ = min(wfs_min_, wfs_image_.min())
    if dont_get_the_phase:
        phase_max_ = None
        phase_min_ = None
    else:
        phase_max_ = max(phase_max_, phase_.max())
        phase_min_ = min(phase_min_, phase_.min())
    voltage_max_ = max(voltage_max_, current_voltage_.max())
    voltage_min_ = min(voltage_min_, current_voltage_.min())

    if frame % 500 == 0:
        if dont_get_the_phase:
            phase_to_print = None
        else:
            phase_to_print = phase_.std()
        print(
            "Frame {} Phase rms {} "
            "Wfs max {} Wfs min {} "
            "voltage max {} voltage min {} "
            "Phase max {} Phase min {}".format(frame_,
                                               phase_to_print,
                                               wfs_max_, wfs_min_,
                                               voltage_max_, voltage_min_,
                                               phase_max_, phase_min_))

    return wfs_max_, wfs_min_, voltage_max_, voltage_min_, phase_max_, phase_min_

def save_data(path_, wfs_image_, phase_, commands_to_save_):
    file = h5py.File(path_, 'w')
    file.create_dataset('arr_0', data=wfs_image_)
    file.create_dataset('arr_1', data=phase_)
    file.create_dataset('arr_2', data=commands_to_save_)
    file.close()

if __name__ == "__main__":
    # Assert we are in hardware mode
    sup, env, args, path_to_data, path_to_data_train, path_to_data_eval, log, log_normalize,\
        wfs_max, wfs_min, phase_max, phase_min, voltage_max, voltage_min,\
        wfs_norm_max, wfs_norm_min, dont_get_phase, dm_shape, pupil_mask = initialize_folders_and_compass()

    for frame in tqdm(range(args.data_size)):

        # a) Input random perturbation
        current_voltage = poke_uniform_actuators(dm_shape, args)

        # b) Move mirror
        next_phase_from_mirror(sup, current_voltage_=current_voltage)

        # c) Save it in training or evaluation folder
        wfs_image = sup.wfs.get_wfs_image(0)

        commands_pzt = current_voltage[:-2]
        commands_tt = env.tt2hddm(current_voltage[-2:])
        commands_to_save = sup.apply_projector_volts1d_to_volts2d(commands_pzt + commands_tt)

        if dont_get_phase:
            # We just set the phase to 0
            phase = 0.
        else:
            phase = sup.target.get_tar_phase(0)
            phase = np.multiply(phase, pupil_mask)
            phase -= phase[pupil_mask == 1].mean()
        if frame < args.train_dataset_percentage * args.data_size:
            save_data(path_to_data_train + "0_" + str(frame) + '.hdf5', wfs_image, phase, commands_to_save)
            wfs_max, wfs_min, voltage_max, voltage_min, phase_max, phase_min = manage_norm_params(frame,
                                                                                                  dont_get_phase,
                                                                                                  wfs_image,
                                                                                                  wfs_max,
                                                                                                  wfs_min,
                                                                                                  current_voltage,
                                                                                                  voltage_max,
                                                                                                  voltage_min,
                                                                                                  phase,
                                                                                                  phase_max,
                                                                                                  phase_min)
        else:
            save_data(path_to_data_eval + "0_" + str(frame) + '.hdf5', wfs_image, phase, commands_to_save)

    log = log.append({'Max wfs': wfs_max, 'Min wfs': wfs_min,
                      'Max phase': phase_max, 'Min phase': phase_min,
                      'Max voltage': voltage_max, 'Min voltage': voltage_min,
                      'Max loguniform': args.max_loguniform, 'Min loguniform': args.min_loguniform},
                     ignore_index=True)
    log.to_csv(path_to_data + "info.csv", index=False)