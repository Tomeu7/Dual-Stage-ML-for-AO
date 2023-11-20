# helper.py
import os
import json
import pandas as pd
import numpy as np
import torch
import random
from src.rl_agent.sac import Sac as Agent
from src.config import obtain_config_env
from src.config import obtain_config_agent
from src.env_methods.env_non_linear import AoEnvNonLinear
from src.env_methods.env import AoEnv


def choose_exploration_exploitation(evaluation_after_steps, total_step, exploitation):
    """
    Chooses between exploration and exploitation
    Args:
        evaluation_after_steps: when are we changing to evaluation
        total_step: current step
        exploitation: current value of exploitation

    Returns: bool

    """
    if total_step == evaluation_after_steps:
        exploitation = True
        print("Changing mode to exploitation", exploitation)

    return exploitation


def initialise(args_):
    """
    Initializes the environment and the RL agent
    Args:
        args_: arguments to start

    Returns:
        Environment object
        RL agent object
        Config dictionary for environment
        Config dictionary for agent
    """
    parameter_file_ = args_.parameter_file
    pure_deterministic_ = args_.pure_deterministic
    controller_type_ = args_.controller_type

    if args_.integrator_exploration_with_only_rl_for > -1:
        assert args_.noise_for_exploration > -1, "If exploration activated you must provide noise value"
    if any([args_.s_dm_residual_non_linear, args_.s_dm_residual_non_linear_tt]):
        assert args_.unet_dir is not None and args_.unet_name is not None, print(args_.unet_dir, args_.unet_name)
    # Configs
    config_env_rl_ = obtain_config_env(parameter_file=parameter_file_,
                                       number_of_modes_filtered=args_.number_of_modes_filtered,
                                       mode=args_.mode,
                                       control_tt=args_.control_tt,
                                       s_dm_residual=args_.s_dm_residual,
                                       s_dm_tt=args_.s_dm_tt,
                                       s_dm_residual_tt=args_.s_dm_residual_tt,
                                       s_dm=not args_.no_s_dm,
                                       s_dm_residual_rl=args_.s_dm_residual_rl,
                                       number_of_previous_s_dm=args_.number_of_previous_s_dm,
                                       number_of_previous_s_dm_residual_rl=args_.number_of_previous_s_dm_residual_rl,
                                       number_of_previous_s_dm_residual_tt=args_.number_of_previous_s_dm_residual_tt,
                                       number_of_previous_s_dm_tt=args_.number_of_previous_s_dm_tt,
                                       s_dm_residual_non_linear=args_.s_dm_residual_non_linear,
                                       number_of_previous_s_dm_residual_non_linear=args_.number_of_previous_s_dm_residual_non_linear,
                                       s_dm_residual_non_linear_tt=args_.s_dm_residual_non_linear_tt,
                                       number_of_previous_s_dm_residual_non_linear_tt=args_.number_of_previous_s_dm_residual_non_linear_tt,
                                       action_scale=args_.action_scale,
                                       value_action_penalizer=args_.value_action_penalizer,
                                       delayed_assignment=args_.delayed_assignment,
                                       reduce_gain_tt_to=args_.reduce_gain_tt_to)
    # For environment
    config_agent_ = obtain_config_agent(agent_type="sac",
                                        gamma=args_.gamma)  # for sac
    # Setting seed in libraries - seed in the env will be set up when we create the object
    torch.manual_seed(args_.seed)
    np.random.seed(args_.seed)
    random.seed(args_.seed)
    if pure_deterministic_:  # maybe this make it slower, only needed if we use convolutions
        torch.backends.cudnn.deterministic = True

    # Environment
    if args_.unet_dir is not None and args_.unet_name is not None:
        # Environment
        env_ = AoEnvNonLinear(unet_dir=args_.unet_dir,
                              unet_name=args_.unet_name,
                              unet_type="volts",
                              device_unet="cuda:" + str(args_.device),
                              gain_factor_unet=None,  # we set it up later
                              gain_factor_linear=None,  # we set it up later
                              normalization_095_005=True,
                              config_env_rl=config_env_rl_,
                              parameter_file=parameter_file_,
                              seed=args_.seed,
                              device=args_.device,
                              normalization_noise_unet=True,
                              normalization_noise_value_unet=config_env_rl_['normalization_noise_value_unet'])
    else:
        env_ = AoEnv(config_env_rl=config_env_rl_,
                     parameter_file=parameter_file_,
                     seed=args_.seed,
                     device=args_.device)

    set_r0_and_gain(args_.r0, args_.parameter_file, env_, args_.gain_linear, args_.gain_non_linear)

    # Agent
    if controller_type_ == "RL":
        agent_ = Agent(config_agent=config_agent_,
                       state_channels=env_.state_size_channel_0,
                       action_shape=env_.action_2d_shape,
                       device=args_.device,
                       mask_valid_actuators=env_.mask_valid_actuators,
                       two_output_critic=not config_env_rl_['joint_tt_into_reward'] and config_env_rl_['control_tt'],
                       two_output_actor=config_env_rl_['separate_tt_into_two_actions'] and config_env_rl_['control_tt'])
    else:
        agent_ = None

    return env_, agent_, config_env_rl_, config_agent_

def get_gains():
    """
    Hardcoded gain for each magnitude/r0
    Returns: dict of gains
    """

    dict_gains = {}
    dict_gains["pyr_40x40_8m_M0_n3.py"] = {}
    dict_gains["pyr_40x40_8m_M9_n3.py"] = {}

    dict_gains["pyr_40x40_8m_M0_n3.py"][0.08] = {}
    dict_gains["pyr_40x40_8m_M9_n3.py"][0.12] = {}
    dict_gains["pyr_40x40_8m_M0_n3.py"][0.16] = {}
    dict_gains["pyr_40x40_8m_M9_n3.py"][0.08] = {}
    dict_gains["pyr_40x40_8m_M0_n3.py"][0.12] = {}
    dict_gains["pyr_40x40_8m_M9_n3.py"][0.16] = {}

    dict_gains["pyr_40x40_8m_M0_n3.py"][0.08]["linear_integrator"] = 0.99
    dict_gains["pyr_40x40_8m_M0_n3.py"][0.08]["linear_combination"] = 0.95
    dict_gains["pyr_40x40_8m_M0_n3.py"][0.08]["non_linear_combination"] = 0.3

    dict_gains["pyr_40x40_8m_M0_n3.py"][0.12]["linear_integrator"] = 0.99
    dict_gains["pyr_40x40_8m_M0_n3.py"][0.12]["linear_combination"] = 0.9
    dict_gains["pyr_40x40_8m_M0_n3.py"][0.12]["non_linear_combination"] = 0.2

    dict_gains["pyr_40x40_8m_M0_n3.py"][0.16]["linear_integrator"] = 0.99
    dict_gains["pyr_40x40_8m_M0_n3.py"][0.16]["linear_combination"] = 0.8
    dict_gains["pyr_40x40_8m_M0_n3.py"][0.16]["non_linear_combination"] = 0.2

    dict_gains["pyr_40x40_8m_M9_n3.py"][0.08]["linear_integrator"] = 0.99
    dict_gains["pyr_40x40_8m_M9_n3.py"][0.08]["linear_combination"] = 0.2
    dict_gains["pyr_40x40_8m_M9_n3.py"][0.08]["non_linear_combination"] = 0.6

    dict_gains["pyr_40x40_8m_M9_n3.py"][0.12]["linear_integrator"] = 0.99
    dict_gains["pyr_40x40_8m_M9_n3.py"][0.12]["linear_combination"] = 0.3
    dict_gains["pyr_40x40_8m_M9_n3.py"][0.12]["non_linear_combination"] = 0.6

    dict_gains["pyr_40x40_8m_M9_n3.py"][0.16]["linear_integrator"] = 0.8
    dict_gains["pyr_40x40_8m_M9_n3.py"][0.16]["linear_combination"] = 0.4
    dict_gains["pyr_40x40_8m_M9_n3.py"][0.16]["non_linear_combination"] = 0.6
    return dict_gains


def set_r0_and_gain(r0, parameter_file, env_, gains_linear, gains_non_linear):
    """
    From a dict of previous obtained gains we set gains for linear and non_linear reconstructor
    Args:
        r0: Current value of r0
        parameter_file: Current parameter file
        env_: The environment object
        gains_linear: gain linear reconstruction
        gains_non_linear: gain non-linear reconstruction

    Returns: None
    """
    assert parameter_file in ["pyr_40x40_8m_M0_n3.py", "pyr_40x40_8m_M9_n3.py"]
    env_.supervisor.atmos.set_r0(r0)
    dict_gains = get_gains()
    gain_linear = dict_gains[parameter_file][r0]["linear_integrator"]
    env_.supervisor.set_gain(gain_linear)
    env_.gain_factor_linear = dict_gains[parameter_file][r0]["linear_combination"]
    env_.gain_factor_unet = dict_gains[parameter_file][r0]["non_linear_combination"]
    if gains_linear > -1:
        env_.gain_factor_linear = gains_linear
    if gains_non_linear > -1:
        env_.gain_factor_unet = gains_non_linear

    if parameter_file in ["pyr_40x40_8m.py", "pyr_40x40_8m_gs_9.py"]:
        print("-Gains; linear_comb {} non_linear_comb {} linear only {}".format(env_.gain_factor_linear,
                                                                                env_.gain_factor_unet, gain_linear))


def manage_atmospheric_conditions_3_layers(args, total_step, env, agent, controller_type):
    """
    Manage atmospheric conditions
    Args:
        args: arguments object
        total_step: current step
        env: environment object
        agent: RL model
        controller_type: which controller are we using

    Returns: None

    """
    if total_step == args.change_atmospheric_conditions_1_at:
        print("Changing atmos conditions 1")
        env.supervisor.atmos.set_wind(0, winddir=90)
        env.supervisor.atmos.set_wind(1, winddir=110)
        env.supervisor.atmos.set_wind(2, winddir=270)
        if args.reset_replay_buffer_when_change_atmos:
            agent.replay_buffer.reset()
        if args.reset_adam_when_change_atmos:
            agent.reset_optimizer()
    elif total_step == args.change_atmospheric_conditions_2_at:
        print("Changing atmos conditions 2")
        env.supervisor.atmos.set_r0(0, r0=0.08)
        if args.reset_replay_buffer_when_change_atmos and controller_type == "RL":
            agent.replay_buffer.reset()
        if args.reset_adam_when_change_atmos and controller_type == "RL":
            agent.reset_optimizer()
    elif total_step == args.change_atmospheric_conditions_3_at:  # error at 3
        print("Changing atmos conditions 3")
        env.supervisor.atmos.set_r0(0, r0=0.16)
        if args.reset_replay_buffer_when_change_atmos and controller_type == "RL":
            agent.replay_buffer.reset()
        if args.reset_adam_when_change_atmos and controller_type == "RL":
            agent.reset_optimizer()
    elif total_step == args.change_atmospheric_conditions_4_at:
        print("Changing atmos conditions 4")
        env.supervisor.atmos.set_wind(0, windspeed=30)
        env.supervisor.atmos.set_wind(1, windspeed=30)
        env.supervisor.atmos.set_wind(2, windspeed=40)
        if args.reset_replay_buffer_when_change_atmos and controller_type == "RL":
            agent.replay_buffer.reset()
        if args.reset_adam_when_change_atmos and controller_type == "RL":
            agent.reset_optimizer()


def save_configs_to_directory(dir_path_, config_env_rl_, config_agent_):
    # Create the directories if they don't exist
    if not os.path.exists(dir_path_):
        os.makedirs(dir_path_)

    # Save the configurations
    with open(os.path.join(dir_path_, "config_env_rl.json"), "w") as f:
        json.dump(config_env_rl_, f, indent=4)

    with open(os.path.join(dir_path_, "config_agent.json"), "w") as f:
        json.dump(config_agent_, f, indent=4)


def manage_metrics(df, r_total_train, r_pzt_total_train, r_tt_total_train, sr_se_total,
                   step, total_step, delta_time, seed, sr_le, s_dict,
                   max_tt_value,
                   min_tt_value,
                   max_pzt_value,
                   min_pzt_value,
                   count_pzt_surpass,
                   count_tt_surpass,
                   a_pzt_total,
                   a_tt_total
                   ):
    r_total_train /= step
    r_pzt_total_train /= step
    r_tt_total_train /= step
    a_pzt_total /= step
    a_tt_total /= step
    sr_se_total /= step

    print("Total steps:", total_step,
          "Seed:", seed,
          "R total:", round(r_total_train, 5),
          "Rec pzt:", round(r_pzt_total_train, 5),
          "Rec tt:", round(r_tt_total_train, 5),
          "Time:", round(delta_time, 5),
          "SR LE:", round(sr_le, 5),
          "SR SE:", round(sr_se_total, 5),
          "Command Pzt Min", min_pzt_value,
          "Command Pzt Max", max_pzt_value,
          "Count PZT surpassing clip", count_pzt_surpass,
          "Command TT Min", min_tt_value,
          "Command TT Max", max_tt_value,
          "Count TT surpassing clip", count_tt_surpass,
          "a pzt",  round(a_pzt_total, 5),
          "a tt",  round(a_tt_total, 5)
          )

    new_row = {"Total steps": total_step,
               "Seed": seed,
               "R total": round(r_total_train, 5),
               "Rec pzt": round(r_pzt_total_train, 5),
               "Rec tt": round(r_tt_total_train, 5),
               "Time:": round(delta_time, 5),
               "SR LE": round(sr_le, 5),
               "SR SE": round(sr_se_total, 5),
               "Command Pzt Min": min_pzt_value,
               "Command Pzt Max": max_pzt_value,
               "Count PZT surpassing clip": count_pzt_surpass,
               "Command TT Min": min_tt_value,
               "Command TT Max": max_tt_value,
               "Count TT surpassing clip": count_tt_surpass,
               "a pzt":  round(a_pzt_total, 5),
               "a tt":  round(a_tt_total, 5)}
    for key, item in s_dict.items():
        new_row[key + "_min"] = item.min()
        new_row[key + "_max"] = item.max()
    df = pd.concat([df, pd.DataFrame([new_row])]).reset_index(drop=True)

    return df


def manage_reward_and_metrics(r, r_total_train, r_pzt_total_train, r_tt_total_train, rec_for_reward, sr_se, sr_se_total,
                              command, clip_value, max_tt_value, min_tt_value, max_pzt_value, min_pzt_value,
                              count_tt_surpass, count_pzt_surpass,
                              a_pzt, a_tt, a_pzt_total, a_tt_total):
    if isinstance(r, np.ndarray):
        r_total_train += r.mean()
        rec_pzt = -np.square(rec_for_reward[:-2]).mean()
        rec_tt = -np.square(rec_for_reward[-2:]).mean()
        r_pzt_total_train += rec_pzt
        r_tt_total_train += rec_tt
        if a_pzt is not None:
            a_pzt_total += np.abs(a_pzt).mean()
        if a_tt is not None:
            a_tt_total += np.abs(a_tt).mean()
    else:
        r_total_train += r

    sr_se_total += sr_se

    tt_command_abs = np.abs(command[-2:])
    pzt_command_abs = np.abs(command[:-2])
    max_tt_value = max(max_tt_value, tt_command_abs.max())
    min_tt_value =  min(min_tt_value, tt_command_abs.min())
    max_pzt_value = max(max_pzt_value, pzt_command_abs.max())
    min_pzt_value =  min(min_pzt_value, pzt_command_abs.min())
    if (tt_command_abs > clip_value).any():
        count_tt_surpass += 1
    if (pzt_command_abs > clip_value).any():
        count_pzt_surpass += 1

    return r_total_train, r_pzt_total_train, r_tt_total_train, sr_se_total, max_tt_value, min_tt_value,\
           max_pzt_value, min_pzt_value, count_tt_surpass, count_pzt_surpass, a_pzt_total, a_tt_total
