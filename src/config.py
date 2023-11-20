# config.py


def obtain_config_env_default(parameter_file, n_reverse_filtered_from_cmat):
    config_env_rl = {}
    # Parameter file
    config_env_rl['parameter_file'] = parameter_file
    # State
    config_env_rl['normalization_bool'] = True
    config_env_rl['s_dm_residual_rl'] = False
    config_env_rl['s_dm'] = True
    config_env_rl['s_dm_residual'] = False
    config_env_rl['number_of_previous_s_dm'] = 0
    config_env_rl['number_of_previous_s_dm_residual'] = 0
    config_env_rl['number_of_previous_s_dm_residual_rl'] = 0
    config_env_rl['normalization_bool'] = True
    config_env_rl['dm_std'] = 1.0
    config_env_rl['dm_residual_std'] = 1.0
    # Reward
    config_env_rl['reward_type'] = "2d_actuators"
    config_env_rl['number_of_previous_a_for_reward'] = 0
    config_env_rl['value_action_penalizer'] = -1
    # Action
    config_env_rl['action_scale'] = 1.0
    # Other
    config_env_rl['filter_state_with_btt'] = True
    config_env_rl['n_reverse_filtered_from_cmat'] = n_reverse_filtered_from_cmat
    config_env_rl['filter_state_actuator_space_with_btt'] = True
    config_env_rl['use_mask_influence_function'] = False
    config_env_rl['filter_commands'] = False
    config_env_rl['command_clip_value'] = 1000
    config_env_rl['reset_when_clip'] = False
    config_env_rl['reduce_gain_tt_to'] = -1
    # Mode: correction or only_rl
    config_env_rl['mode'] = "only_rl"
    # Delayed assignment
    config_env_rl['delayed_assignment'] = 0
    # Reset strehl LE every
    config_env_rl['reset_strehl_every_and_print'] = 999999999999999
    # Control tip tilt?
    config_env_rl['control_tt'] = False
    config_env_rl['number_of_previous_s_dm_residual_tt'] = 0
    config_env_rl['s_dm_residual_tt'] = False
    config_env_rl['number_of_previous_s_dm_tt'] = 0
    config_env_rl['s_dm_tt'] = False
    config_env_rl['joint_tt_into_s_dm'] = False
    config_env_rl['joint_tt_into_s_dm_residual'] = False
    config_env_rl['joint_tt_into_reward'] = False
    config_env_rl['separate_tt_into_two_actions'] = False
    # Unet extra
    config_env_rl['s_dm_residual_non_linear'] = False
    config_env_rl['s_dm_residual_non_linear_tt'] = False
    config_env_rl['number_of_previous_s_dm_residual_non_linear'] = 0
    config_env_rl['number_of_previous_s_dm_residual_non_linear_tt'] = 0
    config_env_rl['joint_tt_into_s_dm_residual_non_linear'] = False
    config_env_rl['no_subtract_mean_from_phase'] = False
    config_env_rl['normalization_noise_unet'] = True
    config_env_rl['normalization_noise_value_unet'] = 3

    return config_env_rl


def obtain_config_env(parameter_file,
                      number_of_modes_filtered=100,
                      mode="correction",
                      s_dm_residual=True,
                      s_dm_residual_rl=False,
                      number_of_previous_s_dm_residual_rl=0,
                      number_of_previous_s_dm=3,
                      s_dm=True,
                      control_tt=False,
                      s_dm_tt=False,
                      s_dm_residual_tt=False,
                      number_of_previous_s_dm_residual_tt=0,
                      number_of_previous_s_dm_tt=3,
                      s_dm_residual_non_linear=True,
                      number_of_previous_s_dm_residual_non_linear=0,
                      s_dm_residual_non_linear_tt=True,
                      number_of_previous_s_dm_residual_non_linear_tt=0,
                      filter_commands=True,
                      command_clip_value=500,
                      joint_tt_into_s_dm=False,
                      joint_tt_into_s_dm_residual=False,
                      scaling_for_residual_tt=1.0,
                      action_scale=10.0,
                      number_of_previous_a_for_reward=2,
                      value_action_penalizer=-1,
                      delayed_assignment=1,
                      reset_when_clip=True,
                      reduce_gain_tt_to=-1,
                      no_subtract_mean_from_phase=False
                      ):

    config_env_rl = {}
    # Parameter file
    config_env_rl['parameter_file'] = parameter_file
    # State
    config_env_rl['normalization_bool'] = True
    config_env_rl['s_dm_residual_rl'] = s_dm_residual_rl
    config_env_rl['s_dm'] = s_dm
    config_env_rl['s_dm_residual'] = s_dm_residual
    config_env_rl['number_of_previous_s_dm'] = number_of_previous_s_dm
    config_env_rl['number_of_previous_s_dm_residual'] = 0
    config_env_rl['number_of_previous_s_dm_residual_rl'] = number_of_previous_s_dm_residual_rl
    config_env_rl['normalization_bool'] = True
    config_env_rl['dm_std'] = 10.0
    config_env_rl['dm_residual_std'] = 10.0
    config_env_rl['correction_hddm_only_rl_tt'] = False
    # Reward
    config_env_rl['reward_type'] = "2d_actuators"  # 2d_actuators or scalar_actuators
    config_env_rl['number_of_previous_a_for_reward'] = number_of_previous_a_for_reward
    config_env_rl['value_action_penalizer'] = value_action_penalizer
    # Action
    config_env_rl['action_scale'] = action_scale
    # Other
    config_env_rl['filter_state_with_btt'] = True
    config_env_rl['n_reverse_filtered_from_cmat'] = number_of_modes_filtered
    config_env_rl['use_mask_influence_function'] = False
    config_env_rl['filter_state_actuator_space_with_btt'] = True
    config_env_rl['filter_commands'] = filter_commands
    config_env_rl['command_clip_value'] = command_clip_value
    config_env_rl['reset_when_clip'] = reset_when_clip
    config_env_rl['reduce_gain_tt_to'] = reduce_gain_tt_to
    # Mode: correction or only_rl
    config_env_rl['mode'] = mode
    # Delayed assignment
    config_env_rl['delayed_assignment'] = delayed_assignment
    # Reset strehl LE every
    config_env_rl['reset_strehl_every_and_print'] = 1000
    # Control tip tilt?
    config_env_rl['control_tt'] = control_tt
    config_env_rl['number_of_previous_s_dm_residual_tt'] = number_of_previous_s_dm_residual_tt
    config_env_rl['s_dm_residual_tt'] = s_dm_residual_tt
    config_env_rl['number_of_previous_s_dm_tt'] = number_of_previous_s_dm_tt
    config_env_rl['s_dm_tt'] = s_dm_tt
    config_env_rl['joint_tt_into_s_dm'] = joint_tt_into_s_dm
    config_env_rl['joint_tt_into_s_dm_residual'] = joint_tt_into_s_dm_residual
    config_env_rl['joint_tt_into_reward'] = True
    config_env_rl['separate_tt_into_two_actions'] = False
    config_env_rl['scaling_for_residual_tt'] = scaling_for_residual_tt
    # Unet extra
    config_env_rl['s_dm_residual_non_linear'] = s_dm_residual_non_linear
    config_env_rl['s_dm_residual_non_linear_tt'] = s_dm_residual_non_linear_tt
    config_env_rl['number_of_previous_s_dm_residual_non_linear'] = number_of_previous_s_dm_residual_non_linear
    config_env_rl['number_of_previous_s_dm_residual_non_linear_tt'] = number_of_previous_s_dm_residual_non_linear_tt
    config_env_rl['joint_tt_into_s_dm_residual_non_linear'] = joint_tt_into_s_dm_residual # same
    config_env_rl['no_subtract_mean_from_phase'] = no_subtract_mean_from_phase
    config_env_rl['normalization_noise_unet'] = True
    config_env_rl['normalization_noise_value_unet'] = 3

    return config_env_rl


def obtain_config_agent(agent_type,
                        replay_buffer_size=50000,
                        entropy_factor=1.0,
                        automatic_entropy_tuning=True,
                        alpha=0.2,
                        gamma=0.1):
    config_agent = {}
    config_agent['replay_capacity'] = replay_buffer_size
    config_agent['batch_size'] = 256
    config_agent['lr'] = 0.0003
    config_agent['target_update_interval'] = 1
    config_agent['tau'] = 0.005
    config_agent['lr_alpha'] = 0.0003
    config_agent['gamma'] = gamma
    config_agent['update_simplified'] = False
    config_agent['train_for_steps'] = 1
    config_agent['train_every_steps'] = 1
    config_agent['print_every'] = 1001
    config_agent['num_layers_critic'] = 2
    config_agent['num_layers_actor'] = 3
    # Weight initialization policy
    config_agent['initialize_last_layer_zero'] = True
    config_agent['initialise_last_layer_near_zero'] = False
    config_agent['agent_type'] = agent_type
    if agent_type == "sac":
        config_agent['alpha'] = alpha
        config_agent['entroy_factor'] = entropy_factor
        config_agent['automatic_entropy_tuning'] = automatic_entropy_tuning
    else:
        raise NotImplementedError

    return config_agent


