# main.py
from src.env_methods.env import AoEnv
from src.rl_agent.utils import DelayedMDP
import numpy as np
import time
import argparse
import pandas as pd
import os
from src.env_methods.env_non_linear import AoEnvNonLinear
from src.mains.mains_rl.helper import \
    manage_atmospheric_conditions_3_layers, save_configs_to_directory, manage_metrics, manage_reward_and_metrics, \
    choose_exploration_exploitation, initialise


def create_args():
    parser = argparse.ArgumentParser(description="Your program description here.")
    # Basics
    parser.add_argument("--experiment_name", type=str)
    parser.add_argument("--number_of_modes_filtered", default=100, type=int)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--controller_type", type=str, default="RL", choices=["RL", "Linear", "UNet+Linear"])
    parser.add_argument("--seed", type=int, default=1234)
    # RL basics
    parser.add_argument("--mode", type=str, default="only_rl", choices=["only_rl", "correction"])
    parser.add_argument("--evaluation_after", type=int, default=30000)
    parser.add_argument("--max_step", type=int, default=60000)
    parser.add_argument("--no_s_dm", action="store_true")
    parser.add_argument("--s_dm_residual", action="store_true")
    parser.add_argument("--s_dm_residual_rl", action="store_true")
    parser.add_argument("--number_of_previous_s_dm", type=int, default=3)
    parser.add_argument("--number_of_previous_s_dm_residual_rl", type=int, default=0)
    parser.add_argument("--integrator_exploration_with_only_rl_for", default=-1, type=int)
    parser.add_argument("--noise_for_exploration", type=float, default=-1)
    parser.add_argument("--action_scale", type=float, default=10.0)
    parser.add_argument("--pure_deterministic", action="store_true")
    parser.add_argument("--value_action_penalizer", type=float, default=-1)
    parser.add_argument("--gamma", type=float, default=0.1)
    parser.add_argument("--delayed_assignment", type=int, default=1)
    parser.add_argument("--evaluation_after_steps", type=int, default=30000)
    # Parameter file parameters
    parser.add_argument("--parameter_file", default="pyr_40x40_8m_M0_n3.py", type=str)
    parser.add_argument("--r0", default=0.16, type=float)
    parser.add_argument("--change_atmospheric_conditions_1_at", type=int, default=-1)
    parser.add_argument("--change_atmospheric_conditions_2_at", type=int, default=-1)
    parser.add_argument("--change_atmospheric_conditions_3_at", type=int, default=-1)
    parser.add_argument("--change_atmospheric_conditions_4_at", type=int, default=-1)
    parser.add_argument("--stop_training_after", type=int, default=99999999)
    # UNet
    parser.add_argument("--unet_name", default=None)
    parser.add_argument("--unet_dir", default="data/models/unet/")
    parser.add_argument("--s_dm_residual_non_linear", action="store_true")
    parser.add_argument("--number_of_previous_s_dm_residual_non_linear", default=0)
    parser.add_argument("--s_dm_residual_non_linear_tt", action="store_true")
    parser.add_argument("--number_of_previous_s_dm_residual_non_linear_tt", default=0)
    parser.add_argument("--gain_linear", type=float, default=-1)
    parser.add_argument("--gain_non_linear", type=float, default=-1)
    # Control TT
    parser.add_argument("--control_tt", action="store_true")
    parser.add_argument("--s_dm_tt", action="store_true")
    parser.add_argument("--s_dm_residual_tt", action="store_true")
    parser.add_argument("--number_of_previous_s_dm_residual_tt", default=0, type=int)
    parser.add_argument("--number_of_previous_s_dm_tt", default=0, type=int)
    parser.add_argument("--reduce_gain_tt_to", default=-1, type=float)

    args_ = parser.parse_args()
    return args_


if __name__ == "__main__":

    # 0.1 Processing arguments
    args = create_args()
    # 0.2 Initialization
    env, agent, config_env_rl, config_agent = initialise(args)
    controller_type, device, seed = args.controller_type, args.device, args.seed
    dir_path = os.path.join("data", "results", "rl", args.experiment_name)
    save_configs_to_directory(dir_path, config_env_rl, config_agent)
    # 0.3 Starting values
    total_step = 0
    r_total_train, r_pzt_total_train, r_tt_total_train, sr_se_total, \
        step_counter_for_metrics, a_pzt_total, a_tt_total = [0] * 7
    max_tt_value, min_tt_value, max_pzt_value, min_pzt_value, count_pzt_surpass, count_tt_surpass = \
        -10000, 10000, -10000, 10000, 0, 0
    columns = ["Total steps", "Seed",
               "R total", "Rec pzt", "Rec tt", "Time", "SR LE", "SR SE",
               "Command Pzt Min", "Command Pzt Max", "Count PZT surpassing clip",
               "Command TT Min", "Command TT Max", "Count TT surpassing clip",
               "a tt", "a pzt"]
    exploitation = False
    df = pd.DataFrame(columns=columns)

    # 0.4 Initialization
    s = env.reset(only_reset_dm=False)
    delayed_mdp_object = DelayedMDP(config_env_rl['delayed_assignment'])
    while True:
        # Check if we change atmospheric conditions
        manage_atmospheric_conditions_3_layers(args, total_step, env, agent, controller_type)

        start_time = time.time()
        # 1. Choose action
        if args.integrator_exploration_with_only_rl_for > total_step:
            if isinstance(env, AoEnvNonLinear):
                a = env.calculate_non_linear_residual()
            elif isinstance(env, AoEnv):
                a = env.calculate_linear_residual()
            else:
                raise NotImplementedError
            a /= config_env_rl['action_scale']  # divide by action scale
            noise = np.random.normal(loc=0, scale=args.noise_for_exploration, size=a.shape)
            a = a + noise
            a = np.clip(a, a_min=-1, a_max=1)  # clip
            a = env.filter_actions(a, exploratory=True)  # filter
        elif controller_type == "RL":
            exploitation = choose_exploration_exploitation(args.evaluation_after_steps, total_step, exploitation)
            a = agent.select_action(s, evaluation=exploitation)
            if device < 0:
                a = a.cpu().numpy()
            a = env.filter_actions(a)
        else:
            a = None

        # 2. Env step
        s_next, r, done, info = env.step(a, controller_type=controller_type)

        # Manage metrics
        r_total_train, r_pzt_total_train, r_tt_total_train, sr_se_total, max_tt_value, min_tt_value,\
            max_pzt_value, min_pzt_value, count_tt_surpass, count_pzt_surpass, a_pzt_total, a_tt_total =\
            manage_reward_and_metrics(r, r_total_train, r_pzt_total_train, r_tt_total_train,
                                      env.reconstruction_for_reward,
                                      info["sr_se"],
                                      sr_se_total,
                                      command=env.supervisor.rtc.get_command(0),
                                      clip_value=config_env_rl['command_clip_value'],
                                      max_tt_value=max_tt_value,
                                      min_tt_value=min_tt_value,
                                      max_pzt_value=max_pzt_value,
                                      min_pzt_value=min_pzt_value,
                                      count_tt_surpass=count_tt_surpass,
                                      count_pzt_surpass=count_pzt_surpass,
                                      a_pzt=env.a_hddm,
                                      a_tt=env.a_hddm_from_tt,
                                      a_pzt_total=a_pzt_total,
                                      a_tt_total=a_tt_total)

        # 3. if the delayed_mdp is ready
        # Save on replay (s, a, r, s_next) which comes from delayed_mdp and the s_next and reward this timestep
        # Depending on some configs it may vary a little
        if controller_type == "RL" and total_step < args.stop_training_after:
            if delayed_mdp_object.check_update_possibility():
                state, action, state_next = delayed_mdp_object.credit_assignment()

                # c) Push to memory for different configs
                agent.update_replay_buffer(state, action, r, state_next)

                # d) Update
                if total_step % agent.train_every_steps == 0:
                    agent.train(total_step)

            # 4. Save s, a, s_next, r, a_next to do the correct credit assignment in replay memory later
            # We use this object because we have delay
            delayed_mdp_object.save(s, a, s_next)

        total_step += 1
        step_counter_for_metrics += 1
        # 5. s = s_next
        s = s_next.copy()

        # Manage metrics
        if total_step % 1000 == 0:
            df.to_csv("data/results.csv", index=False)

        if total_step % env.config_env_rl['reset_strehl_every_and_print'] == 0:

            df = manage_metrics(df,
                                r_total_train, r_pzt_total_train, r_tt_total_train, sr_se_total,
                                step_counter_for_metrics, total_step,
                                time.time() - start_time,
                                seed=env.supervisor.current_seed,
                                sr_le=info["sr_le"],
                                s_dict=env.get_next_state(return_dict=True),
                                max_tt_value=max_tt_value,
                                min_tt_value=min_tt_value,
                                max_pzt_value=max_pzt_value,
                                min_pzt_value=min_pzt_value,
                                count_pzt_surpass=count_pzt_surpass,
                                count_tt_surpass=count_tt_surpass,
                                a_pzt_total=a_pzt_total,
                                a_tt_total=a_tt_total)
            r_total_train, r_pzt_total_train, r_tt_total_train, sr_se_total, step_counter_for_metrics =\
                0, 0, 0, 0, 0
            max_tt_value, min_tt_value, max_pzt_value, min_pzt_value, count_pzt_surpass, count_tt_surpass =\
                -10000, 10000, -10000, 10000, 0, 0
            start_time = time.time()

        if total_step >= args.max_step:
            break
