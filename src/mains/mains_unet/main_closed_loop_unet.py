# main_closed_loop_unet.py
from src.env_methods.env_non_linear import AoEnvNonLinear
import numpy as np
import torch
import random
from src.config import obtain_config_env_default
import pandas as pd
import matplotlib.pyplot as plt
import argparse

plt.style.use("ggplot")

class closedLoopUnetTester:
    def __init__(self,
                 parameter_file,
                 unet_dir,
                 unet_name="40_net_Final_dataset_g9_normalization_noise_L1_relative.pth",
                 seed=1234,
                 device=0,
                 nfilt=100,
                 no_subtract_mean_from_phase=False):

        self.device = device
        # Configs
        config_env_rl = obtain_config_env_default(parameter_file, n_reverse_filtered_from_cmat=nfilt)  # for environment
        config_env_rl['reset_strehl_every_and_print'] = 999999999999999
        config_env_rl['no_subtract_mean_from_phase'] = no_subtract_mean_from_phase

        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        unet_type = "volts"

        # Environment
        self.env = AoEnvNonLinear(unet_dir,
                                  unet_name,
                                  unet_type,
                                  "cuda:" + str(device),
                                  gain_factor_unet=0,  # we set it up later
                                  normalization_095_005=True,
                                  config_env_rl=config_env_rl,
                                  parameter_file=parameter_file,
                                  seed=seed,
                                  device=device,
                                  normalization_noise_unet=True,
                                  normalization_noise_value_unet=3)

    def unet_plus_linear_scan_gains(self, len_loop = 100):
        """
        Scanning the gains
        """
        results = []
        for gain_factor_unet in [0.99, 0.95, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0]:
            for gain_factor_linear in [0.99, 0.95, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0]:
                print("Unet+linear, gain non-linear: ", gain_factor_unet, ", gain linear: ", gain_factor_linear)
                sr_se_tot, sr_le, _ = unet_plus_linear_loop(gain_factor_unet, gain_factor_linear, len_loop)
                # Storing the results in the list
                result = {
                    "gain_factor_unet": gain_factor_unet,
                    "gain_factor_linear": gain_factor_linear,
                    "sr_se_tot": sr_se_tot,
                    "sr_le": sr_le
                }
                results.append(result)

        df = pd.DataFrame(results)

        # Sort by sr_se_tot in descending order and pick top 3
        top_combinations = df.nlargest(3, 'sr_le')
        print("Best 3: ---")
        print(top_combinations[['gain_factor_unet', 'gain_factor_linear', 'sr_le']])

        return df

    def unet_plus_linear_loop(self,
                              gain_factor_unet,
                              gain_factor_linear,
                              len_loop=1000,
                              return_list_of_LE=False,
                              seed=None):
        """
        A closed-loop experiment
        """
        if seed is not None:
            self.env.supervisor.current_seed = seed
        self.env.gain_factor_unet = gain_factor_unet
        self.env.gain_factor_linear = gain_factor_linear
        self.env.reset_without_rl(False)
        list_of_LE = []
        sr_se_tot = 0
        for idx in range(len_loop):
            self.env.step_only_combined_with_linear()
            sr_se_tot += self.env.supervisor.target.get_strehl(0)[0]
            if return_list_of_LE:
                list_of_LE.append(self.env.supervisor.target.get_strehl(0)[1])
        sr_se_tot /= float(len_loop)
        sr_le = self.env.supervisor.target.get_strehl(0)[1]
        print("SR SE:", sr_se_tot, "SR LE: ", sr_le)
        return sr_se_tot, sr_le, list_of_LE

    def give_me_a_gain_and_will_produce_results(self,
                                                gain_linear,
                                                gain_non_linear_combination,
                                                gain_linear_combination,
                                                gain_non_linear, len_loop=1000):
        """
        Experiments over a given configuration with linear/non-linear/combination reconstructions
        """
        _, _, sr_le_list_linear = self.unet_plus_linear_loop(0, gain_linear, len_loop, return_list_of_LE=True)
        _, _, sr_le_list_non_linear = self.unet_plus_linear_loop(gain_non_linear, 0, len_loop, return_list_of_LE=True)
        _, _, sr_le_list_combination = self.unet_plus_linear_loop(gain_non_linear_combination, gain_linear_combination,
                                                             len_loop, return_list_of_LE=True)
        return sr_le_list_linear, sr_le_list_non_linear, sr_le_list_combination


    def iterate_over_r0(self, dict_of_gains):
        """
        Experiments over different r0 with different gains
        """
        dict_of_results = {'0.08': {}, '0.12': {}, '0.16': {}}
        for r0 in [0.08, 0.12, 0.16]:
            print("r0: ", r0)
            self.env.supervisor.atmos.set_r0(r0)
            sr_le_list_linear, sr_le_list_non_linear, sr_le_list_combination = self.give_me_a_gain_and_will_produce_results(
                dict_of_gains[str(r0)]['gain_linear'],
                dict_of_gains[str(r0)]['gain_non_linear_combination'],
                dict_of_gains[str(r0)]['gain_linear_combination'],
                dict_of_gains[str(r0)]['gain_non_linear'])
            dict_of_results[str(r0)]['Lin+U-Net'] = sr_le_list_combination
            dict_of_results[str(r0)]['Lin'] = sr_le_list_linear
            dict_of_results[str(r0)]['U-Net'] = sr_le_list_non_linear

        return dict_of_results

    def iterate_over_r0_different_seeds(self, dict_of_gains):
        """
        Experiments over different seeds
        """
        list_of_dict_of_results = []
        for seed in [1500, 1501, 1502, 1503, 1504]:
            self.env.supervisor.current_seed = seed
            dict_of_results = self.iterate_over_r0(dict_of_gains)
            list_of_dict_of_results.append(dict_of_results)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--parameter_file', default="pyr_40x40_8m_M9_n3.py")
    parser.add_argument("--device", default=0, type=int)
    parser.add_argument("--nfilt", default=100, type=int)
    parser.add_argument("--unet_dir", default="data/models/unet/")
    parser.add_argument("--unet_name", default="40_net_Final_dataset_g9_normalization_noise_L1_relative.pth")
    parser.add_argument("--seed", default=1234, type=int)
    parser.add_argument("--no_subtract_mean_from_phase", action="store_true")
    args = parser.parse_args()
    """
    This list of gains is based on a precomputed optimisation
    """
    dict_of_gains_ = {'0.08': {}, '0.12': {}, '0.16': {}}
    dict_of_gains_['0.08']['gain_linear'] = 0.99
    dict_of_gains_['0.08']['gain_non_linear_combination'] = 0.5
    dict_of_gains_['0.08']['gain_linear_combination'] = 0.5
    dict_of_gains_['0.08']['gain_non_linear'] = 0.7

    dict_of_gains_['0.12']['gain_linear'] = 0.9
    dict_of_gains_['0.12']['gain_non_linear_combination'] = 0.5
    dict_of_gains_['0.12']['gain_linear_combination'] = 0.4
    dict_of_gains_['0.12']['gain_non_linear'] = 0.7

    dict_of_gains_['0.16']['gain_linear'] = 0.7
    dict_of_gains_['0.16']['gain_non_linear_combination'] = 0.5
    dict_of_gains_['0.16']['gain_linear_combination'] = 0.3
    dict_of_gains_['0.16']['gain_non_linear'] = 0.7


    closed_loop_unet_tester = closedLoopUnetTester(args.parameter_file,
                                                   args.unet_dir,
                                                   args.unet_name,
                                                   args.seed,
                                                   args.device,
                                                   args.nfilt,
                                                   args.no_subtract_mean_from_phase)
    df = closed_loop_unet_tester.iterate_over_r0(dict_of_gains_)

    # Example we plot r=0.12 results
    for key in df["0.12"].keys():
        plt.plot(df["0.12"][key], label=key)
    plt.legend()
    plt.ylabel("LE SR")
    plt.xlabel("Frame number")
    plt.savefig("test_closed_loop.png")