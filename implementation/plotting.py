import matplotlib.pyplot as plt
import numpy as np
import json
import os
import pandas as pd
from helper import add_error_info_to_param_list
from brokenaxes import brokenaxes
plt.rcParams['text.usetex'] = True

plt.style.use('science')
figures_dir = r"C:\Users\Niels\Documents\thesis\latex\figures"

def plot_pq_pc_reduction_ratio(df_results, num_perm, x='reduction_ratio', y='pair_completeness', hue='amplified_x_sketch_type'):
    # create the figure and axis objects
    fig, ax = plt.subplots()
    ratio = 18.5 / 10.5
    x_width = 7
    fig.set_size_inches(x_width, x_width / ratio)
    ax.set_xlabel('Reduction ratio (RR)')
    if y == 'pair_completeness':
        ax.set_ylabel('Pair completeness (PC)')
        ax.set_title(f'Pair completeness vs Reduction ratio for several configurations of LSH (n = {num_perm})')
    elif y == 'pair_quality':
        ax.set_ylabel('Pair quality (PQ)')
        ax.set_title(f'Pair Quality vs Reduction ratio for several configurations of LSH (n = {num_perm})')
    elif y == 'f1':
        ax.set_ylabel('F1-measure')
        ax.set_title(f'F1-measure vs Reduction ratio for several configurations of LSH (n = {num_perm})')


    for hue_type in list(df_results[hue].unique()):
        df_to_plot = df_results.loc[df_results[hue] == hue_type]
        x_array = df_to_plot[x].to_numpy()
        y_array = df_to_plot[y].to_numpy()

        # plot the data and customize
        ax.plot(x_array, y_array, label=hue_type, linestyle='--', marker='o', markersize=3)
    ax.legend()
    # plt.show()


def plot_pq_pc_reduction_ratio_improv(df_results, hue_selectors, x='reduction_ratio', y='pair_completeness', title="Title", save_as=None, x_start=0.5, x_end=1):
    # create the figure and axis objects
    fig, ax = plt.subplots()
    # fig = plt.figure()
    # ax = brokenaxes(xlims=((0, 0), (0.5, 1)), hspace=0.05)
    ratio = 18.5 / 10.5
    x_width = 7
    fig.set_size_inches(x_width, x_width / ratio)
    ax.set_xlabel('Reduction ratio (RR)')
    ax.set_xlim(x_start, x_end)
    # ax.set_title(title)
    if y == 'pair_completeness':
        ax.set_ylabel('Pair completeness (PC)')
    elif y == 'pair_quality':
        ax.set_ylabel('Pair quality (PQ)')

    for hue_type in hue_selectors:
        df_selector =df_results[(df_results['num_perm'] == hue_type['num_perm']) &
                                (df_results['sketch_type'] == hue_type['sketch_type']) &
                                (df_results['amplified'] == hue_type['amplified'])]
        x_array = df_selector[x].to_numpy()
        y_array = df_selector[y].to_numpy()

        # plot the data and customize
        ax.plot(x_array,
                y_array,
                label=f'{hue_type["sketch_type"]} - {"amplified" if hue_type["amplified"] else "not amplified"} - (n={hue_type["num_perm"]})',
                linestyle='--',
                marker='o',
                markersize=3)
    ax.legend()
    # plt.show()
    if not save_as:
        plt.savefig(os.path.join(figures_dir, f"{title}.png"), dpi=100)
    else:
        plt.savefig(os.path.join(figures_dir, f"{save_as}.png"), dpi=100)


def plot_minhash_vs_fss_deviation(df):
    # create the figure and axis objects
    fig, ax = plt.subplots()
    ratio = 18.5 / 10.5
    x_width = 7
    fig.set_size_inches(x_width, x_width / ratio)
    ax.set_xlabel('Sketch length (n)')
    ax.set_ylabel('Average absolute deviation of Jaccard similarity estimate ($\hat{\delta}$)')
    # ax.set_title('$\hat{z}$ ' +  f'vs n (threshold = {threshold})')
    ax.set_ylim(bottom=0, top=0.10)
    ax.set_xlim(left=0, right=1050)
    for sketch in ['minhash', 'fss']:
        df_to_plot = df[df['sketch'] == sketch]
        x_array = df_to_plot['sketch_length'].to_numpy()
        y_array = df_to_plot['absolute_deviation'].to_numpy()
        # plot the data and customize
        ax.plot(x_array, y_array, label=f'{sketch}')
    ax.legend()
    # plt.show()
    plt.savefig(os.path.join(figures_dir,
                             f"minhash_vs_fss_preciseness.png"),
                            dpi=100)


def plot_errors_per_threshold(df, threshold, errors_to_plot=('fn', 'fp', 'weighted_average_error')):
    # create the figure and axis objects
    fig, ax = plt.subplots()
    ratio = 18.5 / 10.5
    x_width = 7
    fig.set_size_inches(x_width, x_width / ratio)
    ax.set_xlabel('Number of hash functions (n)')
    ax.set_ylabel('Optimal value of weighted average error ($\hat{z}$)')
    # ax.set_title('$\hat{z}$ ' +  f'vs n (threshold = {threshold})')
    ax.set_ylim(bottom=0, top=0.08)
    ax.set_xlim(left=0, right=1050)
    print(df)
    for amplified_ in [True, False]:
        for type_of_error in errors_to_plot:
            df_to_plot = df.loc[df["threshold"] == threshold]
            print(df_to_plot)
            if amplified_:
                df_to_plot = df_to_plot.loc[df_to_plot['amplified']]
            else:
                df_to_plot = df_to_plot.loc[~df_to_plot['amplified']]
            x_array = df_to_plot['num_perm'].to_numpy()
            y_array = df_to_plot[type_of_error].to_numpy()

            # plot the data and customize
            ax.plot(x_array, y_array, label=f'{"amplified" if amplified_ else "not amplified"}')

            # plot horizontal line at error of non amplified 1024
            print(y_array)
            if not amplified_:
                plt.hlines(y=y_array[len(y_array) - 1], xmin=0, xmax=1024, color='red')
    ax.legend()
    # plt.show()
    plt.savefig(os.path.join(figures_dir, f"z_vs_sketch_length_t_{str(threshold).replace('.','')}.png"), dpi=100)


def plot_s_curve_with_errors(df, threshold, num_perm):
    # create the figure and axis objects
    fig, ax = plt.subplots()
    ratio = 18.5 / 10.5
    x_width = 7
    fig.set_size_inches(x_width, x_width / ratio)
    ax.set_xlabel("Similarity s(p, p')")
    ax.set_ylabel("Pr[(p,p') is a candidate pair|s(p,p')]")
    # ax.set_title("Change in s-curve as a result of increase in n " + f"(threshold = {threshold})")
    plt.axvline(x=threshold, color='red', drawstyle='steps', linestyle='--')
    df_to_plot = df.loc[(df["threshold"] == threshold ) & (~df["amplified"]) & (df["num_perm"] == num_perm)]
    print(df_to_plot)
    b_1 = list(df_to_plot.loc[:, 'b1'])[0]
    b_2 = list(df_to_plot.loc[:, 'b2'])[0]
    r_1 = list(df_to_plot.loc[:, 'r1'])[0]
    r_2 = list(df_to_plot.loc[:, 'r2'])[0]
    _probability = lambda s: 1 - (1 - (1 - (1 - s ** float(r_1)) ** float(b_1)) ** float(r_2)) ** float(b_2)

    # create data
    step = 0.001
    x_array = np.arange(start=0, stop=1 + step, step=step)
    y_array = np.apply_along_axis(_probability, axis=0, arr=x_array)
    print(b_1)
    # print shittt
    ax.plot(x_array, y_array, label=r"$1 - (1-s^{" + str(r_1) + "})^{" + str(b_1) + "}$")
    x_smaller_than_threshold = list(x_array[x_array <= threshold])
    y_array_smaller_than_threshold = list(y_array[:len(x_smaller_than_threshold)])
    x_smaller_than_threshold.append(threshold)
    y_array_smaller_than_threshold.append(0)
    ax.fill(x_smaller_than_threshold, y_array_smaller_than_threshold, color=(239 / 255, 112 / 255, 198 / 255, 0.38),
                label=f'Pr(False Positive)')
    x_greater_than_threshold = list(x_array[x_array > threshold])
    y_array_greater_than_threshold = list(y_array[-len(x_greater_than_threshold):])
    x_greater_than_threshold.append(threshold)
    y_array_greater_than_threshold.append(1)
    ax.fill(x_greater_than_threshold, y_array_greater_than_threshold,
                color=(239 / 255, 147 / 255, 112 / 255, 0.38), label=f'Pr(False Negative)')

    # handles, labels = plt.gca().get_legend_handles_labels()
    # order = [3, 0, 4, 1, 5, 2]
    # plt.legend([handles[idx] for idx in order], [labels[idx] for idx in order])
    plt.legend()
    # plt.show()
    plt.savefig(os.path.join(figures_dir, f"error_s_curve_t{str(threshold).replace('.', '')}_n_{num_perm}.png"), dpi=100)

def plot_s_curves_comparison_num_perm(df, threshold, num_perm):
    # create the figure and axis objects
    fig, ax = plt.subplots()
    ratio = 18.5 / 10.5
    x_width = 7
    fig.set_size_inches(x_width, x_width / ratio)
    ax.set_xlabel("Similarity s(p, p')")
    ax.set_ylabel("Pr[(p,p') is a candidate pair|s(p,p')]")
    # ax.set_title("Change in s-curve as a result of increase in n " + f"(threshold = {threshold})")
    plt.axvline(x=threshold, color='red', drawstyle='steps', linestyle='--')
    for index, nu_p in enumerate(num_perm):
        df_to_plot = df.loc[(df["threshold"] == threshold ) & (~df["amplified"]) & (df["num_perm"] == nu_p)]
        print(nu_p)
        b_1 = list(df_to_plot.loc[:, 'b1'])[0]
        b_2 = list(df_to_plot.loc[:, 'b2'])[0]
        r_1 = list(df_to_plot.loc[:, 'r1'])[0]
        r_2 = list(df_to_plot.loc[:, 'r2'])[0]
        _probability = lambda s: 1 - (1 - (1 - (1 - s ** float(r_1)) ** float(b_1)) ** float(r_2)) ** float(b_2)

        # create data
        step = 0.001
        x_array = np.arange(start=0, stop=1 + step, step=step)
        y_array = np.apply_along_axis(_probability, axis=0, arr=x_array)
        print(b_1)
        # print shittt
        ax.plot(x_array, y_array, label=r"$1 - (1-s^{" + str(r_1) + "})^{" + str(b_1) + "}$")
        x_smaller_than_threshold = list(x_array[x_array <= threshold])
        y_array_smaller_than_threshold = list(y_array[:len(x_smaller_than_threshold)])
        x_smaller_than_threshold.append(threshold)
        y_array_smaller_than_threshold.append(0)

        if index == 1:
            ax.fill(x_smaller_than_threshold, y_array_smaller_than_threshold, color=(239 / 255, 112 / 255, 198 / 255, 0.38),
                    label=f'Pr(False Positive) - n={nu_p}')
        else:
            ax.fill(x_smaller_than_threshold, y_array_smaller_than_threshold,
                    color=(239 / 255, 112 / 255, 140 / 255, 0.3),
                    label=f'Pr(False Positive) - n={nu_p}')
        x_greater_than_threshold = list(x_array[x_array > threshold])
        y_array_greater_than_threshold = list(y_array[-len(x_greater_than_threshold):])
        x_greater_than_threshold.append(threshold)
        y_array_greater_than_threshold.append(1)
        if index == 1:
            ax.fill(x_greater_than_threshold, y_array_greater_than_threshold,
                    color=(239 / 255, 147 / 255, 112 / 255, 0.38), label=f'Pr(False Negative) - n={nu_p}')
        else:
            ax.fill(x_greater_than_threshold, y_array_greater_than_threshold,
                    color=(239 / 255, 147 / 255, 60 / 255, 0.30), label=f'Pr(False Negative) - n={nu_p}')

    handles, labels = plt.gca().get_legend_handles_labels()
    order = [3, 0, 4, 1, 5, 2]
    plt.legend([handles[idx] for idx in order], [labels[idx] for idx in order])

    # plt.show()
    plt.savefig(os.path.join(figures_dir, f"error_s_curve_t{str(threshold).replace('.', '')}_n_{num_perm[0]}_{num_perm[1]}.png"), dpi=100)

def plot_s_curves_comparison_amplif_vs_non(df, threshold, num_perm):
    # create the figure and axis objects
    fig, ax = plt.subplots()
    ratio = 18.5 / 10.5
    x_width = 7
    fig.set_size_inches(x_width, x_width / ratio)
    ax.set_xlabel("Similarity s(p, p')")
    ax.set_ylabel("Pr[(p,p') is a candidate pair|s(p,p')]")
    # ax.set_title("Change in $\hat{z}$ as a result of amplification " + f"(threshold = {threshold}, n={num_perm})")
    plt.axvline(x=threshold, color='red', drawstyle='steps', linestyle='--')
    for amplified_ in [True, False]:
        df_to_plot = df.loc[(df["threshold"] == threshold ) & (df["num_perm"] == num_perm)]
        if amplified_:
            df_to_plot = df_to_plot.loc[df_to_plot['amplified']]
        else:
            df_to_plot = df_to_plot.loc[~df_to_plot['amplified']]
        print(df_to_plot)
        b_1 = list(df_to_plot.loc[:, 'b1'])[0]
        b_2 = list(df_to_plot.loc[:, 'b2'])[0]
        r_1 = list(df_to_plot.loc[:, 'r1'])[0]
        r_2 = list(df_to_plot.loc[:, 'r2'])[0]
        _probability = lambda s: 1 - (1 - (1 - (1 - s ** float(r_1)) ** float(b_1)) ** float(r_2)) ** float(b_2)

        # create data
        step = 0.001
        x_array = np.arange(start=0, stop=1 + step, step=step)
        y_array = np.apply_along_axis(_probability, axis=0, arr=x_array)
        print(b_1)
        # print shittt
        if amplified_:
            ax.plot(x_array, y_array,
                    label=r"$1 - (1-s^{" + str(r_1) + "})^{" + str(b_1) + "})^{" + str(r_2) + "})^{" + str(b_2) + "}$")
        else:
            ax.plot(x_array, y_array, label=r"$1 - (1-s^{" + str(r_1) + "})^{" + str(b_1) + "}$")
        x_smaller_than_threshold = list(x_array[x_array <= threshold])
        y_array_smaller_than_threshold = list(y_array[:len(x_smaller_than_threshold)])
        x_smaller_than_threshold.append(threshold)
        y_array_smaller_than_threshold.append(0)

        if amplified_:
            ax.fill(x_smaller_than_threshold, y_array_smaller_than_threshold, color=(239 / 255, 112 / 255, 198 / 255, 0.38),
                    label='Pr(False Positive) - amplified')
        else:
            ax.fill(x_smaller_than_threshold, y_array_smaller_than_threshold,
                    color=(239 / 255, 112 / 255, 140 / 255, 0.3),
                    label='Pr(False Positive) - not amplified')
        x_greater_than_threshold = list(x_array[x_array > threshold])
        y_array_greater_than_threshold = list(y_array[-len(x_greater_than_threshold):])
        x_greater_than_threshold.append(threshold)
        y_array_greater_than_threshold.append(1)
        if amplified_:
            ax.fill(x_greater_than_threshold, y_array_greater_than_threshold,
                    color=(239 / 255, 147 / 255, 112 / 255, 0.38), label='Pr(False Negative) - amplified')
        else:
            ax.fill(x_greater_than_threshold, y_array_greater_than_threshold,
                    color=(239 / 255, 147 / 255, 60 / 255, 0.30), label='Pr(False Negative) - not amplified')

    handles, labels = plt.gca().get_legend_handles_labels()
    order = [3, 0, 4, 1, 5, 2]
    plt.legend([handles[idx] for idx in order], [labels[idx] for idx in order])

    # plt.show()
    plt.savefig(os.path.join(figures_dir, f"error_s_curve_t{str(threshold).replace('.', '')}_n_{num_perm}.png"), dpi=100)

def plot_multiple_s_curves(params_list, step=0.001, plot_threshold=False, threshold_value=0.5):
    # create the figure and axis objects
    fig, ax = plt.subplots()
    ratio = 18.5 / 10.5
    x = 7
    fig.set_size_inches(x, x / ratio)
    ax.set_xlabel("s(p, p') (Jaccard similarity)")
    ax.set_ylabel("Pr[(p, p') is candidate pair]")
    # ax.set_title(f's-curves for different combinations of r and b')
    # define function
    for params in params_list:
        b = params[0]
        r = params[1]
        _probability = lambda s: 1 - (1 - (1 - (1 - s ** float(r[0])) ** float(b[0])) ** float(r[1])) ** float(b[1])
        # create data
        x_array = np.arange(start=0, stop=1 + step, step=step)
        y_array = np.apply_along_axis(_probability, axis=0, arr=x_array)

        # plot the data and customize
        ax.plot(x_array, y_array, label=r"$1 - (1-s^{" + str(r[0]) +"})^{" + str(b[0]) + "}$")
    if plot_threshold:
        plt.axvline(x=threshold_value, color='red', drawstyle='steps', linestyle='--')
    ax.legend()
    # plt.show()
    plt.savefig(os.path.join(figures_dir, f"multiple_s_curves_{'with_threshold' if plot_threshold else 'without_threshold'}.png"), dpi=100)

def plot_probability_function(self, step=0.001):
    # define function
    _probability = lambda s: 1 - (
                1 - (1 - (1 - s ** float(self.r[0])) ** float(self.b[0])) ** float(self.r[1])) ** float(self.b[1])

    # create data
    x_array = np.arange(start=0, stop=1 + step, step=step)
    y_array = np.apply_along_axis(_probability, axis=0, arr=x_array)

    # create the figure and axis objects
    fig, ax = plt.subplots()

    # plot the data and customize
    ax.plot(x_array, y_array)
    ax.set_xlabel('s (Jaccard similarity)')
    ax.set_ylabel('Probability of being declared a candidate pair')
    # ax.set_title(f'threshold = {self.threshold}, amplified')

    # plt.show()


if __name__ == '__main__':

    with open('./results/parameter_config.json', 'r') as file_:
        list_configuration = json.load(file_)

    num_perm = 256
    amplified = False
    params_list_1 = [x['params'] for x in list_configuration if x['num_perm'] == num_perm and
                                                              x['amplified'] is amplified and
                                                              x['threshold'] in (0.4, 0.5, 0.6)]
    plot_multiple_s_curves(params_list=params_list_1,
                           plot_threshold=True,
                           threshold_value=0.5)


    params_list_2 = [x['params'] for x in list_configuration if x['num_perm'] == num_perm and
                     x['amplified'] is amplified and
                     x['threshold'] in (0.2, 0.3, 0.5, 0.7, 0.8)]

    plot_multiple_s_curves(params_list=params_list_2,
                           plot_threshold=False,
                           threshold_value=0.5)


    df_params = add_error_info_to_param_list(param_config_list=list_configuration)

    ##########################################################################################
    # Change in z as a result of non-amplified -> amplified for multiple parameter combis    #
    ##########################################################################################
    filter_ = [(256, 0.1), (1024, 0.5), (128, 0.8)]
    for (num_perm, threshold) in filter_:
        plot_s_curves_comparison_amplif_vs_non(df_params, threshold=threshold, num_perm=num_perm)


    for t in [0.2, 0.6]:
        print(t)
        plot_errors_per_threshold(df_params, threshold=t, errors_to_plot=['weighted_average_error'])

    plot_s_curve_with_errors(df_params, threshold=0.5, num_perm=256)
    plot_s_curves_comparison_num_perm(df_params, threshold=0.5, num_perm=[512, 128])



    ########################
    #   EVALUATION RESULTS
    ########################

    tuples_of_interest = [(25,64), (25, 128), (25, 256), (25, 512), (25, 1024)]
    list_df = []
    for (iterations, no_perm) in tuples_of_interest:
        try:
            df_bootstrap = pd.read_excel(f"./results/bootstrap_{iterations}_n_{no_perm}.xlsx", index_col=0)

            # Average per hyperparameter combi
            if 'f1' in df_bootstrap.columns:
                df_bootstrap = df_bootstrap.drop(labels='f1', axis=1)
            df_average = df_bootstrap.groupby(["threshold", "amplified", "sketch_type", "amplified_x_sketch_type"],
                                              as_index=False).mean()
            df_average['pair_completeness'] = df_average['pair_completeness'].apply(lambda x: x / 100)
            df_average['alpha'] = df_average['reduction_ratio'] * df_average['pair_completeness']
            df_average['num_perm'] = no_perm
            df_average['f1_adjusted'] = (2 * df_average['reduction_ratio'] * df_average['pair_completeness']) / (df_average['reduction_ratio'] + df_average['pair_completeness'])
            list_df.append(df_average)
        except:
            pass
    df_results = pd.concat(list_df)

    hue_selectors_fss_vs_minhash = [{'amplified': False, 'sketch_type': 'fss', 'num_perm': 64},
                                     {'amplified': False, 'sketch_type': 'minhash', 'num_perm': 64},
                                     {'amplified': False, 'sketch_type': 'fss', 'num_perm': 512},
                                     {'amplified': False, 'sketch_type': 'minhash', 'num_perm': 512}
                                     ]

    plot_pq_pc_reduction_ratio_improv(df_results=df_results,
                                      hue_selectors=hue_selectors_fss_vs_minhash,
                                      x='reduction_ratio',
                                      y='pair_completeness',
                                      title='Comparison of non-amplified MinHash with FSS for n=64,512')

    hue_selectors_fss_vs_minhash = [{'amplified': False, 'sketch_type': 'fss', 'num_perm': 128},
                                     {'amplified': False, 'sketch_type': 'minhash', 'num_perm': 128},
                                     {'amplified': False, 'sketch_type': 'fss', 'num_perm': 512},
                                     {'amplified': False, 'sketch_type': 'minhash', 'num_perm': 512}
                                     ]

    plot_pq_pc_reduction_ratio_improv(df_results=df_results,
                                      hue_selectors=hue_selectors_fss_vs_minhash,
                                      x='reduction_ratio',
                                      y='pair_completeness',
                                      title='Comparison of non-amplified MinHash with FSS for n=128,512')


    hue_selectors = [{'amplified': True, 'sketch_type': 'fss', 'num_perm': 256},
                     {'amplified': False, 'sketch_type': 'fss', 'num_perm': 256},
                     {'amplified': False, 'sketch_type': 'fss', 'num_perm': 512},
                     {'amplified': False, 'sketch_type': 'fss', 'num_perm': 1024},
                     ]

    plot_pq_pc_reduction_ratio_improv(df_results=df_results,
                                      hue_selectors=hue_selectors,
                                      x='reduction_ratio',
                                      y='pair_completeness',
                                      title="Comparing effect of increasing $n$ with amplification",
                                      save_as="increase_n_vs_amplified_effect")

    for no_perm in (64, 128, 256, 512, 1024):
        hue_selectors = [{'amplified': True, 'sketch_type': 'fss', 'num_perm': no_perm},
                         {'amplified': False, 'sketch_type': 'fss', 'num_perm': no_perm}
                         ]

        plot_pq_pc_reduction_ratio_improv(df_results=df_results,
                                          hue_selectors=hue_selectors,
                                          x='reduction_ratio',
                                          y='pair_completeness',
                                          title=f"PC - RR of amplified vs non-amplified MinHash scheme with $n$={no_perm}",
                                          save_as=f"fss_amplified_vs_non_amplified_{no_perm}")

    ########################
    # MINHASH VS FSS preciseness results
    ########################

    df_min_fss = pd.read_excel('./results/minhash_vs_fss_preciseness.xlsx')
    plot_minhash_vs_fss_deviation(df_min_fss)