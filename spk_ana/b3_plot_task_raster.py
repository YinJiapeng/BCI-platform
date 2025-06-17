import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import subprocess
import sys
import multiprocessing
from multiprocessing import Pool
from scipy.ndimage import gaussian_filter1d
from scipy.stats import sem


def group_tasks_and_run(pne_processing_list, t_before=-2000, t_after=2000):
    df_list0 = pd.read_csv(pne_processing_list)

    # Group by task_id field
    df_group = df_list0.groupby('task_id')

    # Iterate through each group
    for name, df_list in df_group:
        print(f"Group: {name}")

        df_list = df_list.reset_index(drop=True)

        # Get unique values of task_type field
        unique_task_types = df_list['task_type'].unique()

        if len(unique_task_types) == 1 and unique_task_types[0] == 'driving':
            print("The unique value of task_type is 'driving'.")
            print(df_list)

            unique_pne_para = df_list['pne_behave_params'].unique()
            if len(unique_task_types) == 1:
                pne_behave_params = unique_pne_para[0]
                df_behave_params = pd.read_excel(pne_behave_params)

                extract_and_plot(df_list, df_behave_params, t_before, t_after)

        elif len(unique_task_types) == 1 and unique_task_types[0] == 'movie':
            print("The unique value of task_type is 'movie', skip...")


def extract_and_plot(df_list, df_behave_params, t_before, t_after):
    check_openpyxl()

    total_cores = multiprocessing.cpu_count()

    # Calculate number of cores to use (50%)
    n_threads = int(total_cores * 0.5)

    # Sort df_list
    df_list = df_list.sort_values(by='brain_area_seq')

    # Create new index column as unit_seq
    df_list['unit_seq'] = range(1, len(df_list) + 1)

    # Convert time and calculate actual milliseconds
    event_cols = ['t_reach_start', 't_touch_fruit', 't_touch_mouth', 't_leave']
    for col in event_cols:
        df_behave_params[col] = df_behave_params[col].apply(time_to_ms)

    # Handle NaN values in delta_t_vid_t_global_ms (fill with 0 here, adjust as needed)
    df_behave_params['delta_t_vid_t_global_ms'] = df_behave_params['delta_t_vid_t_global_ms'].fillna(0).astype(int)
    for col in event_cols:
        df_behave_params[col] = df_behave_params[col] - df_behave_params['delta_t_vid_t_global_ms']

    # Filter rows
    df_behave_params = df_behave_params.dropna(subset=event_cols)

    # Use multiprocessing for parallel processing
    with Pool(n_threads) as pool:
        results = [pool.apply_async(process_unit, args=(unit_seq, df_behave_params, df_list, t_before, t_after)) for
                   unit_seq
                   in df_list['unit_seq']]
        [result.get() for result in results]


def process_unit(unit_seq, df_behave_params, df_list, t_before, t_after):
    events = {
        't_reach_start_seg': 't_reach_start',
        't_touch_fruit_seg': 't_touch_fruit',
        't_touch_mouth_seg': 't_touch_mouth',
        # 't_leave_this_seg': 't_leave_this'
    }

    colors = {
        't_reach_start_seg': 'black',
        't_touch_fruit_seg': 'red',
        't_touch_mouth_seg': 'blue',
        # 't_leave_this_seg': 'darkorange'
    }

    row = df_list[df_list['unit_seq'] == unit_seq].iloc[0]
    pne_spk = row['pne_spk']
    pn_result = row['b3_pn_result']
    pne_result_png = row['b3_pne_result_png']
    pne_result_pdf = row['b3_pne_result_pdf']
    fig_annotation = row['b3_fig_annotation']

    spike_data = pd.read_csv(pne_spk, usecols=['globalTime'])
    spike_data['globalTime'] = spike_data['globalTime'].astype(float)

    bin_size = 50  # Bin size in ms
    bins = np.arange(t_before, t_after + 1, bin_size)
    df_psth = pd.DataFrame()

    all_psths = {event_name: [] for event_name in events.keys()}
    raster_data = {event_name: [] for event_name in events.keys()}

    for event_name, event_time_col in events.items():
        for _, row in df_behave_params.iterrows():
            event_time = row[event_time_col]
            times = spike_data['globalTime'] - event_time
            raster_data[event_name].append(times[(times >= t_before) & (times <= t_after)])
            counts, _ = np.histogram(times, bins=bins)
            rates = counts / (bin_size / 1000)
            all_psths[event_name].append(rates)

        all_psths[event_name] = np.array(all_psths[event_name])
        mean_rates = np.mean(all_psths[event_name], axis=0)
        sem_rates = sem(all_psths[event_name], axis=0)
        df_psth[event_name] = mean_rates
        df_psth[f'{event_name}_sem'] = sem_rates

    os.makedirs(pn_result, exist_ok=True)

    plot_channel(df_psth, unit_seq, bins, colors, raster_data, fig_annotation, pne_result_png, pne_result_pdf, t_before,
                 t_after)


def plot_channel(df_psth, unit_seq, bins, colors, raster_data, fig_annotation, pne_result_png, pne_result_pdf, t_before,
                 t_after):
    # Set global font to Arial
    plt.rcParams['font.family'] = 'Arial'
    plt.rcParams.update({
        'font.size': 5,  # global font size
        'axes.titlesize': 5,  # subplot title font size
        'axes.labelsize': 5,  # axis label font size
        'xtick.labelsize': 5,  # x-axis tick font size
        'ytick.labelsize': 5,  # y-axis tick font size
        'legend.fontsize': 5,  # legend font size
        'axes.linewidth': 0.5,  # axis line width
    })

    events = [col for col in df_psth.columns if not col.endswith('_sem')]
    time = bins[:-1] + (bins[1] - bins[0]) / 2

    num_events = len(events)

    # Set figure size (width 18 cm, height adjusted proportionally)
    fig_width_cm = 8.5  # width in cm
    fig_height_cm = 4  # height in cm
    fig_width_inch = fig_width_cm / 2.54  # convert to inches
    fig_height_inch = fig_height_cm / 2.54  # convert to inches

    fig, axes = plt.subplots(2, num_events, figsize=(fig_width_inch, fig_height_inch))
    # Iterate through all subplots
    for i in range(2):  # iterate rows
        for j in range(num_events):  # iterate columns
            ax = axes[i, j]  # get current subplot

            # Set x and y axis line width to 0.5
            ax.spines['bottom'].set_linewidth(0.5)  # bottom edge
            ax.spines['left'].set_linewidth(0.5)  # left edge

            # Remove top and right edges
            # ax.spines['left'].set_visible(False)   # hide left edge
            ax.spines['top'].set_visible(False)  # hide top edge
            ax.spines['right'].set_visible(False)  # hide right edge

            # Optional: set tick line width
            ax.tick_params(axis='both', width=0.5)

    max_rate = 0  # initialize maximum value as negative infinity

    for i, event in enumerate(events):
        # Plot response curve
        rates = gaussian_filter1d(df_psth[event].values, sigma=1)

        # Update maximum value
        current_max = np.max(rates)
        if current_max > max_rate:
            max_rate = current_max

    if max_rate >= 0.1:
        for i, event in enumerate(events):
            # Plot raster plot
            ax_raster = axes[0, i]
            total_trials = len(raster_data[event])
            for trial, trial_times in enumerate(raster_data[event]):
                ax_raster.vlines(trial_times, trial + 0.5, trial + 1.5, color=colors[event], linewidth=0.2)
            ax_raster.axvline(x=0, color='black', linestyle='--', linewidth=0.75)
            ax_raster.set_xlim([t_before, t_after])
            ax_raster.set_xticks(range(t_before, t_after + 1, 1000))
            ax_raster.set_ylim([0, total_trials + 1])
            ax_raster.set_xlabel('')  # don't show x-axis label
            ax_raster.set_xticklabels([])  # don't show x-axis tick labels
            ax_raster.tick_params(axis='both',  # set both x and y axes
                                  which='both',  # set both major and minor ticks
                                  width=0.5,  # tick line width 0.5
                                  length=1.5,  # tick line length 1.5
                                  labelsize=5)  # set tick label font size to 8

            ax_raster.tick_params(axis='x')
            ax_raster.tick_params(axis='y')
            ax_raster.set_xlabel('')  # don't show x-axis label

            print(f"i = {i}, Now plotting..")

            if i == 0:
                ax_raster.set_ylabel('Trials')
                ax_raster.spines['left'].set_visible(True)  # show left edge
            else:
                ax_raster.set_ylabel('')  # don't show y-axis label
                ax_raster.set_yticklabels([])  # don't show y-axis tick labels
                ax_raster.set_yticks([])  # don't show y-axis ticks
                ax_raster.spines['left'].set_visible(False)  # hide left edge

            # Plot PSTH
            ax_psth = axes[1, i]
            rates = gaussian_filter1d(df_psth[event].values, sigma=1)
            error = gaussian_filter1d(df_psth[f'{event}_sem'].values, sigma=1)
            ax_psth.plot(time, rates, label=event, color=colors[event], linewidth=0.75)
            ax_psth.fill_between(time, rates - error, rates + error, color=colors[event], alpha=0.3)
            ax_psth.axvline(x=0, color='black', linestyle='--', linewidth=0.75, label='Event Time')
            ax_psth.set_xlim([t_before, t_after])
            ax_psth.set_xticks(range(t_before, t_after + 1, 1000))
            ax_psth.set_ylim([0, max_rate * 1.1])
            ax_psth.set_xlabel('Time (ms)')
            ax_psth.tick_params(axis='both',  # set both x and y axes
                                which='both',  # set both major and minor ticks
                                width=0.5,  # tick line width 0.5
                                length=1.5,  # tick line length 1.5
                                labelsize=5)  # set tick label font size to 8

            ax_psth.tick_params(axis='x')
            ax_psth.tick_params(axis='y')

            # Only show y-axis label and ticks for first event
            if i == 0:
                ax_psth.set_ylabel('Firing  Rate (Hz)')
                ax_psth.spines['left'].set_visible(True)  # hide left edge
                # ax_psth.legend()
            else:
                ax_psth.set_ylabel('')  # don't show y-axis label
                ax_psth.set_yticklabels([])  # don't show y-axis tick labels
                ax_psth.set_yticks([])  # don't show y-axis ticks
                ax_psth.spines['left'].set_visible(False)  # hide left edge

            is_disable_xy_label = 0
            if is_disable_xy_label == 1:
                ax_raster.set_xlabel('')  # don't show x-axis label
                ax_raster.set_xticklabels([])  # don't show x-axis tick labels
                ax_raster.set_ylabel('')  # don't show y-axis label
                # ax_raster.set_yticklabels([])   # don't show y-axis tick labels

                ax_psth.set_xlabel('')
                ax_psth.set_xticklabels([])  # don't show x-axis tick labels
                ax_psth.set_ylabel('')  # don't show y-axis label
                # ax_psth.set_yticklabels([])   # don't show y-axis tick labels

        if not is_disable_xy_label:
            plt.suptitle(f'unit_seq  {unit_seq:04d} {fig_annotation} Response')

        plt.tight_layout()
        plt.savefig(pne_result_png, format='png', dpi=1200)
        # plt.savefig(pne_result_pdf, format='pdf')
        plt.close(fig)
        print(pne_result_png, ' was saved.')
    else:
        print(unit_seq)


def check_openpyxl():
    # Install openpyxl (if not already installed)
    try:
        import openpyxl
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "openpyxl"])
        import openpyxl


def time_to_ms(time_str):
    # Time conversion function
    try:
        h, m, s = map(float, time_str.split(':'))
        return int((h * 3600 + m * 60 + s) * 1000)
    except AttributeError:
        return np.nan


if __name__ == '__main__':
    pn_root = r"\\NJJK-NAS\visual\66_paper\MANUSCRIPT\20250610-v6-submit\figShare_upload\DATA"

    # Load processing_list
    pne_processing_list = os.path.join(pn_root, r"03_ana\00_list\b1_ana_list.csv")

    t_before = -2000
    t_after = 2000

    group_tasks_and_run(pne_processing_list, t_before, t_after)