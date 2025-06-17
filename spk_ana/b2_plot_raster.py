import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import subprocess
import sys
import multiprocessing
from multiprocessing import Pool
from create_figure.create_figure import CreatFigure


def group_tasks_and_run(pne_processing_list, **kwargs):
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
                extract_and_plot(df_list, pne_behave_params)  # Main execution part

        elif len(unique_task_types) == 1 and unique_task_types[0] == 'movie':
            print("The unique value of task_type is 'movie', skip...")


def extract_and_plot(df_list, pne_behave_params, **kwargs):
    check_openpyxl()

    t_before = kwargs.setdefault('t_before', 2000)  # ms, initial time range before event
    t_after = kwargs.setdefault('t_after', 2000)  # ms, initial time range after event
    n_threads = kwargs.get('n_threads')

    print(f'Now reading {pne_behave_params}')
    df_behave_params = pd.read_excel(pne_behave_params)

    # Sort df_list
    brain_area_order = dict(zip(df_list['brain_area'], df_list['brain_area_seq']))
    df_list['brain_area_order'] = df_list['brain_area'].map(brain_area_order)
    df_list = df_list.sort_values(by='brain_area_order')

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

    pn_result = df_list['pn_result_b2'][1]

    # Create result directory
    os.makedirs(pn_result, exist_ok=True)

    # # ------ Normal for loop for debugging: ------
    # for _, row in df_behave_params.iterrows():
    #     process_row(row, df_list, t_before, t_after, pn_result)

    # Use multiprocessing for parallel processing:
    with Pool(n_threads) as pool:
        results = [pool.apply_async(process_row, args=(row, df_list, t_before, t_after, pn_result)) for _, row in
                   df_behave_params.iterrows()]
        for result in results:
            result.get()


def process_row(row, df_list, t_before, t_after, pn_result):
    t_reach_start = row['t_reach_start']
    t_touch_fruit = row['t_touch_fruit']
    t_touch_mouth = row['t_touch_mouth']
    t_leave = row['t_leave']
    trial_seq = int(row['trial_seq'])  # Convert trial_seq to integer

    t_start = t_reach_start - t_before
    t_end = t_leave + t_after

    spike_times_all = []

    for idx, processing_row in df_list.iterrows():
        pne_spk = processing_row['pne_spk']
        color = processing_row['color_bak']

        spike_data = pd.read_csv(pne_spk, usecols=['globalTime'])
        spike_data['globalTime'] = spike_data['globalTime'].astype(float)

        # Extract spike data within time range
        time_range = spike_data[(spike_data['globalTime'] >= t_start) & (spike_data['globalTime'] <= t_end)]

        # Add all spike times to spike_times_all
        spike_times_all.append((time_range['globalTime'].values, color))

    # Plot raster plot
    plot_raster(spike_times_all, f"Spike Data for Trial {trial_seq}", t_start, t_end, t_reach_start, t_touch_fruit, t_touch_mouth,
                t_leave, pn_result, trial_seq)


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


import os
import numpy as np
import matplotlib.pyplot as plt


def plot_raster(spike_times, title, t_start, t_end, t_reach_start, t_touch_fruit, t_touch_mouth, t_leave, pn_result, trial_seq):
    # Create Figure
    fig = CreatFigure(width_mm=90, aspect_ratio=1, is_remove_ur_edge=0)
    ax = fig.ax

    # Plot raster
    for i, (spikes, color) in enumerate(spike_times):
        if spikes.size > 0:  # Ensure there are spikes
            ax.vlines(spikes / 1000, i + 0.5, i + 1.5, color=color, linewidth=0.3)

    n_ch = len(spike_times)  # Directly get number of channels

    # Set X-axis time range (convert to seconds)
    t_plot = 18000  # ms
    ax.set_xlim([t_start / 1000, (t_start + t_plot) / 1000])
    ax.set_ylim([0, n_ch])

    # Set title and axis labels
    ax.set_title(title, fontsize=7)
    ax.set_xlabel('Time  (s)', fontsize=7)
    ax.set_ylabel('Channels', fontsize=7)

    # Set xticks
    x_min, x_max = ax.get_xlim()
    xticks = np.arange(np.floor(x_min), np.ceil(x_max) + 1, 2)  # Tick every 2 seconds
    xticks = xticks[(xticks >= x_min) & (xticks <= x_max)]
    ax.set_xticks(xticks)
    ax.set_xticklabels(xticks.astype(int), fontsize=5)

    # Set y-axis tick font
    ax.tick_params(axis='both', labelsize=5, width=0.5)

    # Set line width for all four axes
    for spine in ax.spines.values():
        spine.set_linewidth(0.5)

        # Plot event marker lines
    event_lines = [
        (t_reach_start, 'black', 't_reach_start'),
        (t_touch_fruit, 'red', 't_touch_fruit'),
        (t_touch_mouth, 'blue', 't_touch_mouth'),
        # (t_leave, 'orange', 't_leave') # Optional
    ]
    for t, color, label in event_lines:
        # ax.axvline(x=t / 1000, color='black', linestyle='--', linewidth=0.5, label=label)
        ax.axvline(x=t  / 1000, color=color, linestyle='--', linewidth=0.5, label=label)

    # Save images
    save_path_jpg = os.path.join(pn_result, f"trial_{trial_seq:03d}.jpg")
    save_path_pdf = os.path.join(pn_result, f"trial_{trial_seq:03d}.pdf")

    plt.savefig(save_path_jpg, dpi=1200)
    plt.savefig(save_path_pdf, format='pdf')
    plt.close()
    
    print(f'Trial {trial_seq} was saved to: {save_path_jpg}')


if __name__ == '__main__':
    pn_root = r"\\NJJK-NAS\visual\66_paper\MANUSCRIPT\20250610-v6-submit\figShare_upload\DATA"

    # Load processing_list
    pne_processing_list = os.path.join(pn_root, r"03_ana\00_list\b1_ana_list.csv")

    total_cores = multiprocessing.cpu_count()

    # Initial parameters
    params = {
        'n_threads': int(total_cores * 0.6),  # Number of CPU cores to use (60%)
        't_before': 2000,  # ms, time before event to capture
        't_after': 2000,  # ms, time after event to capture
    }

    group_tasks_and_run(pne_processing_list, **params)