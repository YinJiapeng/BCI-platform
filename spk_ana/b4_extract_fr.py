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
import pickle


def group_tasks_and_run(pne_processing_list, datapath, t_before=-2000, t_after=2000, bin_size=500):
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

            unique_pne_para = df_list['pne_behave_params'].unique()
            if len(unique_task_types) == 1:
                pne_behave_params = unique_pne_para[0]
                df_behave_params = pd.read_excel(pne_behave_params.replace("\\\\NJJK-NAS\\visual\\66_paper\\MANUSCRIPT\\20250610-v6-submit\\DATA", datapath))

                extract_and_plot(datapath, df_list, df_behave_params, t_before, t_after, bin_size)

        elif len(unique_task_types) == 1 and unique_task_types[0] == 'movie':
            print("The unique value of task_type is 'movie', skip...")


def extract_and_plot(rootpath, df_list, df_behave_params, t_before, t_after, bin_size):
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
        results = [pool.apply_async(process_unit, args=(rootpath, unit_seq, df_behave_params, df_list, t_before, t_after, bin_size)) for unit_seq
                   in df_list['unit_seq']]
        [result.get() for result in results]


def process_unit(rootpath, unit_seq, df_behave_params, df_list, t_before, t_after, bin_size):
    events = {
        't_reach_start_seg': 't_reach_start',
        't_touch_fruit_seg': 't_touch_fruit',
        't_touch_mouth_seg': 't_touch_mouth',
        # 't_leave_this_seg': 't_leave_this'
    }

    row = df_list[df_list['unit_seq'] == unit_seq].iloc[0]
    pne_spk = row['pne_spk']
    pn_result = row['b3_pn_result']
    pne_result_png = row['b3_pne_result_png']

    spike_data = pd.read_csv(pne_spk.replace("\\\\NJJK-NAS\\visual\\66_paper\\MANUSCRIPT\\20250610-v6-submit\\DATA", rootpath), usecols=['globalTime'])
    spike_data['globalTime'] = spike_data['globalTime'].astype(float)

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
        df_psth[f'{event_name}_smoothed'] = gaussian_filter1d(mean_rates, sigma=1)

    # save psths
    savedir = pn_result.replace("\\\\NJJK-NAS\\visual\\66_paper\\MANUSCRIPT\\20250610-v6-submit\\DATA", rootpath)
    _, filename = os.path.split(pne_result_png)
    os.makedirs(savedir.replace("04_result", f'04_result_{bin_size}ms'), exist_ok=True)
    saved_all_filename = filename.replace("png", "pkl")
    saved_mean_filename = filename.replace("png", "csv")
    with open(os.path.join(savedir.replace("04_result", f'04_result_{bin_size}ms'), f'FR_All_{saved_all_filename}'), 'wb') as fp:
        pickle.dump(all_psths, fp)
        print('dictionary contains all fr saved successfully to file')
    df_psth.to_csv(os.path.join(savedir.replace("04_result", f'04_result_{bin_size}ms'), f'FR_Plot_{saved_mean_filename}'), encoding='utf-8', index=False)


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

    # 加载 processing_list
    pne_processing_list = os.path.join(pn_root, r"03_ana\00_list\b1_ana_list.csv")

    t_before = -2000
    t_after = 2000
    bin_size = 50 # bin size in ms
    # bin_size = 500 # bin size in ms

    group_tasks_and_run(pne_processing_list, pn_root, t_before, t_after, bin_size)