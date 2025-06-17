import os
import pandas as pd
from datetime import datetime
import glob


def gen_processing_list(pn_root, pn_ana_src_list, pne_ch_info):
    pn_ana = os.path.join(pn_root, '03_ana')
    pn_ana_list = os.path.join(pn_ana, '00_list')
    if not os.path.exists(pn_ana_list):
        os.makedirs(pn_ana_list)

    pn_result = os.path.join(pn_root, '04_result')
    if not os.path.exists(pn_result):
        os.makedirs(pn_result)

    all_files = []

    # Traverse all given directories and find files matching the pattern 'VM*.csv'
    for pn_ana_src_this in pn_ana_src_list:
        for dirpath, _, filenames in os.walk(pn_ana_src_this):
            for file in filenames:
                if file.startswith('VM') and file.endswith('.csv'):
                    all_files.append(os.path.join(dirpath, file))

    # Create a DataFrame with unique file names
    df_spk = pd.DataFrame({'pne_spk': all_files}).drop_duplicates()

    # Add columns for full file name and file name without extension
    df_spk['spk_ne'] = df_spk['pne_spk'].apply(lambda x: os.path.basename(x))
    df_spk['spk_na'] = df_spk['spk_ne'].apply(lambda x: os.path.splitext(x)[0])

    # Split the file name into different components
    def split_file_name(file_name):
        parts = file_name.split('_')
        return {
            'animal_name': parts[0],
            'date': parts[1],
            'exp_tag': parts[2],
            'sys_tag': parts[3],
            'shank_tag': parts[4],
            'ch_tag': parts[5],
            'label_tag': parts[6] if len(parts) > 6 else None
        }

    # Apply the split function to create new columns
    df_spk = df_spk.join(df_spk['spk_na'].apply(lambda x: pd.Series(split_file_name(x))))
    df_spk['ch_day_id'] = df_spk['animal_name'] + '_' + df_spk['date'] + '_' + df_spk['sys_tag'] + '_' + df_spk[
        'shank_tag'] + '_' + df_spk['ch_tag']

    df_spk0 = df_spk[['ch_day_id', 'spk_na', 'spk_ne', 'pne_spk']]

    # Read CSV file and specify brain_area_seq field as integer type
    df_ch_info = pd.read_csv(pne_ch_info, low_memory=False)
    df_ch_info['brain_area_seq'] = df_ch_info['brain_area_seq'].astype('Int64')

    df_list = pd.merge(df_spk0, df_ch_info, on='ch_day_id', how='left')
    df_list['pn_result'] = pn_result

    # Extract task_type
    def extract_task_type(pne_spk):
        parts = pne_spk.split(os.sep)
        if len(parts) > 7:  # Adjust based on path hierarchy
            return parts[-3].split('_')[1]  # Extract required part
        return None

        # Extract task_type

    def extract_task_tmp(pne_spk):
        parts = pne_spk.split(os.sep)
        if len(parts) > 7:  # Adjust based on path hierarchy
            return parts[-3]  # Extract required part
        return None

    df_list['task_type'] = df_list['pne_spk'].apply(extract_task_type)
    df_list['task_tag'] = df_list['pne_spk'].apply(extract_task_tmp)

    # Ensure 'exp_date' field exists and is string type
    if 'exp_date' not in df_list.columns:
        df_list['exp_date'] = df_list['date']  # Assume exp_date is same as date field
    df_list['exp_date'] = df_list['exp_date'].astype(str)

    df_list['task_id'] = df_list['animal_name'] + '_' + df_list['exp_date'] + '_' + df_list['task_tag']

    # For task_type == driving, search for parameter file paths
    def get_para_path(row):
        if row['task_type'] == 'driving':
            para_path = os.path.join(pn_root, '00_common_params', row['animal_name'], '02_params', row['exp_date'])
            para_files = glob.glob(os.path.join(para_path, '*behavior_events.xlsx'))
            if para_files:
                return para_files[0]  # Assume only one matching file
        return None

    df_list['pne_behave_params'] = df_list.apply(get_para_path, axis=1)

    def create_brain_area_folder_name(row):
        brain_area_seq_str = str(row['brain_area_seq']).zfill(2)  # Convert number to 2-digit string
        return os.path.join(brain_area_seq_str + '_' + row['brain_area'])

    df_list['brain_area_folder_name'] = df_list.apply(create_brain_area_folder_name, axis=1)

    # Create 'ch_lb' field, e.g.: ch08_lb26
    df_list['ch_lb'] = df_list['spk_na'].apply(lambda x: '_'.join(x.split('_')[-2:]))

    # ------------------------------------------------
    # Create path list for b2_plot_raster.py  analysis and plotting results
    pn_result_b2 = os.path.join(pn_result, 'b2_raster')
    os.makedirs(pn_result_b2, exist_ok=True)

    def create_pn_result_b2(row):
        return os.path.join(pn_result_b2, row['animal_name'], row['exp_date'], row['task_tag'])

    df_list['pn_result_b2'] = df_list.apply(create_pn_result_b2, axis=1)

    # ------------------------------------------------
    # Create path list for b3_plot_task_raster.py  analysis and plotting results
    b3_pn_result_root = os.path.join(pn_result, 'b3_task_raster')

    def create_b3_pn_result(row):
        return os.path.join(b3_pn_result_root, row['animal_name'], row['exp_date'], row['task_tag'],
                            row['brain_area_folder_name'])

    def create_b3_pne_result_png(row):
        return os.path.join(b3_pn_result_root, row['animal_name'], row['exp_date'], row['task_tag'],
                            row['brain_area_folder_name'], row['brain_area'] + '_' + row['spk_na'] + '.png')

    def create_b3_pne_result_pdf(row):
        return os.path.join(b3_pn_result_root, row['animal_name'], row['exp_date'], row['task_tag'],
                            row['brain_area_folder_name'], row['brain_area'] + '_' + row['spk_na'] + '.pdf')

    def create_b3_fig_annotation(row):
        return os.path.join(row['brain_area'] + '_' + row['spk_na'])

    df_list['b3_pn_result'] = df_list.apply(create_b3_pn_result, axis=1)
    df_list['b3_pne_result_png'] = df_list.apply(create_b3_pne_result_png, axis=1)
    df_list['b3_pne_result_pdf'] = df_list.apply(create_b3_pne_result_pdf, axis=1)
    df_list['b3_fig_annotation'] = df_list.apply(create_b3_fig_annotation, axis=1)

    # ------------------------------------------------
    # Create path list for b4_plot_task_raster.py  analysis and plotting results
    b4_pn_result_root = os.path.join(pn_result, 'b4_task_raster_3d_vid')

    def create_b4_pn_result(row):
        return os.path.join(b4_pn_result_root, row['animal_name'], row['exp_date'], row['task_tag'])

    df_list['b4_pn_result'] = df_list.apply(create_b4_pn_result, axis=1)

    # ------------------------------------------------
    # Create path list for b6_laminar_activity.py analysis and plotting results
    b6_pn_result_root = os.path.join(pn_result, 'b6_laminar_activity')

    def create_b6_pn_result(row):
        return os.path.join(b6_pn_result_root, row['animal_name'], row['exp_date'], row['task_tag'],
                            row['brain_area_folder_name'])

    def create_b6_pna_result_raster(row):
        return os.path.join(b6_pn_result_root, row['animal_name'], row['exp_date'], row['task_tag'],
                            row['brain_area_folder_name'], row['shank_id'] + '_raster')

    def create_b6_pna_result_resp(row):
        return os.path.join(b6_pn_result_root, row['animal_name'], row['exp_date'], row['task_tag'],
                            row['brain_area_folder_name'], row['shank_id'] + '_resp')

    def create_b6_pna_result_resp_sorted(row):
        return os.path.join(b6_pn_result_root, row['animal_name'], row['exp_date'], row['task_tag'],
                            row['brain_area_folder_name'], row['shank_id'] + '_resp_sorted')

    def create_b6_pna_result_test(row):
        return os.path.join(b6_pn_result_root, row['animal_name'], row['exp_date'], row['task_tag'],
                            row['brain_area_folder_name'], row['shank_id'] + '_test')

    def create_b6_fig_annotation(row):
        return os.path.join(row['brain_area'] + '_' + row['shank_id'])

    df_list['b6_pn_result'] = df_list.apply(create_b6_pn_result, axis=1)
    df_list['b6_pna_result_raster'] = df_list.apply(create_b6_pna_result_raster, axis=1)
    df_list['b6_pna_result_resp'] = df_list.apply(create_b6_pna_result_resp, axis=1)
    df_list['b6_pna_result_resp_sorted'] = df_list.apply(create_b6_pna_result_resp_sorted, axis=1)
    df_list['b6_pna_result_test'] = df_list.apply(create_b6_pna_result_test, axis=1)
    df_list['b6_fig_annotation'] = df_list.apply(create_b6_fig_annotation, axis=1)

    df_list['task_ch_lb'] = df_list['task_id'] + '_' + df_list['e_id'] + '_' + df_list['ch_lb']

    # ------------------------------------------------
    # Create path list for b7_gen_hist_resp.py  analysis and plotting results
    b7_pn_result_root = os.path.join(pn_ana, 'b7_gen_hist_resp')

    def create_b7_pn_result(row):
        return os.path.join(b7_pn_result_root, row['animal_name'], row['exp_date'], row['task_tag'],
                            row['brain_area_folder_name'])

    def create_b7_pna_result(row):
        return os.path.join(b7_pn_result_root, row['animal_name'], row['exp_date'], row['task_tag'],
                            row['brain_area_folder_name'], row['shank_id'] + '_hist_resp')

    df_list['b7_pn_result'] = df_list.apply(create_b7_pn_result, axis=1)
    df_list['b7_pna_result'] = df_list.apply(create_b7_pna_result, axis=1)

    # ------------------------------------------------
    # Create path list for b8_granger_causality.py  analysis and plotting results
    b8_pn_result_root = os.path.join(pn_result, 'b8_granger_causality')

    def create_b8_pn_result(row):
        return os.path.join(b8_pn_result_root, row['animal_name'], row['exp_date'], row['task_tag'],
                            row['brain_area_folder_name'])

    def create_b8_pna_result_gc(row):
        return os.path.join(b8_pn_result_root, row['animal_name'], row['exp_date'], row['task_tag'],
                            row['brain_area_folder_name'], row['shank_id'] + '_gc')

    df_list['b8_pn_result'] = df_list.apply(create_b8_pn_result, axis=1)
    df_list['b8_pna_result_gc'] = df_list.apply(create_b8_pna_result_gc, axis=1)

    # ------------------------------------------------
    # Create path list for b9_laminar_activity.py  analysis and plotting results
    b9_pn_result_root = os.path.join(pn_result, 'b9_laminar_activity_aligned')

    def create_b9_pn_result(row):
        return os.path.join(b9_pn_result_root, row['animal_name'], row['exp_date'], row['task_tag'],
                            row['brain_area_folder_name'])

    def create_b9_pna_result_raster(row):
        return os.path.join(b9_pn_result_root, row['animal_name'], row['exp_date'], row['task_tag'],
                            row['brain_area_folder_name'], row['shank_id'] + '_raster')

    def create_b9_pna_result_resp(row):
        return os.path.join(b9_pn_result_root, row['animal_name'], row['exp_date'], row['task_tag'],
                            row['brain_area_folder_name'], row['shank_id'] + '_resp')

    def create_b9_pna_result_resp_sorted(row):
        return os.path.join(b9_pn_result_root, row['animal_name'], row['exp_date'], row['task_tag'],
                            row['brain_area_folder_name'], row['shank_id'] + '_resp_sorted')

    def create_b9_pna_result_test(row):
        return os.path.join(b9_pn_result_root, row['animal_name'], row['exp_date'], row['task_tag'],
                            row['brain_area_folder_name'], row['shank_id'] + '_test')

    def create_b9_fig_annotation(row):
        return os.path.join(row['brain_area'] + '_' + row['shank_id'])

    df_list['b9_pn_result'] = df_list.apply(create_b9_pn_result, axis=1)
    df_list['b9_pna_result_raster'] = df_list.apply(create_b9_pna_result_raster, axis=1)
    df_list['b9_pna_result_resp'] = df_list.apply(create_b9_pna_result_resp, axis=1)
    df_list['b9_pna_result_resp_sorted'] = df_list.apply(create_b9_pna_result_resp_sorted, axis=1)
    df_list['b9_pna_result_test'] = df_list.apply(create_b9_pna_result_test, axis=1)
    df_list['b9_fig_annotation'] = df_list.apply(create_b9_fig_annotation, axis=1)

    df_list['task_ch_lb'] = df_list['task_id'] + '_' + df_list['e_id'] + '_' + df_list['ch_lb']
    # ------------------------------------------------
    # Create path list for d2_plot_dist_resp.py  analysis and plotting results
    d2_pn_result_root = os.path.join(pn_result, 'd2_plot_dist_resp')

    def create_d2_pn_result(row):
        return os.path.join(d2_pn_result_root, row['animal_name'], row['exp_date'], row['task_tag'],
                            row['brain_area_folder_name'])

    def create_d2_pne_result_png(row):
        return os.path.join(d2_pn_result_root, row['animal_name'], row['exp_date'], row['task_tag'],
                            row['brain_area_folder_name'], row['brain_area'] + '_' + row['spk_na'] + '.png')

    def create_d2_pne_result_pdf(row):
        return os.path.join(d2_pn_result_root, row['animal_name'], row['exp_date'], row['task_tag'],
                            row['brain_area_folder_name'], row['brain_area'] + '_' + row['spk_na'] + '.pdf')

    def create_d2_fig_annotation(row):
        return os.path.join(row['brain_area'] + '_' + row['spk_na'])

    df_list['d2_pn_result'] = df_list.apply(create_d2_pn_result, axis=1)
    df_list['d2_pne_result_png'] = df_list.apply(create_d2_pne_result_png, axis=1)
    df_list['d2_pne_result_ppdf'] = df_list.apply(create_d2_pne_result_pdf, axis=1)
    df_list['d2_fig_annotation'] = df_list.apply(create_d2_fig_annotation, axis=1)

    # ------------------------------------------------
    # Create path list for d3_plot_dist_resp_centered.py  analysis and plotting results
    d3_pn_result_root = os.path.join(pn_result, 'd3_plot_dist_resp_centered')

    def create_d3_pn_result(row):
        return os.path.join(d3_pn_result_root, row['animal_name'], row['exp_date'], row['task_tag'],
                            row['brain_area_folder_name'])

    def create_d3_pne_result_png(row):
        return os.path.join(d3_pn_result_root, row['animal_name'], row['exp_date'], row['task_tag'],
                            row['brain_area_folder_name'], row['brain_area'] + '_' + row['spk_na'] + '.png')

    def create_d3_pne_result_pdf(row):
        return os.path.join(d3_pn_result_root, row['animal_name'], row['exp_date'], row['task_tag'],
                            row['brain_area_folder_name'], row['brain_area'] + '_' + row['spk_na'] + '.pdf')

    def create_d3_fig_annotation(row):
        return os.path.join(row['brain_area'] + '_' + row['spk_na'])

    df_list['d3_pn_result'] = df_list.apply(create_d3_pn_result, axis=1)
    df_list['d3_pne_result_png'] = df_list.apply(create_d3_pne_result_png, axis=1)
    df_list['d3_pne_result_pdf'] = df_list.apply(create_d3_pne_result_pdf, axis=1)
    df_list['d3_fig_annotation'] = df_list.apply(create_d3_fig_annotation, axis=1)

    # ------------------------------------------------
    # Create path list for d4_plot_prefered_resp.py  analysis and plotting results
    d4_pn_result_root = os.path.join(pn_result, 'd4_plot_prefered_resp')

    def create_d4_pn_result(row):
        return os.path.join(d4_pn_result_root, row['animal_name'], row['exp_date'], row['task_tag'],
                            row['brain_area_folder_name'])

    def create_d4_pne_result_png(row):
        return os.path.join(d4_pn_result_root, row['animal_name'], row['exp_date'], row['task_tag'],
                            row['brain_area_folder_name'], row['brain_area'] + '_' + row['spk_na'] + '.png')

    def create_d4_pne_result_pdf(row):
        return os.path.join(d4_pn_result_root, row['animal_name'], row['exp_date'], row['task_tag'],
                            row['brain_area_folder_name'], row['brain_area'] + '_' + row['spk_na'] + '.pdf')

    def create_d4_fig_annotation(row):
        return os.path.join(row['brain_area'] + '_' + row['spk_na'])

    df_list['d4_pn_result'] = df_list.apply(create_d4_pn_result, axis=1)
    df_list['d4_pne_result_png'] = df_list.apply(create_d4_pne_result_png, axis=1)
    df_list['d4_pne_result_pdf'] = df_list.apply(create_d4_pne_result_pdf, axis=1)
    df_list['d4_fig_annotation'] = df_list.apply(create_d4_fig_annotation, axis=1)

    # ------------------------------------------------
    # Create path list for d5_plot_dist_resp_centered.py  analysis and plotting results
    d5_pn_result_root = os.path.join(pn_result, 'd5_plot_dist_resp_centered')

    def create_d5_pn_result(row):
        return os.path.join(d5_pn_result_root, row['animal_name'], row['exp_date'], row['task_tag'],
                            row['brain_area_folder_name'])

    def create_d5_pne_result_png(row):
        return os.path.join(d5_pn_result_root, row['animal_name'], row['exp_date'], row['task_tag'],
                            row['brain_area_folder_name'], row['brain_area'] + '_' + row['spk_na'] + '.png')

    def create_d5_pne_result_pdf(row):
        return os.path.join(d5_pn_result_root, row['animal_name'], row['exp_date'], row['task_tag'],
                            row['brain_area_folder_name'], row['brain_area'] + '_' + row['spk_na'] + '.pdf')

    def create_d5_fig_annotation(row):
        return os.path.join(row['brain_area'] + '_' + row['spk_na'])

    def create_d5_pne_behavior(row):
        return os.path.join(pn_root, '02_data', '03_behavior', row['animal_name'], row['exp_date'], row['task_tag'],
                            '04_coordinate_data', 'behavior.csv')

    df_list['d5_pn_result'] = df_list.apply(create_d5_pn_result, axis=1)
    df_list['d5_pne_result_png'] = df_list.apply(create_d5_pne_result_png, axis=1)
    df_list['d5_pne_result_pdf'] = df_list.apply(create_d5_pne_result_pdf, axis=1)
    df_list['d5_fig_annotation'] = df_list.apply(create_d5_fig_annotation, axis=1)
    df_list['d5_pne_behavior'] = df_list.apply(create_d5_pne_behavior, axis=1)

    # ------------------------------------------------
    # Create path list for d6_plot_dist_resp_centered.py  analysis and plotting results
    d6_pn_result_root = os.path.join(pn_result, 'd6_plot_dist_resp_centered_2_targets_tuning')

    def create_d6_pn_result(row):
        return os.path.join(d6_pn_result_root, row['animal_name'], row['exp_date'], row['task_tag'],
                            row['brain_area_folder_name'])

    def create_d6_pne_result_png(row):
        return os.path.join(d6_pn_result_root, row['animal_name'], row['exp_date'], row['task_tag'],
                            row['brain_area_folder_name'], row['brain_area'] + '_' + row['spk_na'] + '.png')

    def create_d6_pne_result_pdf(row):
        return os.path.join(d6_pn_result_root, row['animal_name'], row['exp_date'], row['task_tag'],
                            row['brain_area_folder_name'], row['brain_area'] + '_' + row['spk_na'] + '.pdf')

    def create_d6_fig_annotation(row):
        return os.path.join(row['brain_area'] + '_' + row['spk_na'])

    df_list['d6_pn_result'] = df_list.apply(create_d6_pn_result, axis=1)
    df_list['d6_pne_result_png'] = df_list.apply(create_d6_pne_result_png, axis=1)
    df_list['d6_pne_result_pdf'] = df_list.apply(create_d6_pne_result_pdf, axis=1)
    df_list['d6_fig_annotation'] = df_list.apply(create_d6_fig_annotation, axis=1)

    # ------------------------------------------------
    # Create path list for d7_plot_single_trace_with_arrow.py  analysis and plotting results
    d7_pn_result_root = os.path.join(pn_result, 'd7_plot_single_trace_with_arrow')

    def create_d7_pn_result(row):
        return os.path.join(d7_pn_result_root, row['animal_name'], row['exp_date'], row['task_tag'])

    def create_d7_pne_behavior(row):
        return os.path.join(pn_root, '02_data', '03_behavior', row['animal_name'], row['exp_date'], row['task_tag'],
                            '04_coordinate_data', 'behavior.csv')

    df_list['d7_pn_result'] = df_list.apply(create_d7_pn_result, axis=1)
    df_list['d7_pne_behavior'] = df_list.apply(create_d7_pne_behavior, axis=1)

    # ------------------------------------------------
    # Create path list for d8_plot_strength_vs_firing_rate.py  analysis and plotting results
    d8_pn_result_root = os.path.join(pn_result, 'd8_plot_strength_vs_firing_rate')

    def create_d8_pn_result(row):
        return os.path.join(d8_pn_result_root, row['animal_name'], row['exp_date'], row['task_tag'],
                            row['brain_area_folder_name'])

    def create_d8_pne_behavior(row):
        return os.path.join(pn_root, '02_data', '03_behavior', row['animal_name'], row['exp_date'], row['task_tag'],
                            '04_coordinate_data', 'behavior.csv')

    def create_d8_pne_result_png(row):
        return os.path.join(d8_pn_result_root, row['animal_name'], row['exp_date'], row['task_tag'],
                            row['brain_area_folder_name'], row['brain_area'] + '_' + row['spk_na'] + '.png')

    def create_d8_pne_result_pdf(row):
        return os.path.join(d8_pn_result_root, row['animal_name'], row['exp_date'], row['task_tag'],
                            row['brain_area_folder_name'], row['brain_area'] + '_' + row['spk_na'] + '.pdf')

    def create_d8_fig_annotation(row):
        return os.path.join(row['brain_area'] + '_' + row['spk_na'])

    df_list['d8_pn_result'] = df_list.apply(create_d8_pn_result, axis=1)
    df_list['d8_pne_behavior'] = df_list.apply(create_d8_pne_behavior, axis=1)
    df_list['d8_pne_result_png'] = df_list.apply(create_d8_pne_result_png, axis=1)
    df_list['d8_pne_result_pdf'] = df_list.apply(create_d8_pne_result_pdf, axis=1)
    df_list['d8_fig_annotation'] = df_list.apply(create_d8_fig_annotation, axis=1)

    # ----------------------------------------------------------------
    # Read unit_filter_out.csv  file and merge
    filter_files = glob.glob(os.path.join(pn_root, '00_common_params', '**', 'unit_filter_out.csv'), recursive=True)
    df_filter_out = pd.concat([pd.read_csv(file) for file in filter_files], ignore_index=True)

    # Calculate df_filter_out['task_ch_lb']
    df_filter_out['task_ch_lb'] = df_filter_out.apply(lambda
                                                          row: f"{row['animal_name']}_{row['exp_date']}_{row['task_tag']}_S{str(row['e_seq']).zfill(2)}_{row['shank_tag']}_ch{str(row['ch_seq']).zfill(2)}_lb{str(row['lb_seq']).zfill(2)}",
                                                      axis=1)

    # Remove rows from df_list that match df_filter_out['task_ch_lb']
    df_list = df_list[~df_list['task_ch_lb'].isin(df_filter_out['task_ch_lb'])]

    # --> Save list file and backup
    pne_list = os.path.join(pn_ana_list, 'b1_ana_list.csv')
    df_list.to_csv(pne_list, index=False)

    # --> Save list.csv  backup with precise timestamp
    now = datetime.now()
    formatted_time = now.strftime('%Y%m%d_%H_%M_%S')
    pn_list_bak = os.path.join(pn_ana_list, 'b1_ana_list')
    pne_list_bak = os.path.join(pn_list_bak, 'b1_ana_list@' + formatted_time + '.csv')
    os.makedirs(pn_list_bak, exist_ok=True)

    df_list.to_csv(pne_list_bak, index=False)

    print(f'---> Processing list was saved to {pne_list}')
    print(f'---> Processing list was backed up at {pne_list_bak}')
    print('------> Units counts =', len(df_list))


if __name__ == "__main__":
    pn_root = r"\\NJJK-NAS\visual\66_paper\MANUSCRIPT\20250610-v6-submit\figShare_upload\DATA"
    pne_ch_info = os.path.join(pn_root, '00_common_params', 'ch_info.csv')

    pn_ana_src_list = [
        pn_root + r'\02_data\03_spk\VM20\20231010\01_elph\01_driving',
        pn_root + r'\02_data\03_spk\VM23\20231108\01_elph\01_driving',
    ]  # Replace with actual folder paths

    gen_processing_list(pn_root, pn_ana_src_list, pne_ch_info)
