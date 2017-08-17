#!/usr/bin/env python
import numpy as np
import pandas as pd


def load_data(csv_file, starting_timepoint=0):
    df = pd.read_csv(csv_file)
    if starting_timepoint > 0:
        for t in range(starting_timepoint):
            existing_columns = list(df.columns.values)
            mod_columns = existing_columns + 1
            column_dict = {k: v for k, v in zip(existing_columns, mod_columns)}
            df = df.rename(columns=column_dict)
            df[min(existing_columns)] = np.nan
    return df


def death_point(
        df,
        live_thresh=0.05,
        dead_thresh=0.05,
        mask_timepoint_value=-999.):
    df_copy = df.copy(deep=True)
    new_df = pd.concat(
        [df_copy['index'], df_copy['plate_well_neuron']], axis=1)
    df.drop('index', 1, inplace=True)
    df.drop('plate_well_neuron', 1, inplace=True)
    ordered_columns = np.argsort(np.asarray([int(x) for x in df.columns]))
    mat_df = df.as_matrix()
    mat_df = mat_df[:, ordered_columns]
    num_rows = df.shape[0]
    live_find = np.zeros((num_rows))
    dead_find = np.zeros((num_rows))
    for r_idx in range(num_rows):
        it_row = mat_df[r_idx]
        live_log = it_row < live_thresh
        dead_log = it_row > dead_thresh
        if dead_log[0]:  # Already dead, exclude
            dead_find[r_idx] = mask_timepoint_value
        else:
            # dead_find[r_idx] = df_copy['1'].iloc[r_idx]  # int(dead_log[1])  # Binary dead/live next timepoint 
            dead_find[r_idx] = int(dead_log[1])  # Binary dead/live next timepoint 
    new_df['live_tp'] = live_find
    new_df['dead_tp'] = dead_find
    return new_df


if __name__ == '__main__':
    t0 = load_data('T0GEDI_ratio_per_neuron_per_timepoint.csv')
    t0 = death_point(t0)
    t0.to_csv('proc_T0.csv')
    t1 = load_data('T1GEDI_ratio_per_neuron_per_timepoint.csv')
    t1 = death_point(t1)
    t1.to_csv('proc_T1.csv')
    combined_data = pd.concat([t0, t1], axis=0)

    # Mask glutamate
    mask = combined_data['plate_well_neuron'].str.contains('Glutamate')
    combined_data = combined_data.loc[mask == False]
    combined_data.to_csv('combined_timecourses.csv')
