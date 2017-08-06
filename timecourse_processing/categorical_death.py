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


def death_point(df, live_thresh=0.05, dead_thresh=0.05):
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
        if live_log.sum():
            live_find[r_idx] = np.where(live_log)[0][0] + 1
        else:
            live_find[r_idx] = 0
        if dead_log.sum():
            when_to_die = np.where(dead_log)[0][0]
            # if when_to_die == 2:  # Cell is either dead or about to die
            #     dead_find[r_idx] = 2
            # elif when_to_die == 1:
            #     dead_find[r_idx] = 1
            if when_to_die == 1:
                dead_find[r_idx] = 1
            elif when_to_die == 0:
                dead_find[r_idx] = 0
            else:
                dead_find[r_idx] = 100  # Any cells 
        else:
            # dead_find[r_idx] = 2  # This cell is healthy for a while
            dead_find[r_idx] = 100  # After talking to J, this could be hurting us
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
