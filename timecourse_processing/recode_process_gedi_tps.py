#!/usr/bin/env python
"""Process Jeremy GEDI spreadsheets into annotations for CNN training."""
import numpy as np
import pandas as pd
from scipy.cluster.vq import kmeans2
from tqdm import tqdm


def load_data(csv_file, starting_timepoint=0):
    """Load spreadsheet and process into a dataframe."""
    df = pd.read_csv(csv_file)
    if starting_timepoint > 0:
        for t in range(starting_timepoint):
            existing_columns = list(df.columns.values)
            mod_columns = existing_columns + 1
            column_dict = {k: v for k, v in zip(existing_columns, mod_columns)}
            df = df.rename(columns=column_dict)
            df[min(existing_columns)] = np.nan
    return df


def death_point(df, template, live_thresh=0.05, dead_thresh=0.05):
    """Process spreadsheet into CNN annotations."""
    df_copy = df.copy(deep=True)
    new_df = pd.concat(
        [df_copy['index'], df_copy['plate_well_neuron']], axis=1)
    df.drop('index', 1, inplace=True)
    df.drop('plate_well_neuron', 1, inplace=True)
    ordered_columns = np.argsort(np.asarray([int(x) for x in df.columns]))
    mat_df = df.as_matrix()
    mat_df = mat_df[:, ordered_columns]
    num_rows = df.shape[0]
    dead_find = np.zeros((num_rows))
    index = np.zeros((num_rows))
    pwn = np.asarray([''] * num_rows, dtype=object)
    iteration = np.zeros((num_rows))
    for r_idx in tqdm(range(num_rows)):
        it_row = mat_df[r_idx]
        dead_log = it_row > dead_thresh
        exp_name = new_df.iloc[r_idx]['plate_well_neuron'].split('_')[1]
        exp_mask = template['plate_well_neuron'] == exp_name
        if it_row[0] > dead_thresh:  # If we are dead at the first timepoint
            dead_find[r_idx] = np.nan
            index[r_idx] = np.nan
            pwn[r_idx] = 'empty'
            iteration[r_idx] = np.nan
        else:
            # Only include live images -- if not on this list they are dead
            if any(exp_mask):
                # Only include data from experiments we list in the template
                if dead_log.sum():
                    dif = np.where(dead_log)[0][0]
                    dead_find[r_idx] = int(
                        template[exp_mask][str(dif)].as_matrix())
                    iteration[r_idx] = r_idx
                    index[r_idx] = df_copy['index'].iloc[r_idx]
                    pwn[r_idx] = df_copy['plate_well_neuron'].iloc[r_idx]
            else:
               pwn[r_idx] = 'empty' 
    out_df = pd.DataFrame(
        np.vstack((dead_find, iteration, index, pwn)).transpose(),
        columns=['dead_tp', 'idx', 'index', 'plate_well_neuron'])
    return out_df


def process_template(template, k=3, use_kmeans=True):
    """Process timecourse template into time bins."""
    df = pd.read_csv(template)
    df_copy = df.copy(deep=True)
    df.drop('plate_well_neuron', 1, inplace=True)
    ordered_columns = np.argsort(np.asarray([int(x) for x in df.columns]))
    mat_df = df.as_matrix()
    mat_df = mat_df[:, ordered_columns]
    raveled_mat = mat_df.ravel()
    # raveled_mat = raveled_mat[np.isnan(raveled_mat) == 0]
    raveled_mat[raveled_mat == 0] = np.nan
    masked_data = raveled_mat[np.isnan(raveled_mat) == 0]
    if use_kmeans:
        bin_lengths, groups = kmeans2(masked_data, k, iter=10000)
        fixed_groups = np.zeros((raveled_mat.shape)) * np.nan
        fixed_groups[np.isnan(raveled_mat) == 0] = groups
        groups = fixed_groups
    else:
        sorted_inds = np.argsort(masked_data)
        group_ids = np.array_split(sorted_inds, k)
        groups = np.zeros((len(raveled_mat)))
        for idx, g in enumerate(group_ids):
            for gr in g:
                groups[gr] = idx
        bin_lengths = np.asarray([raveled_mat[sorted_inds==x] for x in range(k)]).ravel()
    sort_idx = np.argsort(bin_lengths)
    sorted_groups = np.zeros((len(groups)), dtype=int) * np.nan
    for idx, g in enumerate(groups):
        if not np.isnan(g):
            sorted_groups[idx] = sort_idx[int(g)]
    print 'Timecourse group means: %s' % np.sort(bin_lengths)
    group_maps = {k: v for k, v in zip(raveled_mat, sorted_groups) if not np.isnan(v)}
    group_maps[0.0] = np.nan
    proc_mat = np.zeros((mat_df.shape))
    for r in range(proc_mat.shape[0]):
        for c in range(proc_mat.shape[1]):
            entry = mat_df[r, c]
            if np.isnan(entry):
                proc_mat[r, c] = entry
            else:
                proc_mat[r, c] = group_maps[entry]
    proc_columns = [str(x) for x in ordered_columns]
    proc_df = pd.DataFrame(proc_mat, columns=proc_columns)
    proc_df['plate_well_neuron'] = df_copy['plate_well_neuron']
    return proc_df


if __name__ == '__main__':
    template = process_template('MLtimesforDrew2.csv', k=2, use_kmeans=True)
    t0 = load_data('T0GEDI_ratio_per_neuron_per_timepoint.csv')
    t0 = death_point(t0, template=template)
    t0.to_csv('proc_T0.csv')
    t1 = load_data('T1GEDI_ratio_per_neuron_per_timepoint.csv')
    t1 = death_point(t1, template=template)
    t1.to_csv('proc_T1.csv')
    combined_data = pd.concat([t0, t1], axis=0)
    combined_data.to_csv('combined_timecourses.csv')
