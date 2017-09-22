#!/usr/bin/env python
"""Process cell images to identify their disease."""

import re
import numpy as np
import pandas as pd
from tqdm import tqdm


def load_data(csv_file, starting_timepoint=0):
    """Load and process data from CSVs."""
    df = pd.read_csv(csv_file)
    if starting_timepoint > 0:
        for t in range(starting_timepoint):
            existing_columns = list(df.columns.values)
            mod_columns = existing_columns + 1
            column_dict = {k: v for k, v in zip(existing_columns, mod_columns)}
            df = df.rename(columns=column_dict)
            df[min(existing_columns)] = np.nan
    return df


def id_experiment(x):
    """Regex split x to ID the experiment."""
    split_x = x.split('_')
    return split_x[1], re.split('\d+', split_x[2])[0]


def tag_with_disease(
        df,
        template=None,
        row_key='plate_well_neuron'):
    """Search the template CSV for the label of each image."""
    if template is not None:
        template = load_data(template)
    else:
        raise RuntimeError(
            'Please pass a template file with info about cell disease.')

    num_rows = df.shape[0]
    rows = []
    for idx, it_row in tqdm(df.iterrows(), total=num_rows):
        line, well = id_experiment(it_row[row_key])
        line_check = template['line'] == line
        well_check = template['wells'] == well
        both = line_check & well_check
        if any(both):
            both_rows = both[both == True]
            if len(both_rows) > 1:
                raise RuntimeError(
                    'Found multiple matches in the template (bad template).')
            it_row['disease'] = template.iloc[both_rows.index[0]]['type']
            rows += [it_row]
    return pd.DataFrame(rows)


if __name__ == '__main__':
    t0 = load_data('T0GEDI_ratio_per_neuron_per_timepoint.csv')
    t1 = load_data('T1GEDI_ratio_per_neuron_per_timepoint.csv')
    t0.to_csv('disease_proc_T0.csv')
    t1.to_csv('disease_proc_T1.csv')
    combined_data = pd.concat([t0, t1], axis=0)
    combined_data = tag_with_disease(
        combined_data,
        template='autopsy_huntington_parkinson.csv')
    combined_data.to_csv('combined_timecourses.csv')
