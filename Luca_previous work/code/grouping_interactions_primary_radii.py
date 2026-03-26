from line_profiler import LineProfiler
import pickle
import sys
import os
import os.path as path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import itertools
from scipy.spatial.distance import cdist


### 0----------------------------------------0000000000000000000--------------------------------------------0 ###
data_to_group = 'full_detector_bulk_140kev_5x20x20cm_20R_100H_water_phantom500_MBq_no_noise_pd.pkl'
radii_distribution_by_category = 'radii_distribution_no_water.pkl'
low_energy_time_unc = 11.5 # in ns
high_energy_time_unc = 3 # in ns
N_groups_to_make = 5000
ENC = 60
px_th = 2
### 0----------------------------------------0000000000000000000--------------------------------------------0 ###



def compute_weights_and_radii(data_dict,perc):
    """
    This takes in the radius categorized by interaction number and it computes the thresholds at a certain percentile
    as well as weights based on the relative population of different event chain lengths.
    """
    #------------------#
    percentile = perc
    #------------------#

    # Step 1: Compute weights based on the number of entries in each key
    num_entries = {key: len(values) for key, values in data_dict.items()}
    max_entries = max(num_entries.values())

    weights = {key: num_entries[key] / max_entries for key in data_dict}

    # Step 2: Compute the radius for each key
    radii = {}
    for key, values in data_dict.items():
        sorted_values = np.sort(values)
        # Find the value corresponding to the 75th percentile
        radius = np.percentile(sorted_values, percentile)
        radii[key] = radius

    # Step 3: Return the weights and radii arrays
    weights_array = np.array([weights[key] for key in range(2,len(data_dict))])
    radii_array = np.array([radii[key] for key in range(2,len(data_dict))])

    return weights_array, radii_array


def process_interactions_v2(df, thresholds,time_uncertainty_model, small_coeff,big_coeff):
    """
    this is the code with the first tentative algorithm proposed. It divides the interactions into time window 3 times the bigger uncertainty.
    Then for each time window it takes the interaction with the lowest time stamp, and from it, it tries to group it to the closest interactions
    It will keep grouping until the current group radius is above the threshold for that length category. After that it saves the resulting group
    in a pandas dataframe and it remove the grouped interactions from the list of intetractions to group in the next loop and it repeat
    """
    #TODO: explain the double window

    # Sort the dataframe by timestamp (ascending)
    df = df.sort_values(by='time')

    grouped_interactions = []  # To keep track of all processed groups

    # Iterate through unique event IDs
    start_index = 0
    end_index = 10000
    ungrouped_interactions = df.iloc[start_index:end_index].copy()
    start_index = end_index
    max_index = len(df)
    count = 0

    while len(ungrouped_interactions) > 0 and count < N_groups_to_make:
        # Get the first interaction and start a new group
        count +=1
        #print(count)
        first_interaction = ungrouped_interactions.iloc[0]
        t_wind_small = small_coeff * np.interp(first_interaction['E_dep'], time_uncertainty_model['E_dep'], time_uncertainty_model['total_uncertainty'])
        t_wind_big = big_coeff * time_uncertainty_model['total_uncertainty'].iloc[2]


        group_time_window_1 = first_interaction['time'] + t_wind_small
        group_time_window_2 = first_interaction['time'] + t_wind_big
        current_event = ungrouped_interactions.iloc[[0]]
        window_interactions_small = ungrouped_interactions[ungrouped_interactions['time'] <= group_time_window_1]
        window_interactions_big = ungrouped_interactions[
            (group_time_window_1 < ungrouped_interactions['time']) &
            (ungrouped_interactions['time'] <= group_time_window_2) &
            (ungrouped_interactions['E_dep'] <= first_interaction['E_dep'])
            ]

        min_timestamp_row = ungrouped_interactions['time'].idxmin() #TODO: make this faster by sorting with respect to time stamp and taking iloc 0
        window_interactions= pd.concat([window_interactions_small, window_interactions_big], axis=0, ignore_index=False)
        window_interactions = window_interactions.drop(min_timestamp_row)

        '''else:

            group_time_window_2 = first_interaction['time'] + t_wind_big
            current_event = ungrouped_interactions.iloc[[0]]
            window_interactions = ungrouped_interactions[ungrouped_interactions['time'] <= group_time_window_2]
            min_timestamp_row = ungrouped_interactions['time'].idxmin() #TODO: make this faster by sorting with respect to time stamp and taking iloc 0
            window_interactions = window_interactions.drop(min_timestamp_row)'''
        # Coordinates for the first group member
        current_coords = np.array([first_interaction[['init_x', 'init_y', 'init_z']]],dtype=np.float64)
        all_coords = current_coords
        # Check if group size exceeds 1
        chain_tot_energy_sum = current_event['E_dep'].iloc[-1]



        while len(window_interactions)>0:
            # Find the closest interaction based on x, y, z
            window_interactions[['init_x', 'init_y', 'init_z']] = window_interactions[
                ['init_x', 'init_y', 'init_z']].apply(pd.to_numeric, errors='coerce')
            distances = cdist(current_coords, window_interactions[['init_x', 'init_y', 'init_z']])
            closest_interaction_idx = np.argmin(distances)
            closest_interaction = window_interactions.iloc[closest_interaction_idx]

            # Calculate centroid and radius of current group
            all_coords = np.append(all_coords, [closest_interaction[['init_x', 'init_y', 'init_z']]], axis=0)
            centroid = np.mean(all_coords, axis=0)
            diff = np.asarray(all_coords - centroid, dtype=np.float64)
            distances = np.linalg.norm(diff, axis=1)
            radius = np.max(distances)
            chain_tot_energy_sum += closest_interaction['E_dep']
            # Check if radius exceeds the threshold
            group_size = len(all_coords)
            if radius <= thresholds[group_size - 2] and chain_tot_energy_sum < 145:
                current_event=pd.concat([current_event,window_interactions.iloc[[closest_interaction_idx]]],ignore_index=False)
                current_coords = np.array([centroid],dtype=np.float64)
                window_interactions = window_interactions.drop(window_interactions.index[closest_interaction_idx])
            else:

                end_index = start_index + len(current_event)
                ungrouped_interactions = ungrouped_interactions.drop(current_event.index, errors = ' ignore')
                if end_index < max_index:
                    ungrouped_interactions= pd.concat([ungrouped_interactions, df.iloc[start_index:end_index]])
                start_index = end_index

                #TODO: finish checking that it reason correctly the algorithms
                break

        if len(window_interactions) == 0:
            end_index = start_index + len(current_event)
            ungrouped_interactions = ungrouped_interactions.drop(current_event.index, errors=' ignore')
            if end_index < max_index:
                ungrouped_interactions = pd.concat([ungrouped_interactions, df.iloc[start_index:end_index]])
            start_index = end_index

        # Store the current group after checking the threshold

        column_to_add = [count] * len(current_event)
        current_event.insert(0,'grouped_ID',column_to_add)
        grouped_interactions.append(current_event)


    grouped_interactions = pd.concat(grouped_interactions, ignore_index=True)
    return grouped_interactions




df = pd.read_pickle(data_to_group)
df['E_dep'] = df['E_dep']*1000
with open(radii_distribution_by_category, "rb") as image_file:
    radii_by_category = (pickle.load(image_file))

#df['E_dep'] = df['E_dep']
df = df[df['E_dep'] >= 0.1].reset_index(drop=True)

time_unc_models = pd.read_csv('time_uncertainty_models.csv')
for name in time_unc_models.columns[10:12]:
    if name != 'Collection time uncertainty (ns)':

        time_model = np.sqrt(time_unc_models['Collection time uncertainty (ns)']**2 + time_unc_models[name]**2)
        unc_model = pd.DataFrame({
            'E_dep': time_unc_models['Deposited energy (keV)'],
            'total_uncertainty': time_model
        })
    else:
        time_model = time_unc_models['Collection time uncertainty (ns)']
        unc_model = pd.DataFrame({
            'E_dep': time_unc_models['Deposited energy (keV)'],
            'total_uncertainty': time_model
        })

    ref_E = unc_model['E_dep'].values
    ref_T = unc_model['total_uncertainty'].values

    # For each E_dep in df, find index of closest in ref_df
    interp_add_time = np.interp(np.asarray(df['E_dep'].values,dtype=float), ref_E, ref_T)

    # Add the corresponding add_time
    for counter in [0, 1, 2]:
        noise = np.random.normal(loc=0, scale=interp_add_time, size=len(df))
        energy_noise = np.random.normal(loc=0, scale=ENC, size=len(df)) * 3.6 / 1000
        print(counter)
        for small_coeff in [3,4]:
            for big_coeff in [1,2,3]:
                for perc in [60,75]:

                    df_unc = df.copy()
                    df_unc['time'] = df_unc['time'] + noise
                    df_unc['E_dep'] = df_unc['E_dep'] + energy_noise
                    weights, thresholds = compute_weights_and_radii(radii_by_category,perc)
                    #df_removed = df_unc[df_unc['E_dep'] < 2].copy()
                    df_unc = df_unc[df_unc['E_dep'] >= px_th].reset_index(drop=True)
                    #weights = np.pad(weights, (0, 20), mode='constant',
                    #                 constant_values=0)  ## pad with 0 to make sure that the index in the process_interaction function never goes above the limit
                    thresholds = np.pad(thresholds, (0, 20), mode='constant', constant_values=0)
                    grouped  = process_interactions_v2(df_unc,thresholds,unc_model,small_coeff,big_coeff)
                    #grouped = pd.concat([grouped, df_removed], ignore_index=True)
                    file_name = os.path.basename(data_to_group)
                    file_name = os.path.splitext(file_name)[0]
                    #file_name_df = file_name +  name +f'_{small_coeff}_{big_coeff}_{perc}_grouped.pkl'
                    file_name_df = os.path.join(
                        "results_no_water_radii_distr",
                        file_name + name + f'_{small_coeff}_{big_coeff}_{perc}_{counter}_realiz_grouped_{px_th}_px_th.pkl'
                    )
                    grouped.to_pickle(file_name_df)