import pandas as pd
import numpy as np
from scipy import spatial
import os
import sys


def add_tiers(csv_file_src, csv_file_tiers):
    """
        Generate user tiers from csv_file_src, save augmented csv to csv_file
    """
    try:
        print("reading from", csv_file_src)
        
        # Read the CSV file
        data = pd.read_csv(csv_file_src).astype('float32')
        
        print("generating tiers and mat routing scheme")


        # Design Space Tiers ("Users")
        # Tier A is User 2**1
        #       ...
        # Tier E is User 2**5

        # It's log-scaled just to make life easier for the plotting code, as most things are log scale

        data['user'] = [5 for x in range(len(data['id']))]
        print('user filter 1/5')
        data['user'] -= [data['subarrays'][x]==data['subarrays'][0] and data['mats'][x]==data['mats'][0] and data['mat_rows'][x]==data['mat_rows'][0] and data['mat_cols'][x]==data['mat_cols'][0] and data['brvsa'][x]==data['brvsa'][0] and data['ha_layout'][x]==data['ha_layout'][0] and data['ha_double_ldls'][x]==data['ha_double_ldls'][0] and data['subchannels'][x]==data['subchannels'][0] and (data['csl_mdl_shared_layer'][x]==data['csl_mdl_shared_layer'][0]) and data['salp_groups'][x]==data['salp_groups'][0] and data['salp_all'][x]==data['salp_all'][0] and data['ldls_mdls'][x]==data['ldls_mdls'][0] and data['atom_size'][x]==data['atom_size'][0] for x in range(len(data['id']))]
        print('user filter 2/5')
        data['user'] -= [data['mat_rows'][x]==data['mat_rows'][0] and data['mat_cols'][x]==data['mat_cols'][0] and data['brvsa'][x]==data['brvsa'][0] and data['ha_layout'][x]==data['ha_layout'][0] and data['ha_double_ldls'][x]==data['ha_double_ldls'][0] and data['subchannels'][x]==data['subchannels'][0] and (data['csl_mdl_shared_layer'][x]==data['csl_mdl_shared_layer'][0]) and data['ldls_mdls'][x]==data['ldls_mdls'][0] and data['atom_size'][x]==data['atom_size'][0] for x in range(len(data['id']))]
        print('user filter 3/5')
        data['user'] -= [data['mat_rows'][x]==data['mat_rows'][0] and data['mat_cols'][x]==data['mat_cols'][0] and (data['csl_mdl_shared_layer'][x]==data['csl_mdl_shared_layer'][0]) and data['ldls_mdls'][x]==data['ldls_mdls'][0] and data['atom_size'][x]==data['atom_size'][0] for x in range(len(data['id']))]
        print('user filter 4/5')
        data['user'] -= [(data['csl_mdl_shared_layer'][x]==data['csl_mdl_shared_layer'][0]) for x in range(len(data['id']))]
        # exponentiate
        data['user'] = [2**data['user'][x] for x in range(len(data['id']))]
        print(f"\nTier A count: {sum(data['user']<=2**1)-1}")
        print(f"Tier B count: {sum(data['user']<=2**2)-1}") # inclusive of lower tiers
        print(f"Tier C count: {sum(data['user']<=2**3)-1}")
        print(f"Tier D count: {sum(data['user']<=2**4)-1}")
        print(f"Tier E count: {sum(data['user']<=2**5)-1}")
        
        #print("\nNote: if the sweep includes the baseline, the baseline will be counted twice here and below.\nAdjust accordingly.")
        # this is now handled by subtracting 1 from all tiers, removing the duplicate baseline from just the count (not data). 
        # baseline by definition must be in lowest tier
        # also, this does not affect the hull sizes below because it is a duplicate

        # append mat routing scheme
        #data['mat_scheme'] = [2**(1 + data['csl_mdl_shared_layer'][x] + data['mdl_over_mat'][x] + data['csl_mdl_over_mat'][x] + (1-data['csl_mdl_shared_layer'][x])*2) for x in range(len(data['id']))]
        print(f"Saving to {csv_file_tiers}")
        data.to_csv(csv_file_tiers, index=False)

        print(f"appended user tiers and saved to {csv_file_tiers}\n")
        
    except FileNotFoundError:
        print(f"Error: The file '{csv_file_src}' was not found.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")


if __name__ == '__main__':

    """
        USAGE: 
        To generate user tiers for the default sweep ("hbm_sweep_default"), run:
        tier_generator.py

        To generate user tiers for a different sweep, run:
        tier_generator.py CSV_NAME

    """

    DATAVERSION = "hbm_sweep_default"

    try:
        arg = str(sys.argv[1])
        if arg:
            DATAVERSION = arg
            print(f"Using input data version {DATAVERSION}")
    except:
        print(f"Using default data version {DATAVERSION}")

    csv_file_src = f'data/{DATAVERSION}/hbm3_{DATAVERSION}.csv'
    csv_file_tiers = f'data/{DATAVERSION}/hbm3_{DATAVERSION}_user.csv'

    if not os.path.exists(csv_file_tiers):
        print("Adding Tiers...")
        add_tiers(csv_file_src, csv_file_tiers)
    else:
        print("Tiers file already exists. Using that instead of regenerating...")

    data = pd.read_csv(csv_file_tiers)

    sets = [data[data["user"] <= 2**i][['bw_gbytes','capacity_gbytes','total_area_mmmm','worst_latency_ns','metric_e_per_bit_closed']] for i in range(1, 6)]
    
    hulls = [spatial.ConvexHull(sets[i]) for i in range(5)]

    metrics = ['bw_gbytes','capacity_gbytes','total_area_mmmm','worst_latency_ns','metric_e_per_bit_closed']


    print("\nCONVEX HULL DATA (RAW)\n")

    for i in range(5):
        print(i+1)
        print('convex_hull_5d_volume', f"{hulls[i].volume:3e}", sep=', ')
        for j in range(5):
            print(metrics[j], sets[i][metrics[j]].min(), sets[i][metrics[j]].max(), sep=', ')
        print()


    print('user', 'count', 'convex_hull_5d_volume', 'bandwidth_min', 'bandwidth_max', 'capacity_min', 'capacity_max', 'area_min', 'area_max', 'latency_min', 'latency_max', 'energy_min', 'energy_max', sep=',')
    for i in range(5):
        print(i+1, len(sets[i]['bw_gbytes'])-1, hulls[i].volume, sep=',', end=',')
        for j in range(5): 
            print(sets[i][metrics[j]].min(), sets[i][metrics[j]].max(), sep=',', end=',' if j!=4 else '')
        print()