import numpy as np
import pandas as pd
import csv
import sys
from itertools import product
import json
import getopt
import os

from tech import Tech
from hbm import Hbm
import parse


def simulate(dram:Hbm, tech, d):
    """ 
    Simulation iteration. Takes inputs in dram, tech, and d; modifies and outputs d.
    Also return a status code (int), where 0 corresponds to no errors. 
    Return early with nonzero status if a configuration is discarded. 
    """

    ''' Pre-Filters '''
    # discard any 
    
    if d['channels'] % d['ch_per_die'] != 0:
        # discard
        return 0, 1

    if d['ha_double_ldls'] > d['ha_layout']:
        # ha double ldls requires ha layout
        # discard
        return 0, 2
    
    if d['salp_all'] == 1 and d['salp_groups'] != 1:
        # do not run both salp all and salp groups
        # discard
        return 0, 3
    
    if (d['mdl_over_mat'] >> 1) and (not d['mdl_over_mat'] & 1): 
        # not possible
        return 0, 4
    if not d['csl_mdl_shared_layer'] and not d['csl_mdl_over_mat']:
        # require separate layers to use both over the mat
        return 0, 5
    
    min_atom = d['mats'] * d['ldls_mdls'] / (np.power(2.0, d['ha_layout'] - d['ha_double_ldls']) * d['subchannels'])
    if min_atom > d['atom_size']:
        # this DRAM cannot support the given atom size
        # discard
        return 0, 6
    elif min_atom != int(min_atom):
        # invalid layout, your pages are likely splitting MATs
        # discard
        return 0, 7
    elif d['atom_size']/min_atom != int(d['atom_size']/min_atom):
        # this DRAM cannot support the given atom size
        # discard
        return 0, 8

    if d['mats'] < (np.power(2.0, d['ha_layout'] - d['ha_double_ldls']) * d['subchannels']):
        # More ind pages than MATs
        return 0, 9
    
    d['pumps_per_atom'] = d['atom_size'] / min_atom
    if d['pumps_per_atom'] > 8:
        # limit to 8 pumps
        return 0, 10

    ''' Die dimensions '''

    dies_stacked, die_width_um, non_tsv_y, cmd_tsv_y, data_tsv_y, other_tsv_y = dram.calc_stack_dims(tech)
    bank_x, bank_y, cell_area = dram.bank_dims(tech)
    
    die_height_um = non_tsv_y+cmd_tsv_y+data_tsv_y+other_tsv_y
    tsv_area_height_um = cmd_tsv_y+data_tsv_y+other_tsv_y

    max_die_size = tech.max_die_dims_mm * 1000
    if die_width_um >= max_die_size or die_height_um >= max_die_size:
        # discard
        return 0, 11
    
    if dies_stacked > tech.max_stack_dies:
        # discard
        return 0, 12

    d['dies'] = dies_stacked
    d['die_x_mm'] = round(die_width_um/1000,3)
    d['die_y_mm'] = round(die_height_um/1000,3)
    d['die_y_tsv_area_mm'] = round(tsv_area_height_um/1000,3)
    d['total_area_mmmm'] = (d['dies']+1) * d['die_x_mm'] * d['die_y_mm']
    d['bank_x_um'] = round(bank_x, 3)
    d['bank_y_um'] = round(bank_y, 3)


    ''' Capacity, Pages, Atoms '''

    d['capacity_gbytes'] = dram.capacity()
    d['storage_density'] = d['capacity_gbytes']*8 / (d['dies']*d['die_x_mm']*d['die_y_mm'])
    d['density_gbit_mm2_ecc'] = d['capacity_gbytes']*8 * dram.ecc_factor() / (d['dies']*d['die_x_mm']*d['die_y_mm'])
    d['page_size_bytes'] = dram.page_act_size()//8
    d['atoms_per_page'] = dram.atoms_per_page()
    if d['atoms_per_page'] < 1 or d['atoms_per_page'] != int(d['atoms_per_page']):
        # discard
        return 0, 13
    ind_pages = dram.ind_pages()
    d['ind_pages'] = ind_pages
    if ind_pages < d['pages_per_bgbus_mux']:
        # discard
        return 0, 14

    #d['dq_to_core_freq_factor'] = dram.dq_speed_factor()


    ''' Wire counts '''

    l = dram.wire_lengths(tech)
    n = dram.wire_counts()

    d['n_csl'] = n['csl']
    d['n_mdl'] = n['mdl']
    d['n_bgbus_b'] = n['bgbus']
    #d['n_gbus_b'] = n['gbus']
    d['n_tsv'] = n['tsv']
    d['n_dq'] = n['dq']
    d['n_dq_total'] = dram.dq_count()
    


    ''' Pitch Ratios and Datarates '''

    csl_ratio, mdl_ratio = dram.csl_mdl_pitch_ratios(tech) # see the function here for more detailed filter
    if csl_ratio < 0.5 or mdl_ratio < 0.5:
        # discard
        return 0, 15
    #d['csl_pitch_ratio'] = csl_ratio
    #d['mdl_pitch_ratio'] = mdl_ratio
    #d['core_pd_ns'] = dram.core_tck(tech)
    d['core_freq_ghz'] = 1 / dram.core_tck(tech)#d['core_pd_ns']
    d['dq_datarate_gbps'] = d['core_freq_ghz'] * dram.dq_speed_factor()
    d['bw_gbytes'] = dram.bandwidth(tech)
    d['atom_time'] = dram.atom_time(tech) #tBURST
    cmd_e, w = dram.per_cmd_energy (tech)


    ''' Timing and BLSA Margins '''
    d['tcl'] = dram.tcl(tech)
    d['trcd'] = dram.trcd(tech)
    d['trp'] = dram.trp(tech)
    d['trcdwr'] = dram.trcdwr(tech)
    d['tras'] = dram.tras(tech)
    d['trc'] = dram.trc(tech)
    d['trrds'] = dram.trrds(tech)
    d['trrdl'] = dram.trrdl(tech)
    d['tfaw'] = dram.tfaw(tech)
    d['trtp'] = dram.trtp(tech)
    d['twr'] = dram.twr(tech)
    d['peri_tck'] = dram.peri_tck(tech)
    d['tccdl'] = dram.tccdl(tech)
    d['tccds'] = dram.tccds(tech)
    d['worst_latency_ns'] = d['tcl'] + d['trcd'] + d['trp']
    d['blsa_deltav'] = dram.blsa_deltav(tech)

    ''' Cell Efficiency '''

    d['cell_eff'] = dram.cell_efficiency(tech)
    #d['cell_eff_mat'] = dram.cell_efficiency_mat(tech)


    ''' Energy per Component Usage '''

    d['e_cmd_pre_pj'] = cmd_e['pre']
    d['e_cmd_act_pj'] = cmd_e['act']
    d['e_cmd_rd_pj'] = cmd_e['rd']
    #d['e_cmd_pre_pj_heat'] = cmd_e['heat-pre']
    #d['e_cmd_act_pj_heat'] = cmd_e['heat-act']

    d['e_set_base_row'] = w['row-base']
    d['e_set_tsv_row'] = w['row-tsv']
    d['e_set_row'] = w['row']
    d['e_set_mwl'] = w['mwl']
    d['e_set_lwl'] = w['lwl']
    d['e_set_bl_act'] = w['bl-act']
    d['e_set_bl_pre'] = w['bl-pre']
    #d['e_set_bl_act'] = w['bl-act']
    #d['e_set_bl_pre'] = w['bl-pre']

    d['e_set_base_col'] = w['col-base']
    d['e_set_tsv_col'] = w['col-tsv']
    d['e_set_col'] = w['col']
    d['e_set_csl'] = w['csl']
    d['e_set_ldl'] = w['ldl']
    d['e_set_mdl'] = w['mdl']
    d['e_set_bus'] = w['bgbus+gbus']
    d['e_set_tsv_data'] = w['tsv']
    d['e_set_base_data'] = w['base']
    d['e_set_dq'] = w['dq']


    ''' metrics '''

    # Bandwidth-Capacity ratio
    d['metric_bw_per_cap'] = d['bw_gbytes'] / d['capacity_gbytes']
    # Energy per bit when reading whole pages sequentially
    d['metric_e_per_bit_seq'] = (d['e_cmd_pre_pj'] + d['e_cmd_act_pj'] + d['atoms_per_page']*d['e_cmd_rd_pj']) / (8*d['page_size_bytes'])
    # Energy per bit for a single atom access (closed row policy)
    d['metric_e_per_bit_closed'] = (d['e_cmd_pre_pj'] + d['e_cmd_act_pj'] + d['e_cmd_rd_pj']) / d['atom_size'] # bits
    # Worst-case power
    d['worst_power_w'] = d['metric_e_per_bit_closed'] * d['bw_gbytes'] * 8e-3 # unit conversion
    # EDP
    d['edp'] = d['metric_e_per_bit_closed'] * d['worst_latency_ns']

    return d, 0



def main(hbm_json_name, tech_json_name, output_csv_name):
    """
    Main simulation loop. 
    Extracts simulation parameters from JSON files. 
    First runs the baseline configuration.
    Then loops through the sweep configurations. 
    """

    ''' Setup '''

    print(f'Memory file: {hbm_json_name}')
    print(f'Tech file: {tech_json_name}')

    # open save file
    csv_name = output_csv_name

    assert (not os.path.exists(csv_name)), f"output CSV file already exists. To re-run, delete {csv_name}"

    with open(csv_name, 'w', newline='') as csvfile:
        print('Overwriting to',csv_name)

        data = []
        count = 0
        discarded = 0


        ''' Tech Node '''
        # load tech parameters
        t = parse.tech(tech_json_name)

        ''' baseline '''
        # load baseline and sweep parameters
        b, s = parse.mem_baseline_and_sweep(hbm_json_name)

        # apply timing baseline overrides (product-specific, overrides process baseline)
        timing_baseline_path = b.pop('_timing_baseline', None)
        if timing_baseline_path:
            t = parse.timing_baseline(timing_baseline_path, t)

        # instantiate tech object
        tech = Tech(f=t['f'], pitch_wl=t['pitch_wl'], pitch_bl=t['pitch_bl'], c_scale_conf=t['c_scale_conf'], c_blsa_scale_conf=t['c_blsa_scale_conf'], tsv_c_pitch_scale_conf=t['tsv_c_pitch_scale_conf'], logic_scale_conf=t['logic_scale_conf'], coldec_scale_conf=t['coldec_scale_conf'], rowdec_scale_conf=t['rowdec_scale_conf'], swd_scale_conf=t['swd_scale_conf'], blsa_scale_conf=t['blsa_scale_conf'], tsv_pitch=t['tsv_pitch'], tsv_koz=t['tsv_koz'], tsv_height=t['tsv_height'], ubump_pitch=t['ubump_pitch'], max_die_dims_mm=t['max_die_dims_mm'], _f=t['_f'], _tsv_pitch=t['_tsv_pitch'], _tsv_koz=t['_tsv_koz'], _tsv_height=t['_tsv_height'], _c_tsv=t['_c_tsv'], _c_load=t['_c_load'], _r_load=t['_r_load'], _c_bus=t['_c_bus'], _c_ca=t['_c_ca'], _c_mwl=t['_c_mwl'], _c_lwl=t['_c_lwl'], _c_bl_per_cell=t['_c_bl_per_cell'], _c_cell=t['_c_cell'], _c_blsa=t['_c_blsa'], _c_csl=t['_c_csl'], _c_ldl=t['_c_ldl'], _c_mdl=t['_c_mdl'], c_dq=t['c_dq'], c_ca_ra_pin=t['c_ca_ra_pin'], _c_within_layer=t['_c_within_layer'], _c_within_layer_top=t['_c_within_layer_top'], _c_within_layer_sparse=t['_c_within_layer_sparse'], _c_within_layer_top_sparse=t['_c_within_layer_top_sparse'], _trcd=t['_trcd'], _trcd_signal=t['_trcd_signal'], _mat_rows_ref=t['_mat_rows_ref'], _trcd_brvsa=t['_trcd_brvsa'], _brvsa_brv_deltav_boost=t['_brvsa_brv_deltav_boost'], _brvsa_cs_proportion=t['_brvsa_cs_proportion'], _brvsa_height_ratio=t['_brvsa_height_ratio'], cell_leak=t['cell_leak'], _min_deltav=t['_min_deltav'], _mdl_over_mat_height_ratio=t['_mdl_over_mat_height_ratio'], vdd=t['vdd'], vpp=t['vpp'], vpp_int=t['vpp_int'], vddql_int=t['vddql_int'], vcore_int=t['vcore_int'], vpp_eff=t['vpp_eff'], _coldec_height=t['_coldec_height'], _rowdec_width=t['_rowdec_width'], _blsa_height=t['_blsa_height'], _swd_width=t['_swd_width'], _trcdwr_blsa_fraction=t.get('_trcdwr_blsa_fraction', 0.55), _tras_restore=t.get('_tras_restore', 13.1), _twr_restore=t.get('_twr_restore', 18.1), _trrds=t.get('_trrds', 2.5), _trrdl=t.get('_trrdl', 2.5))

        # instantiate and simulate
        dram = Hbm(sids=b['sids'], channels=b['channels'], ch_per_die=b['ch_per_die'], pch=b['pch'], horiz_bg=b['horiz_bg'], vert_bg=b['vert_bg'], banks=b['banks'], subarrays=b['subarrays'], mat_rows=b['mat_rows'], mats=b['mats'], mat_cols=b['mat_cols'], brv_sa=b['brvsa'], ldls_mdls=b['ldls_mdls'], mdl_over_mat=b['mdl_over_mat'] & 1, mdl_csl_over_mat=b['mdl_over_mat'] >> 1, csl_mdl_shared_layer=b.get('csl_mdl_shared_layer', 0), ha_layout=b['ha_layout'], ha_double_ldls=b['ha_double_ldls'], salp_groups=b['salp_groups'], salp_all=b['salp_all'], pages_per_bgbus_mux=b['pages_per_bgbus_mux'], mdl_bgbus_sd=b['mdl_bgbus_sd'], bgbuses_per_gbus=b['bgbuses_per_gbus'], bgbus_gbus_sd=b['bgbus_gbus_sd'], gbus_tsv_sd=b['gbus_tsv_sd'], tsv_dq_sd=b['tsv_dq_sd'], subchannels=b['subchannels'], atom_size=b['atom_size'], _mat_rows=b['_mat_rows'], _mat_cols=b['_mat_cols'], _subarrays=b['_subarrays'], _ldls_mdls=b['_ldls_mdls'], _mats=b['_mats'], _subchannels=b['_subchannels'], _tck=b['_tck'], _tcl=b['_tcl'], _channels=b['_channels'], _ch_per_die=b['_ch_per_die'], _sids=b['_sids'], _vert_bg=b['_vert_bg'], _n_bgbus=-1, _pages_per_bgbus_mux=b['_pages_per_bgbus_mux'], _mdl_bgbus_sd=b['_mdl_bgbus_sd'], _bgbus_gbus_sd=b['_bgbus_gbus_sd'], _gbus_tsv_sd=b['_gbus_tsv_sd'], _die_y=-1)
        dreturn, status = simulate(dram, tech, b.copy())

        if status != 0:
            print(f"baseline not valid with error {status}")
            assert status == 0  # baseline should be a valid configuration
        data.append(dreturn)
        print('baseline established')
        
        # remove baseline keys for data saving purposes
        del data[0]['isolation_rows_overhead']
        del data[0]['isolation_cols_overhead']
        del data[0]['_mat_rows']
        del data[0]['_mat_cols']
        del data[0]['_subarrays']
        del data[0]['_ldls_mdls']
        del data[0]['_mats']
        del data[0]['_subchannels']
        del data[0]['_tck']
        del data[0]['_tcl']
        del data[0]['_ch_per_die']
        del data[0]['_channels']
        del data[0]['_sids']
        del data[0]['_vert_bg']
        del data[0]['_pages_per_bgbus_mux']
        del data[0]['_mdl_bgbus_sd']
        del data[0]['_bgbus_gbus_sd']
        del data[0]['_gbus_tsv_sd']

        data[0]['csl_mdl_over_mat'] = data[0]['mdl_over_mat'] >> 1
        data[0]['mdl_over_mat'] = data[0]['mdl_over_mat'] & 1
        

        ''' Sweep '''
        print('Preparing sweep parameters...')
        # load sweep parameters
        sids_list = s['sids_list']
        channels_list = s['channels_list']
        ch_per_die_list = s['ch_per_die_list']
        pch_list = s['pch_list']
        horiz_bg_list = s['horiz_bg_list']
        vert_bg_list = s['vert_bg_list']
        banks_list = s['banks_list']
        subarrays_list = s['subarrays_list']
        mat_rows_list = s['mat_rows_list']
        mats_list = s['mats_list']
        mat_cols_list = s['mat_cols_list']
        brvsa_list = s['brvsa_list']
        ldls_mdls_list = s['ldls_mdls_list']
        mdl_over_mat_list = s['mdl_over_mat_list']
        csl_mdl_shared_layer_list = s['csl_mdl_shared_layer_list']
        ha_layout_list = s['ha_layout_list']
        ha_double_ldls_list = s['ha_double_ldls_list']
        salp_groups_list = s['salp_groups_list']
        salp_all_list = s['salp_all_list']
        pages_per_bgbus_mux_list = s['pages_per_bgbus_mux_list']
        mdl_bgbus_sd_list = s['mdl_bgbus_sd_list']
        bgbuses_per_gbus_list = s['bgbuses_per_gbus_list']
        bgbus_gbus_sd_list = s['bgbus_gbus_sd_list']
        gbus_tsv_sd_list = s['gbus_tsv_sd_list']
        tsv_dq_sd_list = s['tsv_dq_sd_list']
        subchannels_list = s['subchannels_list']
        atom_size_list = s['atom_size_list']

        print('Building sweep list...')
        list_list = [sids_list, channels_list, ch_per_die_list, pch_list, horiz_bg_list, vert_bg_list, banks_list, subarrays_list, mat_rows_list, mats_list, mat_cols_list, brvsa_list, ldls_mdls_list, mdl_over_mat_list, csl_mdl_shared_layer_list, ha_layout_list, ha_double_ldls_list, subchannels_list, salp_groups_list, salp_all_list, pages_per_bgbus_mux_list, mdl_bgbus_sd_list, bgbuses_per_gbus_list, bgbus_gbus_sd_list, gbus_tsv_sd_list, tsv_dq_sd_list, atom_size_list]

        n_datapoints = np.prod([len(l) for l in list_list])
        print(f'Sweep is {n_datapoints} datapoints\nIncludes future discards\nDon\'t worry the final dataset is not nearly as large as this')

        print('Generating list product...')
        full_list = product(*list_list)

        print(f'Begin sweep of {n_datapoints} datapoints')

        _n_bgbus = dram.n_bgbus()
        dies_stacked, die_width, non_tsv_y, cmd_tsv_y, data_tsv_y, other_tsv_y = dram.calc_stack_dims(tech)
        _die_y = non_tsv_y + cmd_tsv_y + data_tsv_y + other_tsv_y
        _bank_x = dram._bank_x_calc(tech)
        _bank_y = dram._bank_y_calc(tech)
        _tcl = dram._tcl

        # Sweep
        for item in full_list:
            
            # counter for quality of life
            count += 1
            if count % 10000 == 0:
                print(f"{count:,}", "of", f"{n_datapoints:,}")

            # instantiate dram
            sids, channels, ch_per_die, pch, horiz_bg, vert_bg, banks, subarrays, mat_rows, mats, mat_cols, brvsa, ldls_mdls, mdl_over_mat, csl_mdl_shared_layer, ha_layout, ha_double_ldls, subchannels, salp_groups, salp_all, pages_per_bgbus_mux, mdl_bgbus_sd, bgbuses_per_gbus, bgbus_gbus_sd, gbus_tsv_sd, tsv_dq_sd, atom_size = item
            brvsa = 0 if brvsa=="blsa" else 1
            dram = Hbm(sids=sids, channels=channels, ch_per_die=ch_per_die, pch=pch, horiz_bg=horiz_bg, vert_bg=vert_bg, banks=banks, subarrays=subarrays, mat_rows=mat_rows, mats=mats, mat_cols=mat_cols, brv_sa=brvsa, ldls_mdls=ldls_mdls, mdl_over_mat=mdl_over_mat & 1, mdl_csl_over_mat=mdl_over_mat >> 1, csl_mdl_shared_layer=csl_mdl_shared_layer, ha_layout=ha_layout, ha_double_ldls=ha_double_ldls, salp_groups=salp_groups, salp_all=salp_all, pages_per_bgbus_mux=pages_per_bgbus_mux, mdl_bgbus_sd=mdl_bgbus_sd, bgbuses_per_gbus=bgbuses_per_gbus, bgbus_gbus_sd=bgbus_gbus_sd, gbus_tsv_sd=gbus_tsv_sd, tsv_dq_sd=tsv_dq_sd, subchannels=subchannels, atom_size=atom_size, _mat_rows=b['_mat_rows'], _mat_cols=b['_mat_cols'], _subarrays=b['_subarrays'], _ldls_mdls=b['_ldls_mdls'], _mats=b['_mats'], _subchannels=b['_subchannels'], _tck=b['_tck'], _tcl=_tcl, _channels=b['_channels'], _ch_per_die=b['_ch_per_die'], _sids=b['_sids'], _vert_bg=b['_vert_bg'], _n_bgbus=_n_bgbus, _pages_per_bgbus_mux=b['_pages_per_bgbus_mux'], _mdl_bgbus_sd=b['_mdl_bgbus_sd'], _bgbus_gbus_sd=b['_bgbus_gbus_sd'], _gbus_tsv_sd=b['_gbus_tsv_sd'], _die_y=_die_y, _bank_y=_bank_y, _bank_x=_bank_x)

            # prepare inputs
            d = {}
            d['id'] = count
            d['sids'] = sids
            d['channels'] = channels
            d['ch_per_die'] = ch_per_die
            d['pch'] = pch
            d['horiz_bg'] = horiz_bg
            d['vert_bg'] = vert_bg
            d['banks'] = banks
            d['subarrays'] = subarrays
            d['mat_rows'] = mat_rows
            d['mats'] = mats
            d['mat_cols'] = mat_cols
            d['brvsa'] = brvsa
            d['ldls_mdls'] = ldls_mdls
            d['mdl_over_mat'] = mdl_over_mat & 1
            d['csl_mdl_over_mat'] = mdl_over_mat >> 1
            d['csl_mdl_shared_layer'] = csl_mdl_shared_layer
            d['ha_layout'] = ha_layout
            d['ha_double_ldls'] = ha_double_ldls
            d['salp_groups'] = salp_groups
            d['salp_all'] = salp_all
            d['pages_per_bgbus_mux'] = pages_per_bgbus_mux
            d['mdl_bgbus_sd'] = mdl_bgbus_sd
            d['bgbuses_per_gbus'] = bgbuses_per_gbus
            d['bgbus_gbus_sd'] = bgbus_gbus_sd
            d['gbus_tsv_sd'] = gbus_tsv_sd
            d['tsv_dq_sd'] = tsv_dq_sd
            d['subchannels'] = subchannels
            d['atom_size'] = atom_size
            
            # simulate
            dreturn, status = simulate(dram, tech, d)

            # add data
            if status != 0:
                discarded += 1
                #print(status)
            else:
                data.append(dreturn)
                if d['id'] == 1:
                    # checking this early
                    # this might save you a lot of time
                    assert(data[0].keys() == data[1].keys()) # baseline and sweep dict keys must match to save properly!


        ''' Save and Exit '''




        # some stats
        print("Sweep Datapoints Searched:", count)
        print("Kept:", count - discarded)
        print("Discarded:", discarded)
        print()
        print("Saving...")
        print("Do not exit yet...")
        print()

        df = pd.DataFrame(data)
        df = df.astype(np.float32)
        df.to_csv(csvfile, index=False)

        print("Saved to",csv_name)
        print("Be sure to change label to not overwrite!")
        print()

        return 0


if __name__ == '__main__':

    # defaults
    default_label =  "hbm_sweep_default"
    hbm_json_name = f"configs/mem/sweep/{default_label}.json"
    tech_json_name = "configs/tech/scaled/16nm_scaled.json" #tech_json_name = "configs/tech/scaled/17nm_hbm2e.json"
    output_label = f"{default_label}"
    output_csv_name = f"data/{output_label}/hbm3_{output_label}.csv"
    hbmtech_default = True
    output_default = True

    args = sys.argv[1:]
    options = "hm:t:o:"
    long_options = ["help", "mem=", "tech=", "out="]

    try:
        arguments, values = getopt.getopt(args, options, long_options)
        for currentArg, currentVal in arguments:
            if currentArg in ("-h", "--Help"):
                print("Usage:")
                print("python3 dreamram.py [-m MEMORY_CONFIG] [-t TECH_CONFIG] [-o OUTPUT_LABEL]")
                print("")
                print("Default output label is \"default\"")
                print("i.e., the default output file is data/default/hbm3_default.csv")
                print("Data is overwritten without checking existance.\nBe sure you have the unique output label if you do not want to overwrite")
                print("")
                exit()
            elif currentArg in ("-m", "--mem"):
                print("Set memory file to:", currentVal)
                hbm_json_name = currentVal
                hbmtech_default = False
            elif currentArg in ("-t", "--tech"):
                print("Set tech file to:", currentVal)
                tech_json_name = currentVal
                hbmtech_default = False
            elif currentArg in ("-o", "--out"):
                print("Set output label to:", currentVal)
                output_csv_name = f"data/{currentVal}/hbm3_{currentVal}.csv"
                output_default = False
                os.makedirs(f'data/{currentVal}', exist_ok=True)

    except getopt.error as err:
        print(str(err))

    if hbmtech_default:
        print("No memory or tech files specified, using defaults.")

    if output_default:
        print("No output file specified, writing to data/default/hbm3_default.csv") #TODO: stop using hbm3 as data file name prefix, here and in plotting
        os.makedirs(f'data/{output_label}', exist_ok=True)

    print()

    sys.exit(main(hbm_json_name, tech_json_name, output_csv_name))