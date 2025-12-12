import numpy as np
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
    Simulation iteration. Take inputs in dict d, add outputs to d and return d. 
    Also return a status code (int), where 0 corresponds to no errors. 
    Return early with nonzero status if a configuration is discarded. 
    """

    ''' Pre-Filters '''

    if d['channels'] % d['ch_per_die'] != 0:
        # discard
        return 0, 9
    
    if d['ha_double_ldls'] > d['ha_layout']:
        # ha double ldls requires ha layout
        # discard
        return 0, 1
    
    if d['salp_all'] == 1 and d['salp_groups'] != 1:
        # do not run both salp all and salp groups
        # discard
        return 0, 2
    
    if d['mdl_over_mat'] and d['mat_cols'] / (2*d['ldls_mdls']) < tech.pitch_mdl_to_bl_min:
        # MDLs too close
        # discard
        return 0, 3
    
    if (not d['mdl_over_mat']) and d['ldls_mdls'] < tech.pitch_mdl_to_bl_min:
        # CSLs too close
        # discard
        return 0, 4
    
    min_atom = d['mats'] * d['ldls_mdls'] / (np.power(2.0, d['ha_layout'] - d['ha_double_ldls']) * d['subchannels'])
    if min_atom > d['atom_size']:
        # this DRAM cannot support the given atom size
        # discard
        return 0, 5
    elif min_atom != int(min_atom):
        # invalid layout, your pages are likely splitting MATs
        # discard
        return 0, 55
    elif d['atom_size']/min_atom != int(d['atom_size']/min_atom):
        # this DRAM cannot support the given atom size
        # discard
        return 0, 555
    
    if d['mats'] < (np.power(2.0, d['ha_layout'] - d['ha_double_ldls']) * d['subchannels']):
        # More ind pages than MATs
        return 0, 6
    
    d['pumps_per_atom'] = d['atom_size'] / min_atom
    if d['pumps_per_atom'] > 8:
        # limit to 8 pumps
        return 0, 5555

    d['gbuses_out'] = dram.gbuses()
    if d['gbuses_out'] < 2: # 1 for HBM2E
        # too much muxing
        return 0, 8
    if d['gbuses_out'] > 8:
        # unrealistic
        return 0, 88

    d['virtual_banks_per_bg'] = d['pages_per_bgbus_mux']
    ind_pages = dram.ind_pages()
    d['virtual_bgs_per_pch'] = ind_pages * d['banks'] * d['vert_bg'] * d['horiz_bg'] / d['pages_per_bgbus_mux']
    d['virtual_gbuses_per_pch'] = d['gbuses_out']

    ''' Die dimensions '''

    dies_stacked, die_width_um, non_tsv_y, cmd_tsv_y, data_tsv_y, other_tsv_y = dram.calc_stack_dims(tech)
    bank_x, bank_y, cell_area = dram.bank_dims(tech)
    
    die_height_um = non_tsv_y+cmd_tsv_y+data_tsv_y+other_tsv_y
    tsv_area_height_um = cmd_tsv_y+data_tsv_y+other_tsv_y

    max_die_size = tech.max_die_dims_mm * 1000
    if die_width_um >= max_die_size or die_height_um >= max_die_size:
        # discard
        return 0, 7
    
    if dies_stacked > tech.max_stack_dies:
        # discard
        return 0, 8

    d['dies'] = dies_stacked
    d['die_x_mm'] = round(die_width_um/1000,3)
    d['die_y_mm'] = round(die_height_um/1000,3)
    d['die_y_tsv_area_mm'] = round(tsv_area_height_um/1000,3)
    d['total_area_mmmm'] = (d['dies']+1) * d['die_x_mm'] * d['die_y_mm']
    d['bank_x_um'] = round(bank_x, 3)
    d['bank_y_um'] = round(bank_y, 3)


    ''' Capacity, Pages, Atoms '''

    d['capacity_gbytes'] = dram.capacity()
    d['density_gbit_mm2_ecc'] = d['capacity_gbytes']*8 * dram.ecc_factor() / (d['dies']*d['die_x_mm']*d['die_y_mm'])
    d['page_size_bytes'] = dram.page_act_size()//8
    d['atoms_per_page'] = dram.atoms_per_page()
    if d['atoms_per_page'] < 1 or d['atoms_per_page'] != int(d['atoms_per_page']):
        # discard
        return 0, 7
    d['ind_pages'] = ind_pages

    d['mdls_per_atom'] = dram.mdl_width_per_page()
    #d['bgbus_width_per_atom'] = dram.bgbus_width()
    #d['gbus_width_per_atom'] = dram.gbus_width()
    d['tsv_to_core_freq_factor'] = dram.tsv_speed_factor()
    d['dq_to_core_freq_factor'] = dram.dq_speed_factor()


    ''' Wire counts '''

    l = dram.wire_lengths(tech)
    n = dram.wire_counts()

    #d['row_cmd_b'] = n['row']
    #d['col_cmd_b'] = n['col']
    d['n_bl'] = n['bl']
    d['n_csl'] = n['csl']
    #d['n_ldl'] = n['ldl']
    d['n_mdl'] = n['mdl']
    d['n_bgbus_b'] = n['bgbus']
    d['n_gbus_b'] = n['gbus']
    d['n_tsv'] = n['tsv']
    d['n_dq'] = n['dq']
    d['n_dq_total'] = dram.dq_count()
    #d['n_bgbus_half_die'] = dram.channels * dram.pch * dram.horiz_bg * dram.vert_bg/2 * n['bgbus']
    d['bgbus_max_pitch_um'] = d['bank_x_um'] / (n['bgbus'] * dram.md_ecc_factor() * dram.ind_pages() / (dram.pages_per_bgbus_mux*dram.mdl_bgbus_sd) * dram.vert_bg/2 * np.pow(dram.dbi_factor, dram.dbi_for_bgbus))



    ''' Pitch Ratios and Datarates '''

    csl_ratio, mdl_ratio = dram.csl_mdl_pitch_ratios(tech)
    d['csl_pitch_ratio'] = csl_ratio
    d['mdl_pitch_ratio'] = mdl_ratio
    d['core_pd_ns'] = dram.core_tck(tech)
    d['core_freq_ghz'] = 1 / d['core_pd_ns']
    #d['dq_datarate_ns'] = d['core_pd_ns'] / dram.dq_speed_factor() # datarate includes DDR
    d['dq_datarate_gbps'] = d['core_freq_ghz'] * dram.dq_speed_factor()
    d['bw_gbytes'] = dram.bandwidth(tech)
    d['atom_time'] = dram.atom_time(tech)
    cmd_e, w = dram.per_cmd_energy (tech)


    ''' Timing and BLSA Margins '''
    d['tcl'] = dram.tcl(tech)
    d['trcd'] = dram.trcd(tech)
    d['trp'] = dram.trp(tech)
    d['worst_latency_ns'] = d['tcl'] + d['trcd'] + d['trp']
    d['blsa_deltav'] = dram.blsa_deltav(tech)


    ''' Cell Efficiency '''

    d['cell_eff'] = dram.cell_efficiency(tech)
    d['cell_eff_mat'] = dram.cell_efficiency_mat(tech)


    ''' Energy per Component Usage '''

    d['e_cmd_pre_pj'] = cmd_e['pre']
    d['e_cmd_act_pj'] = cmd_e['act']
    d['e_cmd_rd_pj'] = cmd_e['rd']
    d['e_cmd_pre_pj_heat'] = cmd_e['heat-pre']
    d['e_cmd_act_pj_heat'] = cmd_e['heat-act']

    d['e_set_base_row'] = w['row-base']
    d['e_set_tsv_row'] = w['row-tsv']
    d['e_set_row'] = w['row']
    d['e_set_mwl'] = w['mwl']
    d['e_set_lwl'] = w['lwl']
    d['e_set_bl_act'] = w['bl-act']
    d['e_set_bl_pre'] = w['bl-pre']
    d['e_set_bl_act'] = w['bl-act']
    d['e_set_bl_pre'] = w['bl-pre']

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
    d['metric_e_per_bit_closed'] = (d['e_cmd_pre_pj'] + d['e_cmd_act_pj'] + d['e_cmd_rd_pj']) / d['atom_size']
    # Worst-case power
    d['worst_power_w'] = d['metric_e_per_bit_closed'] * d['bw_gbytes'] * 8e-3 # units

    return d, 0



def main(hbm_json_name, tech_json_name, output_csv_name):
    """
    Main simulation loop. Extract simulation parameters from JSON files. 
    First run the baseline configuration, then loop through the sweep configurations. 
    """

    ''' Setup '''

    print(f'Memory file: {hbm_json_name}')
    print(f'Tech file: {tech_json_name}')

    # open save file
    csv_name = output_csv_name

    with open(csv_name, 'w', newline='') as csvfile:
        print('Overwriting to',csv_name)

        data = []
        count = 0
        discarded = 0


        ''' Tech Node '''
        # load tech parameters
        t = parse.tech(tech_json_name)
        # instantiate tech object
        tech = Tech(f=t['f'], pitch_wl=t['pitch_wl'], pitch_bl=t['pitch_bl'], pitch_ldl_to_wl_min=t['pitch_ldl_to_wl_min'], pitch_mdl_to_bl_min=t['pitch_mdl_to_bl_min'], c_scale_conf=t['c_scale_conf'], c_blsa_scale_conf=t['c_blsa_scale_conf'], tsv_c_pitch_scale_conf=t['tsv_c_pitch_scale_conf'], logic_scale_conf=t['logic_scale_conf'], coldec_scale_conf=t['coldec_scale_conf'], rowdec_scale_conf=t['rowdec_scale_conf'], swd_scale_conf=t['swd_scale_conf'], blsa_scale_conf=t['blsa_scale_conf'], tsv_pitch=t['tsv_pitch'], tsv_koz=t['tsv_koz'], tsv_height=t['tsv_height'], max_die_dims_mm=t['max_die_dims_mm'], _f=t['_f'], _tsv_pitch=t['_tsv_pitch'], _tsv_koz=t['_tsv_koz'], _tsv_height=t['_tsv_height'], _c_tsv=t['_c_tsv'], _c_load=t['_c_load'], _r_load=t['_r_load'], _c_bus=t['_c_bus'], _c_ca=t['_c_ca'], _c_mwl=t['_c_mwl'], _c_lwl=t['_c_lwl'], _c_bl_per_cell=t['_c_bl_per_cell'], _c_cell=t['_c_cell'], _c_blsa=t['_c_blsa'], _c_csl=t['_c_csl'], _c_ldl=t['_c_ldl'], _c_mdl=t['_c_mdl'], c_dq=t['c_dq'], _c_within_layer=t['_c_within_layer'], _c_within_layer_top=t['_c_within_layer_top'], _c_within_layer_sparse=t['_c_within_layer_sparse'], _c_within_layer_top_sparse=t['_c_within_layer_top_sparse'], _trcd=t['_trcd'], _trcd_signal=t['_trcd_signal'], _mat_rows_ref=t['_mat_rows_ref'], _trcd_brvsa=t['_trcd_brvsa'], _brvsa_brv_deltav_boost=t['_brvsa_brv_deltav_boost'], _brvsa_cs_proportion=t['_brvsa_cs_proportion'], _brvsa_height_ratio=t['_brvsa_height_ratio'], cell_leak=t['cell_leak'], _min_deltav=t['_min_deltav'], _mdl_over_mat_height_ratio=t['_mdl_over_mat_height_ratio'], vdd=t['vdd'], vpp=t['vpp'], vpp_int=t['vpp_int'], vddql_int=t['vddql_int'], vcore_int=t['vcore_int'], vpp_eff=t['vpp_eff'], _coldec_height=t['_coldec_height'], _rowdec_width=t['_rowdec_width'], _blsa_height=t['_blsa_height'], _swd_width=t['_swd_width'])


        ''' baseline '''
        # load baseline and sweep parameters
        b, s = parse.mem_baseline_and_sweep(hbm_json_name)

        # instantiate and simulate
        dram = Hbm(ranks=b['ranks'], channels=b['channels'], ch_per_die=b['ch_per_die'], pch=b['pch'], horiz_bg=b['horiz_bg'], vert_bg=b['vert_bg'], banks=b['banks'], subarrays=b['subarrays'], mat_rows=b['mat_rows'], mats=b['mats'], mat_cols=b['mat_cols'], brv_sa=b['brvsa'], ldls_mdls=b['ldls_mdls'], mdl_over_mat=b['mdl_over_mat'], ha_layout=b['ha_layout'], ha_double_ldls=b['ha_double_ldls'], salp_groups=b['salp_groups'], salp_all=b['salp_all'], pages_per_bgbus_mux=b['pages_per_bgbus_mux'], mdl_bgbus_sd=b['mdl_bgbus_sd'], bgbuses_per_gbus=b['bgbuses_per_gbus'], bgbus_gbus_sd=b['bgbus_gbus_sd'], gbuses_out=b['gbuses_out'], gbus_tsv_sd=b['gbus_tsv_sd'], tsv_dq_sd=b['tsv_dq_sd'], subchannels=b['subchannels'], atom_size=b['atom_size'], _mat_rows=b['_mat_rows'], _mat_cols=b['_mat_cols'], _subarrays=b['_subarrays'], _ldls_mdls=b['_ldls_mdls'], _mats=b['_mats'], _subchannels=b['_subchannels'], _tck=b['_tck'], _tcl=b['_tcl'], _channels=b['_channels'], _ch_per_die=b['_ch_per_die'], _ranks=b['_ranks'])
        dreturn, status = simulate(dram, tech, b.copy())

        # add data

        assert status == 0, f"baseline not valid with error {print(status)}" # baseline should be a valid configuration
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
        del data[0]['_ranks']


        ''' Sweep '''
        print('Preparing sweep parameters...')
        # load sweep parameters
        ranks_list = s['ranks_list']
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
        ha_layout_list = s['ha_layout_list']
        ha_double_ldls_list = s['ha_double_ldls_list']
        salp_groups_list = s['salp_groups_list']
        salp_all_list = s['salp_all_list']
        pages_per_bgbus_mux_list = s['pages_per_bgbus_mux_list']
        mdl_bgbus_sd_list = s['mdl_bgbus_sd_list']
        bgbuses_per_gbus_list = s['bgbuses_per_gbus_list']
        bgbus_gbus_sd_list = s['bgbus_gbus_sd_list']
        gbuses_out_list = s['gbuses_out_list']
        gbus_tsv_sd_list = s['gbus_tsv_sd_list']
        tsv_dq_sd_list = s['tsv_dq_sd_list']
        subchannels_list = s['subchannels_list']
        atom_size_list = s['atom_size_list']

        print('Building sweep list...')
        list_list = [ranks_list, channels_list, ch_per_die_list, pch_list, horiz_bg_list, vert_bg_list, banks_list, subarrays_list, mat_rows_list, mats_list, mat_cols_list, brvsa_list, ldls_mdls_list, mdl_over_mat_list, ha_layout_list, ha_double_ldls_list, subchannels_list, salp_groups_list, salp_all_list, pages_per_bgbus_mux_list, mdl_bgbus_sd_list, bgbuses_per_gbus_list, bgbus_gbus_sd_list, gbuses_out_list, gbus_tsv_sd_list, tsv_dq_sd_list, atom_size_list]

        n_datapoints = np.prod([len(l) for l in list_list])
        print(f'Sweep is {n_datapoints} datapoints\nIncluds future discards')

        print('Generating list product...')
        full_list = product(*list_list)

        print(f'Begin sweep of {n_datapoints} datapoints')

        # Sweep
        for item in full_list:
            
            # quality of life
            count += 1
            if count % 10000 == 0:
                print(f"{count:,}", "of", f"{n_datapoints:,}")

            # instantiate dram
            ranks, channels, ch_per_die, pch, horiz_bg, vert_bg, banks, subarrays, mat_rows, mats, mat_cols, brvsa, ldls_mdls, mdl_over_mat, ha_layout, ha_double_ldls, subchannels, salp_groups, salp_all, pages_per_bgbus_mux, mdl_bgbus_sd, bgbuses_per_gbus, bgbus_gbus_sd, gbuses_out, gbus_tsv_sd, tsv_dq_sd, atom_size = item
            brvsa = 0 if brvsa=="blsa" else 1
            dram = Hbm(ranks=ranks, channels=channels, ch_per_die=ch_per_die, pch=pch, horiz_bg=horiz_bg, vert_bg=vert_bg, banks=banks, subarrays=subarrays, mat_rows=mat_rows, mats=mats, mat_cols=mat_cols, brv_sa=brvsa, ldls_mdls=ldls_mdls, mdl_over_mat=mdl_over_mat, ha_layout=ha_layout, ha_double_ldls=ha_double_ldls, salp_groups=salp_groups, salp_all=salp_all, pages_per_bgbus_mux=pages_per_bgbus_mux, mdl_bgbus_sd=mdl_bgbus_sd, bgbuses_per_gbus=bgbuses_per_gbus, bgbus_gbus_sd=bgbus_gbus_sd, gbuses_out=gbuses_out, gbus_tsv_sd=gbus_tsv_sd, tsv_dq_sd=tsv_dq_sd, subchannels=subchannels, atom_size=atom_size, _mat_rows=b['_mat_rows'], _mat_cols=b['_mat_cols'], _subarrays=b['_subarrays'], _ldls_mdls=b['_ldls_mdls'], _mats=b['_mats'], _subchannels=b['_subchannels'], _tck=b['_tck'], _tcl=b['_tcl'], _channels=b['_channels'], _ch_per_die=b['_ch_per_die'], _ranks=b['_ranks'])
            
            # prepare inputs
            d = {}
            d['id'] = count
            d['ranks'] = ranks
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
            d['mdl_over_mat'] = mdl_over_mat
            d['ha_layout'] = ha_layout
            d['ha_double_ldls'] = ha_double_ldls
            d['salp_groups'] = salp_groups
            d['salp_all'] = salp_all
            d['pages_per_bgbus_mux'] = pages_per_bgbus_mux
            d['mdl_bgbus_sd'] = mdl_bgbus_sd
            d['bgbuses_per_gbus'] = bgbuses_per_gbus
            d['bgbus_gbus_sd'] = bgbus_gbus_sd
            d['gbuses_out'] = gbuses_out
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

        # save
        fieldnames = data[0].keys()
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(data)

        print("Saved to",csv_name)
        print("Be sure to rename it before rerunning!")
        print()

        return 0


if __name__ == '__main__':

    # defaults
    hbm_json_name = "configs/mem/sweep/hbm_sweep_full_databus.json"
    tech_json_name = "configs/tech/scaled/16nm_scaled.json"
    #hbm_json_name = "configs/mem/sweep/hbm2e.json"
    #tech_json_name = "configs/tech/scaled/17nm_hbm2e.json"
    output_csv_name = 'data/default/hbm3_default.csv'
    hbmtech_default = True
    output_default = True

    args = sys.argv[1:]
    options = "hmto:"
    long_options = ["help", "mem", "tech", "out"]

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
                os.makedirs(f'data/{OUTPUT_LABEL}', exist_ok=True)

    except getopt.error as err:
        print(str(err))

    if hbmtech_default:
        print("No memory or tech files specified, using defaults.")

    if output_default:
        print("No output file specified, writing to data/default/hbm3_default.csv") #TODO: stop using hbm3 as data file name prefix, here and in plotting
        os.makedirs(f'data/default', exist_ok=True)

    print()

    sys.exit(main(hbm_json_name, tech_json_name, output_csv_name))