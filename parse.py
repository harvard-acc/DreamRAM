import json

def mem_baseline(file_path):
    # Extract memory baseline

    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Error: The file '{file_path}' was not found.")
        return None
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from the file '{file_path}'.")
        return None

    # get the first few layers
    memconfig = data.get('memconfig', {})

    organization = memconfig.get('organization', {})
    bank = memconfig.get('bank', {})
    mat = memconfig.get('mat', {})
    databus = memconfig.get('databus', {})
    mods = memconfig.get('mods', {})
    blsa = memconfig.get('blsa', {})
    calibration = memconfig.get('calibration', {})

    # Create dictionary
    config_details = {
        'id': 0,
        'sids': organization.get('sids'),
        'channels': organization.get('channels'),
        'ch_per_die': organization.get('channels per die'),
        'pch': organization.get('pseudochannels'),
        'horiz_bg': organization.get('horizontal bankgroups'),
        'vert_bg': organization.get('vertical bankgroups'),
        'banks': organization.get('banks'),
        'subarrays': bank.get('subarrays'),
        'mat_rows': mat.get('wordlines'),
        'mats': bank.get('mats'),
        'mat_cols': mat.get('bitlines'),
        'brvsa': 0 if blsa.get('type')=="blsa" else 1,
        'ldls_mdls': mods.get('mdls'),
        # 'mdl_csl_over_mat': mods.get('mdl/csl config') >> 1,
        # 'mdl_over_mat': mods.get('mdl/csl config') & 1,
        'mdl_over_mat': mods.get('mdl/csl config'),
        'csl_mdl_shared_layer': mods.get('csl mdl shared layer', 0),
        'ha_layout': mods.get('ha layout'),
        'ha_double_ldls': mods.get('ha full'),
        'subchannels': mods.get('subchannels'),
        'salp_groups': mods.get('salp groups'),
        'salp_all': mods.get('salp all'),
        'pages_per_bgbus_mux': databus.get('pages per bgbus mux'),
        'mdl_bgbus_sd': databus.get('mdl-bgbus serdes'),
        'bgbuses_per_gbus': databus.get('bgbuses per gbus'),
        'bgbus_gbus_sd': databus.get('bgbus-gbus serdes'),
        #'gbuses_out': databus.get('gbuses'),
        'gbus_tsv_sd': databus.get('gbus-tsv serdes'),
        'tsv_dq_sd': databus.get('tsv-dq serdes'),
        'atom_size': mods.get('atom size'),
        # baseline references and other values that need to be set in sweep
        'isolation_rows_overhead': mat.get('wordline overhead')+1,
        'isolation_cols_overhead': mat.get('bitline overhead')+1,
        '_mat_rows': mat.get('wordlines'),
        '_mat_cols': mat.get('bitlines'),
        '_subarrays': bank.get('subarrays'),
        '_ldls_mdls': mods.get('mdls'),
        '_mats': bank.get('mats'),
        '_subchannels': mods.get('subchannels'),
        '_tck': calibration.get('tck'),
        '_tcl': calibration.get('tcl'),
        '_channels': organization.get('channels'),
        '_ch_per_die': organization.get('channels per die'),
        '_sids': organization.get('sids'),
        '_timing_baseline': memconfig.get('timing baseline'), 
        '_vert_bg': organization.get('vertical bankgroups'), 
        '_pages_per_bgbus_mux': databus.get('pages per bgbus mux'),
        '_mdl_bgbus_sd': databus.get('mdl-bgbus serdes'),
        '_bgbus_gbus_sd': databus.get('bgbus-gbus serdes'),
        '_gbus_tsv_sd': databus.get('gbus-tsv serdes')
    }

    return config_details


def mem_baseline_and_sweep(file_path):
    # Get both the baseline and sweep, returning b, s

    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Error: The file '{file_path}' was not found.")
        return None
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from the file '{file_path}'.")
        return None

    # First few gets
    memconfig = data.get('memconfig', {})

    organization = memconfig.get('organization', {})
    bank = memconfig.get('bank', {})
    mat = memconfig.get('mat', {})
    blsa = memconfig.get('blsa', {})
    databus = memconfig.get('databus', {})
    mods = memconfig.get('mods', {})

    # dictionary
    sweep_details = {
        'sids_list': organization.get('sids', []),
        'channels_list': organization.get('channels', []),
        'ch_per_die_list': organization.get('channels per die', []),
        'pch_list': organization.get('pseudochannels', []),
        'horiz_bg_list': organization.get('horizontal bankgroups', []),
        'vert_bg_list': organization.get('vertical bankgroups', []),
        'banks_list': organization.get('banks', []),
        'subarrays_list': bank.get('subarrays', []),
        'mat_rows_list': mat.get('wordlines', []),
        'mats_list': bank.get('mats', []),
        'mat_cols_list': mat.get('bitlines', []),
        'brvsa_list': blsa.get('type', []),
        'ldls_mdls_list': mods.get('mdls', []),
        # 'mdl_csl_over_mat_list': mods.get('mdl/csl config', []) >> 1,
        # 'mdl_over_mat_list': mods.get('mdl/csl config', []) & 1,
        'mdl_over_mat_list': mods.get('mdl/csl config', []),
        'csl_mdl_shared_layer_list': mods.get('csl mdl shared layer', []),
        'ha_layout_list': mods.get('ha layout', []),
        'ha_double_ldls_list': mods.get('ha full', []),
        'subchannels_list': mods.get('subchannels', []),
        'salp_groups_list': mods.get('salp groups', []),
        'salp_all_list': mods.get('salp all', []),
        'pages_per_bgbus_mux_list': databus.get('pages per bgbus mux', []),
        'mdl_bgbus_sd_list': databus.get('mdl-bgbus serdes', []),
        'bgbuses_per_gbus_list': databus.get('bgbuses per gbus', []),
        'bgbus_gbus_sd_list': databus.get('bgbus-gbus serdes', []),
        #'gbuses_out_list': databus.get('gbuses', []),
        'gbus_tsv_sd_list': databus.get('gbus-tsv serdes', []),
        'tsv_dq_sd_list': databus.get('tsv-dq serdes', []),
        'atom_size_list': mods.get('atom size', [])
    }

    # get baseline
    baseline_path = memconfig.get('baseline')
    baseline_details = mem_baseline(baseline_path)

    return baseline_details, sweep_details


def timing_baseline(file_path, a):
    # Parse HBM3 timing baseline and merge into tech dict (overrides process baseline values)

    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Error: The file '{file_path}' was not found.")
        return a
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from the file '{file_path}'.")
        return a

    timing = data.get('timing', {})
    s = timing.get('sense amp', {})
    decomp = timing.get('decomposition', {})
    cal = timing.get('calibration', {})

    # sense amp overrides
    if s.get('trcd') is not None:
        a['_trcd'] = s['trcd']
    if s.get('trcd signal') is not None:
        a['_trcd_signal'] = s['trcd signal']
    if s.get('wordlines ref') is not None:
        a['_mat_rows_ref'] = s['wordlines ref']
    if s.get('trcd brvsa') is not None:
        a['_trcd_brvsa'] = s['trcd brvsa']
    if s.get('brv delta v boost') is not None:
        a['_brvsa_brv_deltav_boost'] = s['brv delta v boost']
    if s.get('brvsa cs proportion') is not None:
        a['_brvsa_cs_proportion'] = s['brvsa cs proportion']
    if s.get('brvsa height ratio') is not None:
        a['_brvsa_height_ratio'] = s['brvsa height ratio']
    if s.get('cell leak') is not None:
        a['cell_leak'] = s['cell leak']
    if s.get('min delta v') is not None:
        a['_min_deltav'] = s['min delta v']
    if s.get('mdl over mat height ratio') is not None:
        a['_mdl_over_mat_height_ratio'] = s['mdl over mat height ratio']

    # decomposition overrides
    if decomp.get('trcdwr blsa fraction') is not None:
        a['_trcdwr_blsa_fraction'] = decomp['trcdwr blsa fraction']
    if decomp.get('tras restore') is not None:
        a['_tras_restore'] = decomp['tras restore']
    if decomp.get('twr restore') is not None:
        a['_twr_restore'] = decomp['twr restore']
    if decomp.get('trrds') is not None:
        a['_trrds'] = decomp['trrds']
    if decomp.get('trrdl') is not None:
        a['_trrdl'] = decomp['trrdl']

    # calibration overrides
    if cal.get('tck') is not None:
        a['_tck'] = cal['tck']
    if cal.get('tcl') is not None:
        a['_tcl'] = cal['tcl']
    if cal.get('die y') is not None:
        a['_die_y'] = cal['die y']

    # voltage overrides
    v = timing.get('voltage', {})
    if v.get('vcore') is not None:
        a['vcore_int'] = v['vcore']
    if v.get('vperi') is not None:
        a['vperi'] = v['vperi']
    if v.get('vpre ldl') is not None:
        a['vpre_ldl'] = v['vpre ldl']

    # tsv overrides
    tsv = timing.get('tsv', {})
    if tsv.get('tsv pitch') is not None:
        a['tsv_pitch'] = tsv['tsv pitch']
    if tsv.get('tsv koz') is not None:
        a['tsv_koz'] = tsv['tsv koz']
    if tsv.get('tsv height') is not None:
        a['tsv_height'] = tsv['tsv height']
    if tsv.get('ubump pitch') is not None:
        a['ubump_pitch'] = tsv['ubump pitch']

    return a


def tech_baseline(file_path, a):
    # extract tech baseline parameters

    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Error: The file '{file_path}' was not found.")
        return None
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from the file '{file_path}'.")
        return None

    # extract first few gets
    tech = data.get('tech', {})
    d = tech.get('die', {})
    u = tech.get('unscaled node', {})
    t = tech.get('tsv', {})
    c = tech.get('capacitance', {})
    w = c.get('same layer proportions', {})
    s = tech.get('sense amp', {})
    y = s.get('type', {})
    blsa = y.get('blsa', {})
    brvsa = y.get('brvsa', {})
    v = tech.get('voltage', {})
    vext = v.get('external', {})
    vint = v.get('internal', {})
    z = tech.get('tile size', {})

    # make dictionary
    a['max_die_dims_mm'] = d.get('max dimension')/1000
    a['_f'] = u.get('feature size')
    a['_tsv_pitch'] = t.get('tsv pitch')
    a['_tsv_koz'] = t.get('tsv koz')
    a['_tsv_height'] = t.get('tsv height')
    a['_c_tsv'] = t.get('tsv c')
    a['_r_load'] = t.get('tsv r load')
    a['_c_load'] = t.get('tsv c load')
    a['_ubump_pitch'] = t.get('ubump pitch', 55)
    a['_c_bus'] = c.get('c bus')
    a['_c_ca'] = c.get('c ca')
    a['_c_mwl'] = c.get('c mwl')
    a['_c_lwl'] = c.get('c lwl')
    a['_c_bl_per_cell'] = c.get('c bl per cell')
    a['_c_cell'] = c.get('c cell')
    a['_c_blsa'] = c.get('c blsa')
    a['_c_csl'] = c.get('c csl')
    a['_c_ldl'] = c.get('c ldl')
    a['_c_mdl'] = c.get('c mdl')
    a['c_dq'] = c.get('c dq')
    a['c_ca_ra_pin'] = c.get('c ca ra pin')
    a['_c_within_layer'] = w.get('min pitch')
    a['_c_within_layer_top'] = w.get('min pitch top')
    a['_c_within_layer_sparse'] = w.get('sparse')
    a['_c_within_layer_top_sparse'] = w.get('sparse top')
    a['_trcd'] = blsa.get('trcd')
    a['_trcd_signal'] = blsa.get('trcd signal')
    a['_mat_rows_ref'] = blsa.get('wordlines ref')
    a['_trcd_brvsa'] = brvsa.get('trcd')
    a['_brvsa_brv_deltav_boost'] = brvsa.get('brv delta v boost')
    a['_brvsa_cs_proportion'] = brvsa.get('brvsa cs proportion')
    a['_brvsa_height_ratio'] = brvsa.get('brvsa height ratio')
    a['cell_leak'] = s.get('cell leak')
    a['_min_deltav'] = s.get('min delta v')
    a['_mdl_over_mat_height_ratio'] = s.get('mdl over mat height ratio')
    a['vdd'] = vext.get('vdd')
    a['vpp'] = vext.get('vpp')
    a['vpp_int'] = vint.get('vpp')
    a['vddql_int'] = vint.get('vddql')
    a['vcore_int'] = vint.get('vcore')
    a['vpp_eff'] = vint.get('vpp pump eff')
    a['_coldec_height'] = z.get('column decoder height')
    a['_rowdec_width'] = z.get('row decoder width')
    a['_blsa_height'] = z.get('blsa height')
    a['_swd_width'] = z.get('swd width')

    return a


def tech(file_path):
    # extract all tech parameters
    # calls the tech_baseline

    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Error: The file '{file_path}' was not found.")
        return None
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from the file '{file_path}'.")
        return None

    # gets
    tech = data.get('tech', {})
    node = tech.get('scaled node', {})
    conf = tech.get('scale confidence', {})
    tsv = tech.get('tsv', {})

    # dictionary
    tech_details = {
        'f': node.get('feature size'),
        'pitch_wl': node.get('wlp'),
        'pitch_bl': node.get('blp'),
        'pitch_ldl_to_wl_min': node.get('ldl to wl pitch'),
        'pitch_mdl_to_bl_min': node.get('mdl to bl pitch'), 
        'c_scale_conf': conf.get('c scale conf'),
        'c_blsa_scale_conf': conf.get('c blsa scale conf'),
        'tsv_c_pitch_scale_conf': conf.get('c tsv scale conf'), 
        'logic_scale_conf': conf.get('logic scale conf'),
        'coldec_scale_conf': conf.get('column decode scale conf'), 
        'rowdec_scale_conf': conf.get('row decode scale conf'), 
        'swd_scale_conf': conf.get('swd scale conf'),
        'blsa_scale_conf': conf.get('blsa scale conf'),
    }

    # tsv (optional in scaled config, can be overridden by timing baseline)
    if tsv.get('tsv pitch') is not None:
        tech_details['tsv_pitch'] = tsv['tsv pitch']
    if tsv.get('tsv koz') is not None:
        tech_details['tsv_koz'] = tsv['tsv koz']
    if tsv.get('tsv height') is not None:
        tech_details['tsv_height'] = tsv['tsv height']
    if tsv.get('ubump pitch') is not None:
        tech_details['ubump_pitch'] = tsv['ubump pitch']

    # get baseline
    baseline_path = tech.get('baseline')
    tech_details = tech_baseline(baseline_path, tech_details)

    return tech_details