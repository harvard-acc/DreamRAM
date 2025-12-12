import numpy as np
from dataclasses import dataclass

@dataclass
class Tech:
    # TSV
    tsv_pitch: float = 55 # um
    tsv_koz: float = 27.5 # um
    tsv_height: float = 30 # um z direction

    # Max die dimensions
    max_die_dims_mm: float = 13 # mm
    max_stack_dies: float = 16 # dies

    # Scaled node
    f: float = 0.016 # um
    pitch_wl: float = 0.041 # um
    pitch_bl: float = 0.047 # um
    pitch_ldl_to_wl_min: float = 4
    pitch_mdl_to_bl_min: float = 4
    pitch_ldl: float = pitch_wl*pitch_ldl_to_wl_min # min
    pitch_mdl: float = pitch_bl*pitch_mdl_to_bl_min # min

    # baseline feature size
    # you can think of the leading underscore as, well, a "base" "line"
    _f: float = 0.026 # um

    # Scaling confidences
    # defines the confidence [0,1] that a parameter scales with node
    # calculated as (node / baseline) ^ conf
    c_scale_conf: float = 0.7
    c_blsa_scale_conf: float = 0.7
    tsv_c_pitch_scale_conf: float = 1
    logic_scale_conf: float = 0.9
    coldec_scale_conf: float = 0.9
    rowdec_scale_conf: float = 0.7
    swd_scale_conf: float = 0.7
    blsa_scale_conf: float = 0.7

    # baseline capacitances from Ha for 2y
    _c_bus: float = 0.0003 #pF/um
    _c_ca: float = 0.00035 # pF/um
    _c_mwl: float = 0.00155 # pF/um
    _c_lwl: float = 0.000886 # pF/um
    _c_bl_per_cell: float = 0.0000417 # pF/cell
    _c_cell: float = 0.008 # pF
    _c_blsa: float = 0.0033 # pF
    _c_csl: float = 0.00098 # pF/um
    _c_ldl: float = 0.00049 # pF/um
    _c_mdl: float = 0.00049 # pF/um

    # cell leakage before read
    cell_leak: float = 0.2

    # TSV baseline
    _tsv_pitch: float = 55 # um
    _tsv_koz: float = 27 # um
    _tsv_height: float = 50 # um z direction
    _c_tsv: float = 1.1 # pF/layer
    _r_load: float = 100 # Ohm
    _c_load: float = 0.120 # pF
    
    c_dq: float = 2.4 # pF

    # baseline tile sizes
    _coldec_height: float = 150.0 # um
    _rowdec_width: float = 82.5 # um
    _blsa_height: float = 11.0 # um
    _swd_width: float = 5.0 # um

    # Voltage Domains
    # external
    vdd: float = 1.1 # V
    vpp: float = 1.8
    # internal
    vpp_int: float = 2.5 # internal VPP for wordlines
    vddql_int: float = 0.4 # low swing
    vcore_int: float = 1.0
    vpp_eff: float = 0.5 # vpp pump efficiency

    # Proportions of wire capacitance from wires in the same layer (discard fringe)
    _c_within_layer: float = 0.8 # for tightly packed
    _c_within_layer_top: float = 0.89
    _c_within_layer_sparse: float = 0.66 # for wider double-pitch wires
    _c_within_layer_top_sparse: float = 0.8

    # baseline trcd u+4s
    _trcd: float = 13.9 # ns
    _trcd_brvsa: float = 10.8 # ns
    _trcd_signal: float = 2.9 # ns signal travel time
    _brvsa_cs_proportion: float = 0.67 # brvsa trcd dominated by OC/CS/BRV, especially with already-wide DeltaV
    _brvsa_brv_deltav_boost: float = 1.75
    _brvsa_height_ratio: float = 1.33
    _mdl_over_mat_height_ratio: float = 1.26
    _mat_rows_ref: float = 640
    _min_deltav: float = 0.1 # Minimum blsa DeltaV before sensing yield drops. For reference only.

    # misc
    bl_pre_beta: float = 0.1 # PRE sharing factor: ideal = 0, worst case = 0.25



    def tsv_density(self):
        # density in TSVs per square um
        return 2 / (np.sqrt(3) * np.power(self.tsv_pitch, 2) )
    
    def scale_cap(self, c):
        # scale capacitance by node, according to the scaling confidence
        return c * np.pow((self.f/self._f), self.c_scale_conf)
    
    def scale_cap_blsa(self, c):
        # scale capacitance by node, according to the scaling confidence
        return c * np.pow((self.f/self._f), self.c_blsa_scale_conf)
    
    def scaled_cap_tsv(self):
        # scale capacitance by TSV pitch and height, according to the scaling confidence
        return self._c_tsv * np.pow((self.tsv_pitch/self._tsv_pitch), self.tsv_c_pitch_scale_conf) * (self.tsv_height/self._tsv_height) + self._c_load
    
    def scale_logic_dim(self, l):
        # scale dimensions by node, according to the scaling confidence
        return l * np.pow((self.f/self._f), self.logic_scale_conf)
        
    def scaled_coldec_height(self):
        # scale dimensions by node, according to the scaling confidence
        return self._coldec_height * np.pow((self.f/self._f), self.logic_scale_conf)
        
    def scaled_rowdec_width(self):
        # scale dimensions by node, according to the scaling confidence
        return self._rowdec_width * np.pow((self.f/self._f), self.logic_scale_conf)
        
    def scaled_blsa_height(self):
        # scale dimensions by node, according to the scaling confidence
        return self._blsa_height * np.pow((self.f/self._f), self.logic_scale_conf)
    
    def scaled_swd_width(self):
        # scale dimensions by node, according to the scaling confidence
        return self._swd_width * np.pow((self.f/self._f), self.logic_scale_conf)
        

    def rc_ratio(self, pitch_scale_ratio, p):
        # within layer portion has C/ratio
        # between layer portion gets C*ratio
        # all get R/ratio
        return ((1-p)*pitch_scale_ratio + p / pitch_scale_ratio) / pitch_scale_ratio
    
    def c_ratio(self, pitch_scale_ratio, p):
        # within layer portion has C/ratio
        # between layer portion gets C*ratio
        return ((1-p)*pitch_scale_ratio + p / pitch_scale_ratio)
