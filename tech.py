import numpy as np
from dataclasses import dataclass

@dataclass
class Tech:
    # TSV via (through silicon)
    tsv_pitch: float = 25         # um (TSV via center-to-center pitch)
    tsv_koz: float = 4            # um (active device exclusion from TSV edge)
    tsv_height: float = 30        # um (die thickness for HBM3)

    # Microbump (die-to-die surface connection)
    ubump_pitch: float = 40       # um (microbump pitch, limits surface packing)

    # Max die dimensions
    max_die_dims_mm: float = 13 # mm
    max_stack_dies: float = 16 # dies

    # Scaled node
    f: float = 0.016 # um
    pitch_wl: float = 0.041 # um
    pitch_bl: float = 0.047 # um
    pitch_ldl_to_wl_min: float = 2 # only used in blsa_height for shared layer
    pitch_mdl_to_bl_min: float = 4 # shared layer min (4×BL); doubled to 8×BL for separate layers
    pitch_ldl: float = pitch_wl*pitch_ldl_to_wl_min # min
    pitch_mdl: float = pitch_bl*pitch_mdl_to_bl_min # min # keep as the minimum allowed (4x) for shared. only usage is in swd_width for shared layer

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
    rowdec_scale_conf: float = 0.9 # match to json
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
    _c_mdl: float = 0.00098 #0.00049 # pF/um # match CSL in baseline

    # cell leakage before read
    cell_leak: float = 0.2

    # TSV baseline
    _tsv_pitch: float = 55 # um (baseline reference for scaling)
    _tsv_koz: float = 27 # um
    _tsv_height: float = 50 # um z direction
    _c_tsv: float = 1.1 # pF/layer (baseline reference)
    _r_load: float = 100 # Ohm
    _c_load: float = 0.120 # pF
    _ubump_pitch: float = 55 # um (baseline microbump pitch for scaling)
    
    c_dq: float = 1.58 # pF
    c_ca_ra_pin: float = 0.5 # pF, JEDEC JESD238B.01 Table 73 C_ADDR

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
    vperi_ldl: float = 1.2 # peripheral voltage for LDL
    vpre_ldl: float = 0.6 # LDL precharge voltage
    vpp_eff: float = 0.5 # vpp pump efficiency

    # Proportions of wire capacitance from wires in the same layer (discard fringe)
    # Derived from C_c/C_g = 2W/S:
    #   min pitch (S=W):  C_c/C_g=2.0, p=2/3= 0.67
    #   4x pitch (S=4W):  C_c/C_g=0.5, p=(0.5/1.5) = 0.33
    _c_within_layer: float = 0.8 # for tightly packed
    _c_within_layer_top: float = 0.89
    _c_within_layer_sparse: float = 0.66 # for wider pitch wires
    _c_within_layer_top_sparse: float = 0.8
    _c_within_layer_sparse_csl_mdl: float = 0.33 # for CSL/MDL at ~4x BL pitch

    # baseline trcd u+4s
    _trcd: float = 18 # ns
    _trcd_brvsa: float = 10.8 # ns
    _trcd_signal: float = 2.9 # ns signal travel time
    _brvsa_cs_proportion: float = 0.89 # brvsa trcd dominated by OC/CS/BRV, especially with already-wide DeltaV
    _brvsa_brv_deltav_boost: float = 1.75
    _brvsa_height_ratio: float = 1.33
    _mdl_over_mat_height_ratio: float = 1.26
    _mat_rows_ref: float = 512  # HBM3 baseline wordlines per MAT (was 640 for Ha/2y process ref)
    _min_deltav: float = 0.1 # Minimum blsa DeltaV before sensing yield drops. For reference only.

    # === Timing baselines for new parameters ===

    # tRCDWR: fraction of BLSA time needed for write (write driver overdrives SA before full sensing)
    # Physical basis: — write driver beta ratio 5:1-8:1 overpowers SA after partial development
    # Calibrated: tRCDWR_baseline/tRCDRD_baseline ≈ 9/13.9 ≈ 0.65 of BLSA component
    _trcdwr_blsa_fraction: float = 0.404

    # tRAS restore: time for SA to restore charge into cell after full amplification
    # Physical basis: RC time through access transistor (VPP-biased gate) into cell capacitor
    # Calibrated: tRAS_ref - tRCDRD_ref = 27 - 18 = 9.0 ns at mat_rows_ref=512
    _tras_restore: float = 9.0  # ns, at mat_rows_ref=512

    # tRRD: minimum time between ACT commands to different banks
    # Physical basis: shared row command path (address bus, row decoder, wordline drivers)
    # must settle before the next ACT can be issued. Scales with bank_y (row cmd path length).
    # Calibrated: HBM3 tRRDS = tRRDL = 2.5 ns at baseline bank_y
    _trrds: float = 2.5  # ns, different bank group
    _trrdl: float = 2.5  # ns, same bank group

    # tWR: write recovery — SA must fully restore cell after write data overwrites SA
    # Physical basis: same RC physics as tRAS restore but starting from write driver state
    # Calibrated: HBM3 tWR ≈ 21.5 ns. Decomposed: tWR_baseline = 21.5 - 2.9 (signal) = 18.6 ns BLSA portion
    _twr_restore: float = 18.6  # ns, at mat_rows_ref=512

    # misc
    bl_pre_beta: float = 0.25 # PRE sharing factor: ideal = 0, worst case = 0.25 



    def tsv_density(self):
        # density in TSVs per square um (rectangular packing)
        # Surface area limited by whichever is larger: microbump or TSV via pitch
        pitch = max(self.ubump_pitch, self.tsv_pitch)
        #return 2 / (np.sqrt(3) * np.power(pitch, 2)) for hexa packing 
        return 1 / np.power(pitch, 2) # square packing
    
    def scale_cap(self, c):
        # scale capacitance by node, according to the scaling confidence
        return c * np.pow((self.f/self._f), self.c_scale_conf)
    
    def scale_cap_blsa(self, c):
        # scale capacitance by node, according to the scaling confidence
        return c * np.pow((self.f/self._f), self.c_blsa_scale_conf)
    
    def scaled_cap_tsv(self):
        # scale capacitance by TSV via pitch and height
        # TSV capacitance scales with via surface area (pitch × height), not microbump pitch
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

    def rc_ratio_fixed_width(self, pitch_scale_ratio, p):
        # Fixed wire width, only spacing changes
        # within layer coupling C scales as 1/ratio (more spacing)
        # between layer C unchanged (same wire width)
        # R unchanged (same wire cross-section)
        return (1-p) + p / pitch_scale_ratio

    def c_ratio(self, pitch_scale_ratio, p):
        # within layer portion has C/ratio
        # between layer portion gets C*ratio
        return ((1-p)*pitch_scale_ratio + p / pitch_scale_ratio)
