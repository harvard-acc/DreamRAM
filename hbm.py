import numpy as np
from dataclasses import dataclass
import math
from tech import Tech

@dataclass
class Hbm:
    
    # Organization
    sids: int = 2
    channels: int = 16 # total
    ch_per_die: int = 4
    pch: int = 2
    horiz_bg: int = 1
    vert_bg: int = 4
    banks: int = 4

    # Bank Org
    subarrays: int = 16
    mat_rows: int = 512
    mats: int = 32
    mat_cols: int = 512
    
    # Bitline Architecture
    open: int = 1      # Open bitline (6F²)
    folded: int = 0    # Folded bitline (8F²)
    

    # Error and Repair
    isolation_rows_overhead: float = 1.015 # factor of a cell array
    isolation_cols_overhead: float = 1.015 # factor of a cell array
    repair_subarray: int = 0
    md_ecc_prop: float = 1/16 # (19,17) Reed Solomon: 256b data + 16b metadata + 32b ECC
    od_ecc_prop: float = 1/8

    # Modifications
    # default values are shown here for setting the modification to "off"
    ha_layout: int = 0 # y/n
    ha_double_ldls: int = 0 # y/n. Must have ha_layout also 1
    subchannels: int = 1 # number of subchannels
    mdl_over_mat: int = 1 # Must be mdl over mats
    mdl_csl_over_mat: int = 1 # MDL/CSL routed over MAT, BLSA uses base height
    csl_mdl_shared_layer: int = 1 # 0 = separate metal layers for CSL and MDL, 1 = shared metal layer (tighter pitch)
    salp_groups: int = 1 # incompatible with salp_all
    salp_all: int = 0 # incompatible with salp_groups
    ldls_mdls: int = 8 # non-differential datalines per MAT
    atom_size: int = 256 # bits
    brv_sa: int = 0

    # Data I/O Hierarchy
    pages_per_bgbus_mux: int = 1 # MDL sets to BGBUS mux factor when ind_pages != 1; otherwise only 1 MDL set (default)
    mdl_bgbus_sd: float = 1 # serdes
    bgbuses_per_gbus: int = 2 # sets number of gbuses
    bgbus_gbus_sd: float = 1
    #gbuses_out: int = 2 # per pch # INSTEAD USE gbuses()
    gbus_tsv_sd: float = 1 # before it was 4, fixed it to 1
    tsv_dq_sd: float = 8 # from TSV to DQ should be 8
    dbi_factor: float = 1.125 # 1 bit per byte
    dbi_for_bgbus: int = 1

    # Reference
    # baseline values for tCK and other timing calculations
    # don't worry, these are copied from the baseline design at runtime, including the -1's
    _mat_rows: float = 512
    _mat_cols: float = 512
    _subarrays: float = 16
    _ldls_mdls: float = 8
    _mats: float = 32
    _subchannels: float = 1
    _vert_bg: float = 4
    _n_bgbus: float = 256
    _pages_per_bgbus_mux: float = 1
    _mdl_bgbus_sd: float = 1
    _bgbus_gbus_sd: float = 1
    _gbus_tsv_sd: float = 1
    _sids: float = 2
    _channels: float = 16
    _ch_per_die: float = 4
    _tck: float = 2
    _die_y: float = -1
    _tcl: float = 38.4
    _bank_y: float = -1
    _bank_x: float = -1



    ''' Some Constants '''

    def md_ecc_factor(self):
        return 1 + self.md_ecc_prop


    def ecc_factor(self):
        return 1 + self.md_ecc_prop + self.od_ecc_prop


    def dbi_transition_factor_avg(self):
        # scale transitions with the DBI overhead
        if self.dbi_factor == 1.125:
            return 0.818
        if self.dbi_factor == 1.0625:
            return 0.854
        if self.dbi_factor == 1:
            return 1
        assert(False) # DBI factor should be 1, 9/8, or 17/16

    
    def dbi_transition_factor_max(self):
        # DBI scales max number of transitions by half
        return 0.5



    ''' COMMAND AND PAGE WIDTHS '''

    def ind_row_pages(self):
        pages = np.power(2, self.ha_layout) * self.subchannels
        pages_int = int(pages)
        assert(pages_int == pages)
        return pages_int


    def ind_pages(self):
        # independent subbanks
        pages = np.power(2, self.ha_layout - self.ha_double_ldls) * self.subchannels
        pages_int = int(pages)
        assert(pages_int == pages)
        return pages_int


    def ch_cmd_bits(self): # Changed in the comments
        # command bits per channel
        ind_row_pages = self.ind_row_pages()
        ind_col_pages = self.ind_pages()

        sid_bits = max(2, int(np.ceil(np.log2(max(self.sids, 2)))))
        ba_bits = np.ceil(np.log2(self.vert_bg*self.horiz_bg*self.banks))
        pc_bits = np.ceil(np.log2(self.pch))
        ra_bits = np.ceil(np.log2(self.subarrays)) + np.ceil(np.log2(self.mat_rows)) + np.log2(ind_row_pages)
        ca_bits = np.ceil(np.log2(self.atoms_per_page())) # + np.log2(ind_col_pages) # not needed, atoms_per_page is logical page
        
        # Row bus width: ROW uses 3 clock edges                                                                                                             
        row_opcode = 3  #these are non-toggling counts regardless of architecture                                                                                                                                        
        row_first_edge = row_opcode + pc_bits + sid_bits + ba_bits                                                                                                        
        row_second_third_edge = 2 + int(np.ceil(ra_bits / 2))                                                                                                                   
        row_int = max(row_first_edge, row_second_third_edge)
        row= row_int
        # Column bus width: READ/WRITE uses 2 clock edges      
        col_opcode = 4    #these are non-toggling counts regardless of architecture                                                                                                                                                        
        col_total = col_opcode + pc_bits + sid_bits + ba_bits + ca_bits                                                                                              
        col_int = int(np.ceil(col_total / 2)) 
        col = col_int
        
        assert(row_int == row)
        assert(col_int == col)
        return row, col



    ''' STACK DIMENSIONS '''

    def decoder_dims(self, tech:Tech):
        #um
        return tech.scaled_rowdec_width(), tech.scaled_coldec_height()


    def swd_width(self, tech:Tech):
        # um
        ref_w = tech.scaled_swd_width()
        return ref_w


    def blsa_height(self, tech:Tech):
        # um
        # scaled reference
        ref_y = tech.scaled_blsa_height()
        y = ref_y

        if self.brv_sa:
            # estimate overhead of OC/ISO with same N:P
            y = y * tech._brvsa_height_ratio

        if self.mdl_over_mat:
            # increased height from MDL driver and scale if more than 8 ldls
            y = max(ref_y * tech._mdl_over_mat_height_ratio, y + (self.ldls_mdls - 8) * tech.pitch_ldl)
        else:
            # scale if more than 8 ldls
            y = max(ref_y, y + (self.ldls_mdls - 8) * tech.pitch_ldl)

        assert(y>0)
        return y


    def bank_dims(self, tech:Tech):
        # um
        # MATs, subarrays, and column/row decoders

        # col/row dec
        row_dec_width, col_dec_y = self.decoder_dims(tech)

        # SWDs
        swd_width = self.swd_width(tech)
        # swd_width_total = swd_width * (self.mats + self.subchannels + 1) * self.ecc_factor() # +1 added
        # one more than the number of subarrays 
        ecc_mats = math.ceil((self.md_ecc_prop + self.od_ecc_prop) * self.mats) # 6MATS
        # (mats + ecc_mats + 1) = baseline SWDs
        # (subchannels - 1) = extra LWD stripes for subchannel segmentation (Chatterjee HPCA 2017)
        swd_count = (self.mats + ecc_mats + 1) + (self.subchannels - 1)
        swd_width_total = swd_width * swd_count

        # BLSAs
        blsa_y = self.blsa_height(tech)
        # BLSAs: always N+1 (shared between adjacent subarrays, plus one at each edge)
        blsa_y_total = blsa_y * (self.subarrays + (self.salp_groups-1) + self.repair_subarray + 1)

        # all cell arrays
        data_mats_width = self.mats * (self.mat_cols * self.isolation_cols_overhead) * tech.pitch_bl # 32 * 512 *1.5% * 0.047
        data_bits_per_row = self.mats * self.mat_cols # 32 * 512 = 16384
        ecc_bits_per_row = data_bits_per_row * ( self.md_ecc_prop + self.od_ecc_prop) # 3072
        ecc_block_width = ecc_bits_per_row * self.isolation_cols_overhead * tech.pitch_bl # 3072 * 1.5% *0.047
        cells_width_total = data_mats_width + ecc_block_width # add all!

        cells_y_total = (self.subarrays + (self.salp_groups-1) + (self.repair_subarray) + 1) * (self.mat_rows * self.isolation_rows_overhead) * tech.pitch_wl

        cell_area = cells_width_total * cells_y_total

        return row_dec_width + cells_width_total + swd_width_total, col_dec_y + cells_y_total + blsa_y_total, cell_area
    

    def _bank_x_calc(self, tech:Tech):
        # um
        if self._bank_x < 0:
            # baseline
            self._bank_x, _, __ = self.bank_dims(tech)
        return self._bank_x


    def _bank_y_calc(self, tech:Tech):
        # um
        if self._bank_y < 0:
            # baseline
            _, self._bank_y, __ = self.bank_dims(tech)
        return self._bank_y


    def dummy_subarray_height(self, tech:Tech):
        # um — total height of edge dummy reference subarrays (half-height each)
        # Open BL edges need 2 dummy subarrays; folded edges need 0
        dummy_count = 2 if self.open else 0
        mat_height = (self.mat_rows * self.isolation_rows_overhead) * tech.pitch_wl
        return dummy_count * 0.5 * mat_height


    def od_ecc_height(self, tech:Tech):
        # OD-ECC Engine itself should be about the height of 5%??? seems to high for 75% of bank_y
        # since one BG shares OD-ECC Engine
        return (0.01*self._bank_y_calc(tech))


    def mbus_peri_height(self, tech:Tech):
        return (0.55*self._bank_y_calc(tech)) # reduced from 0.75 based on HBM3 die analysis


    def bankdie_dims(self, tech:Tech):
        # all lengths in um
        # bank levels
        bank_x, bank_y, cell_area = self.bank_dims(tech)
        # OD -ECC Engine
        od_ecc_height = self.od_ecc_height(tech)
        # Mbus Height
        mbus_peri_height = self.mbus_peri_height(tech)
        # Row DEC + Col Dec
        row_dec_width, col_dec_height = self.decoder_dims(tech)
        center_height = 2.0*row_dec_width # ASSUME from die shot
        #Channel * pseudo channel * number of bgs * (bank_x + bank_dec_height)
        x = self.ch_per_die * self.pch * self.horiz_bg * (bank_x + center_height)
        # vertical * num banks * bank_y + od_ecc_height, but where is IOSA?
        # dummy subarrays (open BL edges) are per-bank physical overhead, not part of signal path
        dummy_y = self.dummy_subarray_height(tech)
        y = (self.vert_bg * self.banks * (bank_y + dummy_y)) + (self.vert_bg * od_ecc_height) + mbus_peri_height
        
        return x, y


    def calc_stack_dims(self, tech:Tech):
        # um
        # dies stacked
        dies_stacked = self.sids * self.channels / self.ch_per_die
        assert(dies_stacked == int(dies_stacked))
        dies_stacked = int(dies_stacked)
        
        # bank-area dims
        die_width, non_tsv_y = self.bankdie_dims(tech)

        # TSV heights
        row, col = self.ch_cmd_bits()
        #cmd_tsv_count = self.channels * (row+col)

        ra_signals = 4 * row # RA_E/O_F/S * RA[9:0] 
        ca_signals = 4 * col  # CA_E/O_F/S * CA[7:0] 
        ck_signals = 4  # 0/90/180/270 phase clocks for sampling address
        cke_signal = 1  # optional
        cmd_tsv_count = self.channels * (ra_signals + ca_signals + ck_signals + cke_signal)
        
        dbi_signals = (self.dbi_factor-1)*self.bgbus_width() # constant DBI inform bits
        wck_signals = 8 # 8 WCK Phases
        sev_signals = 16 # SEV + extra DQ flag signals
        data_tsv_count = self.channels * (self.ldls_mdls * self.mats + 1/((self.md_ecc_factor()-1)) + dbi_signals + sev_signals + wck_signals )

        # ASSUME TSVs must keep KOZ away from die edge
        height_per_tsv = 1/ ((die_width - 2 * tech.tsv_koz) * tech.tsv_density())
        assert(height_per_tsv >= 0)
        cmd_tsv_y = cmd_tsv_count * height_per_tsv
        data_tsv_y = data_tsv_count * height_per_tsv
        other_tsv_y = 0.2 * (cmd_tsv_y + data_tsv_y)  # 20% of data+cmd TSV area (reduced power TSV overhead)
        other_tsv_y += 2 * tech.tsv_koz

        return dies_stacked, die_width, non_tsv_y, cmd_tsv_y, data_tsv_y, other_tsv_y



    ''' PAGES '''

    def capacity(self):
        # GB
        # no ECC
        capacity = self.sids * self.channels * self.pch * self.vert_bg * self.horiz_bg * self.banks * self.subarrays * self.mat_rows * self.mats * self.mat_cols / np.power(2,10) / np.power(2,10) / np.power(2,10) / np.power(2,3)
        return capacity


    def page_act_size(self):
        # b
        p = self.mats * self.mat_cols / (self.subchannels * np.power(2,self.ha_layout - self.ha_double_ldls))
        assert(p == int(p)) # integer page size
        return int(p)


    def atoms_per_page(self):
        a = self.page_act_size() / self.atom_size
        #assert(a == int(a)) # integer number of atoms per page
        return a
    
    
    def min_atom(self):
        # b per pump
        return self.mats * self.ldls_mdls / (np.power(2.0, self.ha_layout - self.ha_double_ldls) * self.subchannels)
    

    def pumps_per_atom(self):
        a = self.atom_size / self.min_atom()
        assert(a == int(a)) # integer number of pumps per atom
        return a
    


    ''' IO RATES '''

    def mdl_width_per_page(self):
        # REMINDER: for energy, include both ECCs
        return self.mats * self.ldls_mdls / (np.power(2, self.ha_layout - self.ha_double_ldls) * self.subchannels)


    def bgbus_width(self):
        # REMINDER: for energy, include both ECCs
        # Per-subchannel bgbus width. Each subchannel has its own independent bgbus.
        # Total physical bgbus width across all subchannels = bgbus_width() * subchannels
        return self.mdl_width_per_page() / self.mdl_bgbus_sd


    def n_bgbus(self):
        # width of bgbus
        return np.ceil(self.bgbus_width() * self.md_ecc_factor())


    def gbus_width(self):
        # REMINDER: for energy, include MD-ECC
        return self.bgbus_width() / self.bgbus_gbus_sd


    def bank_clks_per_atom(self):
        # Number of MDL pumps per atom (pump count for energy calculations).
        # Each pump asserts a different CSL within the subchannel to read the next portion.
        # Pump rate: paper says tCCDL defines the cycle time, but since each subchannel
        # has an independent MDL, pumps may run faster (tCCDs) — TBD.
        # For energy, only the pump COUNT matters (number of wire switches), not the rate.
        x = self.atom_size / self.mdl_width_per_page()
        assert(x == int(x))
        return int(x)


    def bgbus_clks_per_atom(self):
        x= self.atom_size / self.bgbus_width()
        assert(x == int(x))
        return int(x)


    def gbus_clks_per_atom(self):
        x= self.atom_size / self.gbus_width()
        assert(x == int(x))
        return int(x)


    def tsv_speed_factor(self):
        # multiply by core freq for true speed
        # higher = faster
        return self.mdl_bgbus_sd * self.bgbus_gbus_sd * self.gbus_tsv_sd


    def dq_speed_factor(self):
        # added bgbuses_per_gbus because technically gbus is 2tCK shared by BGBUS
        # multiply by core freq for true speed
        # higher = faster
        return self.mdl_bgbus_sd * self.bgbus_gbus_sd * self.gbus_tsv_sd * self.tsv_dq_sd * self.bgbuses_per_gbus # 2
    
    
    def dq_count(self):
        # DQ pins are a physical resource — total count should not shrink with subchannels.
        # gbus_width() is per-subchannel; multiply by subchannels to get total physical gbus width.
        # Each subchannel has its own independent datapath (MDL, bgbus, gbus, I/Os),
        # so total DQ = sum across all subchannels.
        x = np.ceil(self.md_ecc_factor() * self.gbus_width() * self.subchannels / (self.gbus_tsv_sd * self.tsv_dq_sd) ) * self.pch * self.channels
        assert(x == int(x))
        return int(x)


    def wire_lengths(self, tech:Tech): 

        l = {}
        mbus_y = self.mbus_peri_height(tech)
        dies_stacked, die_width, non_tsv_y, cmd_tsv_y, data_tsv_y, other_tsv_y = self.calc_stack_dims(tech)
        tsv_area_y = cmd_tsv_y + data_tsv_y + other_tsv_y
        bank_area_y = non_tsv_y/2
        bank_only_y = bank_area_y - mbus_y
        bank_x, bank_y, cell_area = self.bank_dims(tech)
        swd_x = self.swd_width(tech)

        # row
        l["od-row-avg"] = tsv_area_y/2 + mbus_y + bank_only_y/2
        l["od-row-max"] = tsv_area_y/2 +bank_area_y
        l["mwl"] = bank_x
        l["lwl"] = (self.mat_cols * self.isolation_cols_overhead) * tech.pitch_bl + swd_x/2
        l["bl"] = (self.mat_rows * self.isolation_rows_overhead) * tech.pitch_wl # do not include BLSA
        # col
        l["od-col-avg"] = tsv_area_y/2 + mbus_y + bank_only_y/2 + bank_x/2
        l["od-col-max"] = bank_x + tsv_area_y/2 + bank_area_y 
        l["csl"] = bank_y - (self.mat_rows)  * tech.pitch_wl if self.folded == 1 else bank_y  # bank_y - last MAT height
        l["ldl"] = (self.mat_cols * self.isolation_cols_overhead) * tech.pitch_bl # runs horizontally across BLSA
        l["mdl"]= bank_y - (self.mat_rows)  * tech.pitch_wl if self.folded == 1 else bank_y # bank_y - last MAT height
        l["bgbus+gbus"] = mbus_y + bank_only_y # ASSUME group bgbus+gbus
        l["tsv-n-layers-avg"] = dies_stacked/2 #*tech.pitch_die 
        l["tsv-n-layers-max"] = dies_stacked #*tech.pitch_die
        l["base"] = (tsv_area_y+bank_area_y)/2

        return l


    def wire_counts(self):
        # wire counts active PER row or column command (NOT total)
        # for mdl and onwards: #nCK = atom_size / wire_count
        # clock rates come from serdes terms
        # so window = #nCK / serdes is in core clocks for a given atom
        # all the windows should line up due to muxing, except DQ, which merges the gbuses
        n = {}

        row, col = self.ch_cmd_bits()

        # per row cmd
        n["row"] = row
        n["mwl"] = 1 # one MWL asserts across the bank regardless of BL architecture
        n["lwl"] =(self.mats * self.ecc_factor()) / self.ind_row_pages() # open BL: two LWLs enabled per MAT (data + reference)
        n["bl"] = (self.mats * self.ecc_factor()) * (self.mat_cols)  / self.ind_row_pages()
        # per col cmd
        n["col"] = col
        n["csl"] = np.ceil((self.mats * self.ecc_factor()) / self.ind_pages()) # count 1 per MAT per cycle
        n["ldl"] = self.mdl_width_per_page() * self.ecc_factor()
        if not self.mdl_csl_over_mat and self.mdl_over_mat:
            # repurpose ldls to mean the horizontal part of csl
            n["ldl"] = n["csl"]
        else:
            assert(n['ldl'] <= self.atom_size * self.ecc_factor())
        n["mdl"] = np.ceil(self.mdl_width_per_page() * self.ecc_factor())
        n["bgbus"] = np.ceil(self.bgbus_width() * self.md_ecc_factor()) 
        n["gbus"] = np.ceil(self.gbus_width() * self.md_ecc_factor())
        n["tsv"] = np.ceil(self.gbus_width() * self.md_ecc_factor() / self.gbus_tsv_sd)
        n["dq"] = np.ceil(self.dq_count() / (self.pch * self.channels * self.ind_pages() ))
        
        assert(n['mdl'] <= self.atom_size * self.ecc_factor())
        assert(n['bgbus'] <= self.atom_size * self.md_ecc_factor())
        assert(n['gbus'] <= self.atom_size * self.md_ecc_factor())
        assert(n['tsv'] <= self.atom_size * self.md_ecc_factor())

        return n


    
    ''' BANDWIDTH AND TIMING CALCULATIONS '''

    def csl_mdl_pitch_ratios(self, tech:Tech):
        # Ensure the columns divide nicely over the MDLs/LDLs
        assert(self.mat_cols % self.ldls_mdls == 0)

        # Effective MDL-to-BL minimum pitch depends on metal layer sharing:
        #   Shared layer (shared=1):    CSL+MDL compete for same layer → tech min (4× BL pitch)
        #   Separate layers (shared=0): each signal type gets its own metal layer → 2× tech min (8× BL pitch)
        effective_mdl_to_bl_min = tech.pitch_mdl_to_bl_min #if self.csl_mdl_shared_layer else tech.pitch_mdl_to_bl_min * 2
        # minimum pitch allowed is a physical thing, does not depend on if you split your MDLs or not

        # In baseline, CSL and MDL share the same metal layer over MAT
        # Total wires = MDL (differential) + CSL
        _mdl_wires = self._ldls_mdls * 2  # differential MDL pairs
        _csl_wires = self._mat_cols / self._ldls_mdls  # CSL count per MAT
        _total_wires = _mdl_wires + _csl_wires
        _shared_pitch = self._mat_cols / _total_wires  # BL pitches per wire

        _mdl_to_bl_pitch = _shared_pitch
        _csl_to_bl_pitch = _shared_pitch

        if not self.mdl_csl_over_mat:
            if self.mdl_over_mat:

                mdl_to_bl_pitch = self.mat_cols / (self.ldls_mdls*2) 
                csl_count_original = 2 * mdl_to_bl_pitch
                pumps = self.pumps_per_atom()
                csl_count = csl_count_original
                csl_to_bl_pitch = (self.swd_width(tech) / tech.pitch_bl) / csl_count # convert SWD width in um to BLs first

                mdl_pitch_ratio = mdl_to_bl_pitch / _mdl_to_bl_pitch
                csl_pitch_ratio = csl_to_bl_pitch / _csl_to_bl_pitch

            else:
                # MDLs over SWD, CSLs over MAT
                mdl_to_bl_pitch = (self.swd_width(tech) / tech.pitch_bl) / (self.ldls_mdls*2)
                csl_to_bl_pitch = self.ldls_mdls

                mdl_pitch_ratio = mdl_to_bl_pitch / _mdl_to_bl_pitch
                csl_pitch_ratio = csl_to_bl_pitch / _csl_to_bl_pitch

        else:
            # Default: mdl_csl_over_mat
            if self.csl_mdl_shared_layer:
                # CSL and MDL share the same metal layer over MAT
                # Total wires = MDL (differential) + CSL
                mdl_wires = self.ldls_mdls * 2  # differential MDL pairs
                csl_wires = self.mat_cols / self.ldls_mdls  # CSL count per MAT
                total_wires = mdl_wires + csl_wires
                shared_pitch = self.mat_cols / total_wires  # BL pitches per wire

                mdl_to_bl_pitch = shared_pitch
                csl_to_bl_pitch = shared_pitch

            else:
                # Separate metal layers — each gets full MAT width
                mdl_to_bl_pitch = self.mat_cols / (self.ldls_mdls*2)
                csl_to_bl_pitch = self.ldls_mdls

            mdl_pitch_ratio = mdl_to_bl_pitch / _mdl_to_bl_pitch
            csl_pitch_ratio = csl_to_bl_pitch / _csl_to_bl_pitch

        if self.mdl_over_mat and (mdl_to_bl_pitch < effective_mdl_to_bl_min or mdl_pitch_ratio < 1):
            # too close of a pitch could impact reliability due to leak between MDLs
            #discard
            mdl_pitch_ratio = 0.01 # trigger the filter in dreamram.py

        return csl_pitch_ratio, mdl_pitch_ratio
    

    def bgbus_pitch_ratio(self, tech:Tech):
        # pitch ratio relative to baseline

        if self._die_y == -1:
            # baseline
            return 1

        bank_x, bank_y, cell_area = self.bank_dims(tech)

        # bgbus width with ECC
        n_bgbus = np.ceil(self.bgbus_width() * self.md_ecc_factor())

        pitch = bank_x / (n_bgbus * self.ind_pages() / (self.pages_per_bgbus_mux*self.mdl_bgbus_sd) * self.vert_bg/2 * np.pow(self.dbi_factor, self.dbi_for_bgbus))
        _pitch = self._bank_x_calc(tech) / (self._n_bgbus * 1.0 / (self._pages_per_bgbus_mux*self._mdl_bgbus_sd) * self._vert_bg/2 * np.pow(self.dbi_factor, self.dbi_for_bgbus))
        
        # return ratio
        return pitch/_pitch


    def mdl_csl_driver_time(self, tech:Tech):
        # minimum timing for CSL, LDL, MDL, and precharging
        return 1 #ns


    def _mdl_csl_driver_time(self, tech:Tech):
        return 1 #ns
    
    
    def core_tck(self, tech:Tech):
        # ns
        mat_width = self.swd_width(tech) + self.mat_cols * self.isolation_cols_overhead * tech.pitch_bl
        _mat_width = tech.scaled_swd_width() + self._mat_cols * self.isolation_cols_overhead * tech.pitch_bl
        bank_x, bank_y, cell_area = self.bank_dims(tech)

        # calculate baseline
        _bank_y =  self._bank_y_calc(tech)

        # calculate rc ratios
        csl_pitch_ratio, mdl_pitch_ratio = self.csl_mdl_pitch_ratios(tech)

        # Fixed-width: only spacing changes between wires, wire width stays at metal minimum
        rc_ratio_csl = tech.rc_ratio_fixed_width(csl_pitch_ratio, tech._c_within_layer_sparse_csl_mdl)
        rc_ratio_mdl = tech.rc_ratio_fixed_width(mdl_pitch_ratio, tech._c_within_layer_sparse_csl_mdl)

        res_ratio = tech.pitch_bl / tech.pitch_wl # estimate resistance ratio
        assert(res_ratio > 0)

        # Get constant driver and minimum times
        driver_time = self.mdl_csl_driver_time(tech)

        c_csl = tech.scale_cap(tech._c_csl)
        c_mdl = tech.scale_cap(tech._c_mdl)
        c_ldl = tech.scale_cap(tech._c_ldl)

        # baseline
        # scale as sum over R_i C_i l_i
        # critical path is CSL, LDL, MDL rise, LDL precharge
        denom = (c_csl + 2*c_mdl) * _bank_y + (res_ratio * c_ldl) * _mat_width

        # this design
        # MDL-over-MAT is already taken into account by ratio
        # csl_pitch_ratio for MDL-over-MAT describes the CSL pitch compared to the baseline MDL pitch
        numer = (rc_ratio_csl*c_csl + 2*rc_ratio_mdl*c_mdl) * bank_y + (res_ratio * c_ldl) * mat_width
        
        # add constant driver time and scale against baseline tck
        core_tck = driver_time + (self._tck - driver_time) * numer / denom

        return core_tck


    def bandwidth(self, tech:Tech):
        # GB/s
        # no ECC
        core_tck = self.core_tck(tech) # ns
        dqs = self.dq_count()
        dq_speed_factor = self.dq_speed_factor()

        return dqs * dq_speed_factor / (core_tck * self.md_ecc_factor() * 8) #8 bits per byte
    

    def atom_time(self, tech):
        # window for atom on DQs (ns)
        return self.atom_size * self.md_ecc_factor() * self.core_tck(tech) / (self.dq_speed_factor() * (self.dq_count() / (self.pch * self.channels * self.ind_pages())))


    def blsa_deltav(self, tech:Tech):
        # BLSA Delta V = CS amount / VCORE (dimensionless)

        # no need to scale the cap itself, this is all relative
        bitlines = tech._c_bl_per_cell * self.mat_rows
        
        deltav = 0
        if self.brv_sa:
            deltav = (tech._c_cell/2)/(tech._c_cell + bitlines) #bls separated from sensing nodes during CS
            deltav *= tech._brvsa_brv_deltav_boost # wider deltav from BRV
        else:
            deltav = (tech._c_cell/2)/(tech._c_cell + bitlines + tech._c_blsa)
    
        return deltav
    

    def rc_fall_to(self, p):
        # fall time in time constants from 1 to p proportion
        return -np.log(p)


    def rc_rise_to(self, p):
        # rise time in time constants from 0 to p proportion
        return -np.log(1-p)


    def tcl(self, tech:Tech):
        if self._die_y == -1: 
            # baseline cannot calculate on itself, take baseline value
            return self._tcl
        # ns
        # time from CAS to first data out
        _dies_stacked = self._sids * self._channels / self._ch_per_die
        dies_stacked, die_width, non_tsv_y, cmd_tsv_y, data_tsv_y, other_tsv_y = self.calc_stack_dims(tech)
        die_y = non_tsv_y + cmd_tsv_y + data_tsv_y + other_tsv_y
        _die_y = self._die_y
        bank_x, bank_y, cell_area = self.bank_dims(tech)
        _bank_y = self._bank_y_calc(tech)
        bgbus_pitch_ratio = self.bgbus_pitch_ratio(tech)
        bgbus_cap_ratio = tech.c_ratio(bgbus_pitch_ratio, tech._c_within_layer_sparse)
        base_cap_ratio = tech.c_ratio(bgbus_pitch_ratio * (self._bgbus_gbus_sd * self._gbus_tsv_sd)/(self.bgbus_gbus_sd * self.gbus_tsv_sd), tech._c_within_layer_sparse)

        _t_die_y = (self._tcl - 2 * _dies_stacked * self.rc_rise_to(0.8) * tech._r_load * tech.scaled_cap_tsv()/1000) / 2 # assume baseline uses same TSVs
        tcl = (base_cap_ratio + bgbus_cap_ratio) * _t_die_y * (die_y-bank_y)/(_die_y-_bank_y) + 2 * dies_stacked * self.rc_rise_to(0.8) * tech._r_load * tech.scaled_cap_tsv()/1000
        assert(tcl > 0)
        return tcl

    
    def trcd(self, tech:Tech, for_trp=False):
        # ns
        # time from signal assertion to blsa amplification

        # separate signal and blsa
        tsig = tech._trcd_signal
        tblsa = -tsig
        if self.brv_sa and not for_trp:
            tblsa += tech._trcd_brvsa
        else:
            tblsa += tech._trcd
        # scale signal by length
        bank_x, bank_y, cell_area = self.bank_dims(tech)
        _bank_x = self._bank_x_calc(tech)
        tsig_scaled = tsig * ( bank_x / _bank_x)

        # scale blsa by total capacitance involved
        # no need to scale the cap itself, this is all relative
        bitlines = tech._c_bl_per_cell * self.mat_rows * self.isolation_rows_overhead
        _bitlines = tech._c_bl_per_cell * tech._mat_rows_ref * self.isolation_rows_overhead
        tblsa_scaled = tblsa
        if self.brv_sa and not for_trp:
            # CS scales without BLSA, since in brvsa, bitlines are separated from sensing nodes during CS
            # MS scales with whole capacitance
            p = tech._brvsa_cs_proportion
            tblsa_scaled *= (p * (tech._c_cell + bitlines) / (tech._c_cell + _bitlines)) + ((1-p) * (tech._c_cell + bitlines + tech._c_blsa) / (tech._c_cell + _bitlines + tech._c_blsa))
        else:
            tblsa_scaled *= (tech._c_cell + bitlines + tech._c_blsa) / (tech._c_cell + _bitlines + tech._c_blsa)

        return tsig_scaled+tblsa_scaled
    

    def trp(self, tech:Tech):
        # ns
        # ASSUME trp ~= trcd
        return self.trcd(tech, for_trp=True)


    def peri_tck(self, tech: Tech):
        # ns — DRAM peripheral clock period
        # nCCDL = 4 peripheral cycles per core cycle 
        return self.core_tck(tech) / 4.0


    def trcdwr(self, tech: Tech):
        # ns — time from ACT to WRITE command
        # Same wordline activation as tRCDRD, but write driver overdrives SA
        # before full bitline sensing completes
        # Signal propagation (identical to tRCDRD — wordline RC is the same)
        tsig = tech._trcd_signal
        tblsa = -tsig
        if self.brv_sa:
            tblsa += tech._trcd_brvsa
        else:
            tblsa += tech._trcd

        # Scale signal by bank width (same as trcd)
        bank_x, bank_y, cell_area = self.bank_dims(tech)
        _bank_x = self._bank_x_calc(tech)
        tsig_scaled = tsig * (bank_x / _bank_x)

        # Scale BLSA by capacitance (same as trcd)
        bitlines = tech._c_bl_per_cell * self.mat_rows * self.isolation_rows_overhead
        _bitlines = tech._c_bl_per_cell * tech._mat_rows_ref * self.isolation_rows_overhead
        if self.brv_sa:
            p = tech._brvsa_cs_proportion
            tblsa_scaled = tblsa * (
                p * (tech._c_cell + bitlines) / (tech._c_cell + _bitlines) +
                (1-p) * (tech._c_cell + bitlines + tech._c_blsa) / (tech._c_cell + _bitlines + tech._c_blsa))
        else:
            tblsa_scaled = tblsa * (
                (tech._c_cell + bitlines + tech._c_blsa) / (tech._c_cell + _bitlines + tech._c_blsa))

        # Write driver only needs partial SA development
        return round(tsig_scaled + tblsa_scaled * tech._trcdwr_blsa_fraction,2) #round to 2 decimals to avoid awkward tails


    def tras(self, tech: Tech):
        # ns — minimum time row must remain active
        # Components: (1) full tRCD for sensing, (2) restore time for SA to
        # drive charge back into cell through access transistor

        trcd_val = self.trcd(tech)

        # Restore time scales with bitline + cell capacitance
        bitlines = tech._c_bl_per_cell * self.mat_rows * self.isolation_rows_overhead
        _bitlines = tech._c_bl_per_cell * tech._mat_rows_ref * self.isolation_rows_overhead
        cap_ratio = (tech._c_cell + bitlines) / (tech._c_cell + _bitlines)
        t_restore = tech._tras_restore * cap_ratio

        return trcd_val + t_restore


    def trc(self, tech: Tech):
        # ns — minimum time between ACTs to same bank
        # Physically: activate row + use it + precharge = tRAS + tRP
        return self.tras(tech) + self.trp(tech)


    def trrds(self, tech: Tech):
        # ns — minimum time between ACTs to different banks (different bank group)
        # Row command path (address bus, row decoder, wordline drivers) must settle
        # before the next ACT. Scales with bank_y (row command path length).
        bank_x, bank_y, cell_area = self.bank_dims(tech)
        _bank_y = self._bank_y_calc(tech)
        return tech._trrds * (bank_y / _bank_y)


    def trrdl(self, tech: Tech):
        # ns — minimum time between ACTs to different banks (same bank group)
        # Same row command path constraint as tRRDS, but banks in the same bank group
        # share additional local decode logic, requiring similar or more settling time.
        bank_x, bank_y, cell_area = self.bank_dims(tech)
        _bank_y = self._bank_y_calc(tech)
        return tech._trrdl * (bank_y / _bank_y)


    def tfaw(self, tech: Tech):
        # ns — rolling window limiting 4 activations
        # Constrains ACT command rate to avoid row command path contention
        return max(4.0 * self.trrds(tech), 4.0 * self.trrdl(tech))


    def trtp(self, tech: Tech):
        # TODO: Need modification
        # ns — minimum time from READ command to PRE command
        # Column data pipeline must fully drain before precharge
        # Fixed at 5 peripheral clock cycles (pipeline depth: CSL→LDL→MDL→BGBUS→latch)
        return 5.0 * self.peri_tck(tech) * self.pumps_per_atom()


    def tccdl(self, tech: Tech):
        # TODO: Need modification
        # ns — minimum time between column commands to different banks, same bank group
        # Fixed at 4 nCK 
        return 4.0 * self.peri_tck(tech) * self.pumps_per_atom()


    def tccds(self, tech: Tech):
        # TODO: Need modification
        # ns — minimum time between column commands to different banks, different bank group
        # Fixed at 2 nCK 
        return 2.0 * self.peri_tck(tech) * self.pumps_per_atom()


    def twr(self, tech: Tech):
        # ns — time after last write data before precharge can begin
        # Cell must be fully charged/discharged after write driver overwrites SA

        # Signal component (wordline must stay active)
        tsig = tech._trcd_signal
        bank_x, bank_y, cell_area = self.bank_dims(tech)
        _bank_x = self._bank_x_calc(tech)
        tsig_scaled = tsig * (bank_x / _bank_x)

        # Restore component scales with capacitance (same physics as tRAS restore)
        bitlines = tech._c_bl_per_cell * self.mat_rows * self.isolation_rows_overhead
        _bitlines = tech._c_bl_per_cell * tech._mat_rows_ref * self.isolation_rows_overhead
        cap_ratio = (tech._c_cell + bitlines) / (tech._c_cell + _bitlines)
        t_restore = tech._twr_restore * cap_ratio

        return tsig_scaled + t_restore


    def total_area(self, tech:Tech):
        # mm^2
        dies_stacked, die_width, non_tsv_y, cmd_tsv_y, data_tsv_y, other_tsv_y = self.calc_stack_dims(tech)
        return (dies_stacked+1) * die_width * (non_tsv_y+cmd_tsv_y+data_tsv_y+other_tsv_y) / 1000 / 1000
    

    def cell_efficiency(self, tech:Tech):
        # cell area / die area

        # cells
        bank_x, bank_y, cell_area = self.bank_dims(tech) # cell area per bank
        cell_area_total = self.ch_per_die * self.pch * self.vert_bg * self.horiz_bg * self.banks * cell_area

        # die
        dies_stacked, die_width, non_tsv_y, cmd_tsv_y, data_tsv_y, other_tsv_y = self.calc_stack_dims(tech)
        die_area = (non_tsv_y+cmd_tsv_y + data_tsv_y + other_tsv_y) * die_width

        return cell_area_total / die_area
    

    def cell_efficiency_mat(self, tech:Tech):
        # cell area / MAT area

        cells_width = (self.mat_cols * self.isolation_cols_overhead) * tech.pitch_bl
        cells_height = (self.mat_rows * self.isolation_rows_overhead) * tech.pitch_wl

        blsa_height = self.blsa_height(tech)
        swd_width = self.swd_width(tech)

        return cells_width * cells_height / ((cells_width+swd_width)*(cells_height+blsa_height))


    def per_cmd_energy(self, tech:Tech):
        # pJ
        # energy to use a specific wire set during one command's worth of use. ASSUME worst case 100% activity from average location, similar to JEDEC IDDs

        # Track energy on the power supply (IDD#/IPP#)
        # E = a * n * ((C/L)*L) * V_int * V_ext

        e = {}

        n = self.wire_counts() # counts per command
        l = self.wire_lengths(tech)
        row, col = self.ch_cmd_bits()

        mdl_pumps = self.pumps_per_atom()
        tsv_pumps = mdl_pumps

        bgbus_pitch_ratio = self.bgbus_pitch_ratio(tech)
        bgbus_cap_ratio = tech.c_ratio(bgbus_pitch_ratio, tech._c_within_layer_sparse)
        base_cap_ratio = tech.c_ratio(bgbus_pitch_ratio * (self._bgbus_gbus_sd * self._gbus_tsv_sd)/(self.bgbus_gbus_sd * self.gbus_tsv_sd), tech._c_within_layer_sparse)

        vcorevdd = tech.vcore_int * tech.vdd
        vppvpp = tech.vpp_int * tech.vpp / tech.vpp_eff
        vddqlvdd = tech.vddql_int * tech.vdd
        vddqlvddql = tech.vddql_int * tech.vddql_int
        vperivperi_ldl = tech.vperi_ldl * tech.vperi_ldl # LDL voltage domain
        vddvdd = tech.vdd * tech.vdd # MDL voltage domain

        bank_clks_per_atom = self.bank_clks_per_atom()

        ras_edges = 3 # 3 edges for RAS
        cas_edges = 2 # 2 edges for CAS

        # capacitance ratios of CSL and MDL
        csl_pitch_ratio, mdl_pitch_ratio = self.csl_mdl_pitch_ratios(tech)
        c_ratio_csl = tech.rc_ratio_fixed_width(csl_pitch_ratio, tech._c_within_layer_sparse_csl_mdl)
        c_ratio_mdl = tech.rc_ratio_fixed_width(mdl_pitch_ratio, tech._c_within_layer_sparse_csl_mdl) # fixed-width: wire width stays at metal min, only spacing changes

        # row
        e["row"] = ras_edges * n["row"]/2 * l["od-row-avg"] * tech.scale_cap(tech._c_ca) * vcorevdd  # assume data and cmd buses go to 1 half the time
        e["mwl"] = n["mwl"] * l["mwl"] * tech.scale_cap(tech._c_mwl) * vppvpp #changed number of mwl
        e["lwl"] = n["lwl"] * l["lwl"] * tech.scale_cap(tech._c_lwl) * vppvpp
        # heat of ACT/PRE
        e["heat-act-mwl"] = 0.347 * e["mwl"] # 0.5 is when V_int = V_ext, 
        e["heat-pre-mwl"] = 0.653 * e["mwl"] # this is becuase during charging more energy is required from V_ext
        e["heat-act-lwl"] = 0.653 * e["lwl"]
        e["heat-pre-lwl"] = 0.347 * e["lwl"]

        # bitlines
        cap_bitline = tech.scale_cap_blsa(tech._c_bl_per_cell*self.mat_rows*self.isolation_rows_overhead + 2*tech.cell_leak*tech._c_cell + tech._c_blsa)

        ncv2 = n["bl"] * cap_bitline * vcorevdd
        e["bl-act"] = 0.5 * ncv2 # half-swing from VDD/2 to rail 
        e["bl-pre"] = tech.bl_pre_beta * ncv2 # non-ideal charge-sharing loss 
        # heat of ACT/PRE
        e["heat-act-bl"] = 0.25 * ncv2 # IR losses in SA during sensing
        e["heat-pre-bl"] = (0.25 + tech.bl_pre_beta) * ncv2 # 0.25 intrinsic equalization heat + non-ideal loss

        # col
        e["col"] = cas_edges * n["col"]/2 * l["od-col-avg"] * tech.scale_cap(tech._c_ca) * vddvdd # differential but still full swing  # assume data and cmd buses go to 1 half the time
        e["csl"] = (bank_clks_per_atom * n["csl"]) * l["csl"] * (tech.scale_cap(tech._c_csl)*c_ratio_csl) * vddvdd
        e["ldl"] = bank_clks_per_atom * n["ldl"] * l["ldl"] * tech.scale_cap(tech._c_ldl) * vperivperi_ldl/2 # swing from vpre_ldl to vperi
        e["mdl"] = bank_clks_per_atom * n["mdl"] * l["mdl"] * (tech.scale_cap(tech._c_mdl)*c_ratio_mdl) * vddvdd/2 # half swing as complementary will not swing, precharge at vdd
        e["bgbus+gbus"] = self.bgbus_clks_per_atom() * n["bgbus"]/2 * l["bgbus+gbus"] * tech.scale_cap(tech._c_bus) * bgbus_cap_ratio * vddvdd * np.pow(self.dbi_transition_factor_avg(),self.dbi_for_bgbus) # data and cmd buses go to 1 half the time  # databus inversion decreases avg frequency of transition

        e["dq"] = (n["dq"] * mdl_pumps * self.tsv_dq_sd)/2 * tech.c_dq * vddqlvdd # DQ output at VDDQL swing, pre-driver/serializer at VDD; pins toggle tsv_dq_sd times per bank pump
        e["row-dq"] = ras_edges * row/2 * tech.c_ca_ra_pin * vddvdd # DRAM RX input capacitance for row CA signals
        e["col-dq"] = cas_edges * col/2 * tech.c_ca_ra_pin * vddvdd # DRAM RX input capacitance for col CA signals
        e["tsv"] = (n["tsv"] * tsv_pumps)/2 * l["tsv-n-layers-avg"] * tech.scaled_cap_tsv() * vddvdd * self.dbi_transition_factor_avg() # data and cmd buses go to 1 half the time  # databus inversion decreases avg frequency of transition
        e["row-tsv"] = ras_edges * row/2 * l["tsv-n-layers-avg"] * tech.scaled_cap_tsv() * vddvdd # data and cmd buses go to 1 half the time
        e["col-tsv"] = cas_edges * col/2 * l["tsv-n-layers-avg"] * tech.scaled_cap_tsv() * vddvdd # data and cmd buses go to 1 half the time
        e["base"] = (n["tsv"] * tsv_pumps)/2 * l["base"] * tech.scale_cap(tech._c_bus) * base_cap_ratio * vddvdd * self.dbi_transition_factor_avg() # data and cmd buses go to 1 half the time  # databus inversion decreases avg frequency of transition
        e["row-base"] = ras_edges * row/2 * l["base"] * tech.scale_cap(tech._c_ca) * vddvdd # data and cmd buses go to 1 half the time
        e["col-base"] = cas_edges * col/2 * l["base"] * tech.scale_cap(tech._c_ca) * vddvdd # data and cmd buses go to 1 half the time

        cmd_e = {}
        cmd_e["pre"] = e["row-dq"] + e["row-base"] + e["row-tsv"] + e["row"] + e["bl-pre"] # MWL/LWL fall during PRE (discharge, no supply draw)
        cmd_e["act"] = e["row-dq"] + e["row-base"] + e["row-tsv"] + e["row"] + e["mwl"] + e["lwl"] + e["bl-act"] # MWL/LWL both rise during ACT (charge, supply draw)
        cmd_e["rd"]  = e["col-dq"] + e["col-base"] + e["col-tsv"] + e["col"] + e["csl"] + e["ldl"] + e["mdl"] + e["bgbus+gbus"] + e["tsv"] + e["base"] + e["dq"]
        
        cmd_e["heat-pre"] = e["row-dq"] + e["row-base"] + e["row-tsv"] + e["row"] + e["heat-pre-mwl"] + e["heat-pre-lwl"] + e["heat-pre-bl"]
        cmd_e["heat-act"] = e["row-dq"] + e["row-base"] + e["row-tsv"] + e["row"] + e["heat-act-mwl"] + e["heat-act-lwl"] + e["heat-act-bl"]

        return cmd_e, e
