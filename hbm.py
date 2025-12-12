import numpy as np
from dataclasses import dataclass

from tech import Tech

@dataclass
class Hbm:
    
    # Organization
    ranks: int = 2
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
    mdl_over_mat: int = 0 # y/n
    salp_groups: int = 1 # incompatible with salp_all
    salp_all: int = 0 # incompatible with salp_groups
    ldls_mdls: int = 8 # non-differential datalines per MAT
    atom_size: int = 256 # bits
    brv_sa: int = 0

    # Data I/O Hierarchy
    pages_per_bgbus_mux: int = 4 # independent pages (virtual banks) per "bgbus"
    mdl_bgbus_sd: float = 1 # serdes
    bgbuses_per_gbus: int = 2 # sets number of gbuses
    bgbus_gbus_sd: float = 1
    gbuses_out: int = 2 # per pch # INSTEAD USE gbuses()
    gbus_tsv_sd: float = 4
    tsv_dq_sd: float = 4
    dbi_factor: float = 1.125 # 1 bit per byte
    dbi_for_bgbus: int = 1

    # Reference
    # baseline values for tCK and other timing calculations
    _mat_rows: float = 512
    _mat_cols: float = 512
    _subarrays: float = 16
    _ldls_mdls: float = 8
    _mats: float = 32
    _subchannels: float = 1
    _vert_bg: float = 4 # REPLACECONSTANT: parse does not update this value to reflect baseline
    _n_bgbus: float = 256 # REPLACECONSTANT:
    _pages_per_bgbus_mux: float = 4 # REPLACECONSTANT:
    _mdl_bgbus_sd: float = 1 # REPLACECONSTANT:
    _bgbus_gbus_sd: float = 1 # REPLACECONSTANT:
    _gbus_tsv_sd: float = 4 #REPLACECONSTANT:
    _ranks: float = 2
    _channels: float = 16
    _ch_per_die: float = 4
    _tck: float = 2 #ns
    _die_y: float = 11000 #um # fixed for reference (REPLACECONSTANT:?)
    _tcl: float = 38.4 #ns


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


    ''' COMMAND WIDTHS '''

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

    def ch_cmd_bits(self):
        # command bits per channel
        # ASSUME some small overhead (4 and 5) of other bits to specify more specific commands, and for parity
        ind_row_pages = self.ind_row_pages()
        ind_col_pages = self.ind_pages()

        # allow non-powers-of-2 by rounding up number of decode bits
        row = 4 + np.log2(self.ranks*self.pch*self.vert_bg*self.horiz_bg*self.banks) + np.ceil(np.log2(self.subarrays)) + np.ceil(np.log2(self.mat_rows)) + np.log2(ind_row_pages)
        col = 5 + np.log2(self.ranks*self.pch*self.vert_bg*self.horiz_bg*self.banks) + np.ceil(np.log2(self.atoms_per_page())) + np.log2(ind_col_pages)

        row_int = int(row)
        col_int = int(col)
        assert(row_int == row)
        assert(col_int == col)
        return row_int, col_int


    ''' STACK DIMENSIONS '''

    def decoder_dims(self, tech:Tech):
        #um
        return tech.scaled_rowdec_width(), tech.scaled_coldec_height()


    def swd_width(self, tech:Tech):
        # um
        # start with scaled reference
        ref_w = tech.scaled_swd_width()
        w = 0.0
        
        if self.mdl_over_mat:
            # scale with CSLs
            csl_count_original = self.mat_cols/self.ldls_mdls
            pumps = self.pumps_per_atom()
            csl_count = pumps + np.log2(csl_count_original / pumps) # one-hot one per pump, rest binary

            w = (csl_count+2) * tech.pitch_mdl # 2 fx
            w = np.max([ref_w, w])

        else:
            # scale with MDLs
            w = ((self.ldls_mdls * 2.0) + 2) * tech.pitch_mdl # differential # 2 fx
            w = np.max([ref_w, w])
            
        assert(w>0)
        return w


    def blsa_height(self, tech:Tech):
        # um
        # scaled reference
        y = tech.scaled_blsa_height()
 
        if self.brv_sa:
            # estimate overhead of OC/ISO with same N:P
            y = y * tech._brvsa_height_ratio

        if self.mdl_over_mat:
            # ASSUME increased height from extra driver
            # ASSUME CSLs replace LDLs, with additional routing space over driver
            y = y * tech._mdl_over_mat_height_ratio
        else: 
            # if not MDL-over-MAT, scale if more than 8 ldls per MAT
            # does not apply to MDL-over-MAT, since there are no LDLs
            # ASSUME originally 4 LDLs/BLSA, 8 LDLs/MAT, differential
            y += (self.ldls_mdls - 8) * tech.pitch_bl

        assert(y>0)
        return y


    def bank_dims(self, tech:Tech):
        # um
        # MATs, subarrays, and column/row decoders

        # col/row dec
        row_dec_width, col_dec_y = self.decoder_dims(tech)

        # SWDs
        swd_width = self.swd_width(tech)
        # ASSUME ignore ECC and subchannels interaction
        swd_width_total = swd_width * (self.mats + self.subchannels + 1) * self.ecc_factor() # +1 added
        # BLSAs
        blsa_y = self.blsa_height(tech)
        # one more than the number of subarrays 
        blsa_y_total = blsa_y * ((self.subarrays + (self.salp_groups-1) + (self.repair_subarray) + 1) + 1)
        
        # all cell arrays
        cells_width_total = (self.mats * self.ecc_factor()) * (self.mat_cols * self.isolation_cols_overhead) * tech.pitch_bl
        
        # subarrays: +1 for open bitline scheme, insert a blank subarray between salp groups
        cells_y_total = (self.subarrays + (self.salp_groups-1) + (self.repair_subarray) + 1) * (self.mat_rows * self.isolation_rows_overhead) * tech.pitch_wl

        cell_area = cells_width_total * cells_y_total

        return row_dec_width+cells_width_total+swd_width_total, col_dec_y+cells_y_total+blsa_y_total, cell_area
    

    def _bank_x(self, tech:Tech):
        # um
        row_dec_width, col_dec_y = self.decoder_dims(tech) # if these scale with config, update this function
        swd_width = self.swd_width(tech)
        swd_width_total = swd_width * (self._mats + self._subchannels + 1) * self.ecc_factor()
        cells_width_total = (self._mats * self.ecc_factor()) * (self._mat_cols * self.isolation_cols_overhead) * tech.pitch_bl

        return row_dec_width+cells_width_total+swd_width_total
    

    def _bank_y(self, tech:Tech):
        # um
        return (tech.scaled_blsa_height() + self._mat_rows * self.isolation_rows_overhead * tech.pitch_wl) * (self._subarrays + 1) + tech.scaled_blsa_height() + tech.scaled_coldec_height()


    def od_ecc_height(self, tech:Tech):
        # OD-ECC and GBUS mux happen in the space between TSVs and banks
        return (0.75*self._bank_y(tech)) # ASSUME from die shot


    def bankdie_dims(self, tech:Tech):
        # all lengths in um
        bank_x, bank_y, cell_area = self.bank_dims(tech)
        od_ecc_height = self.od_ecc_height(tech)
        row_dec_width, col_dec_height = self.decoder_dims(tech)
        bank_dec_height = 2*row_dec_width # ASSUME from die shot

        x = self.ch_per_die * self.pch * self.horiz_bg * (bank_x + bank_dec_height)
        y = (self.vert_bg * self.banks * bank_y) + od_ecc_height
        
        return x, y


    def calc_stack_dims(self, tech:Tech):
        # um
        # dies stacked
        dies_stacked = self.ranks * self.channels / self.ch_per_die
        assert(dies_stacked == int(dies_stacked))
        dies_stacked = int(dies_stacked)
        
        # bank-area dims
        die_width, non_tsv_y = self.bankdie_dims(tech)

        # TSV heights
        row, col = self.ch_cmd_bits()
        cmd_tsv_count = self.channels * (row+col)
        
        data_tsv_count = self.channels * self.pch * (self.ldls_mdls * self.mats * self.md_ecc_factor()) * self.gbuses()
        data_tsv_count = data_tsv_count / (self.mdl_bgbus_sd * self.bgbus_gbus_sd * self.gbus_tsv_sd)

        # ASSUME TSVs must keep KOZ away from die edge
        height_per_tsv = 1/ ((die_width - 2 * tech.tsv_koz) * tech.tsv_density())
        assert(height_per_tsv >= 0)
        cmd_tsv_y = cmd_tsv_count * height_per_tsv
        data_tsv_y = data_tsv_count * height_per_tsv
        other_tsv_y = 2 * (np.sqrt(3)/2) * tech.tsv_pitch #ASSUME two rows of power TSVs, hexagonally
        other_tsv_y += 2 * tech.tsv_koz

        return dies_stacked, die_width, non_tsv_y, cmd_tsv_y, data_tsv_y, other_tsv_y


    ''' PAGES '''

    def capacity(self):
        # GB
        # no ECC
        capacity = self.ranks * self.channels * self.pch * self.vert_bg * self.horiz_bg * self.banks * self.subarrays * self.mat_rows * self.mats * self.mat_cols / np.power(2,10) / np.power(2,10) / np.power(2,10) / np.power(2,3)
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
        return self.mdl_width_per_page() / self.mdl_bgbus_sd


    def gbus_width(self):
        # REMINDER: for energy, include MD-ECC
        return self.bgbus_width() / self.bgbus_gbus_sd


    def bank_clks_per_atom(self): 
        # window nCK
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
        # multiply by core freq for true speed
        # higher = faster
        return self.mdl_bgbus_sd * self.bgbus_gbus_sd * self.gbus_tsv_sd * self.tsv_dq_sd


    def gbuses(self):
        # number of gbuses per pch
        total_pages = self.ind_pages() * self.banks * self.vert_bg * self.horiz_bg
        a = total_pages / (self.pages_per_bgbus_mux * self.bgbuses_per_gbus)
        assert(a == int(a) or a<1) # catch a<2 in main.py (ADL)
        return a
    
    
    def dq_count(self):
        x = np.ceil(self.md_ecc_factor() * self.gbus_width() * self.gbuses() / (self.gbus_tsv_sd * self.tsv_dq_sd) ) * self.pch * self.channels 
        assert(x == int(x))
        return int(x)


    def wire_lengths(self, tech:Tech):

        l = {}
        
        dies_stacked, die_width, non_tsv_y, cmd_tsv_y, data_tsv_y, other_tsv_y = self.calc_stack_dims(tech)
        tsv_area_y = cmd_tsv_y + data_tsv_y + other_tsv_y
        bank_area_y = non_tsv_y
        bank_x, bank_y, cell_area = self.bank_dims(tech)
        swd_x = self.swd_width(tech)

        # row
        l["od-row-avg"] = tsv_area_y/2 + bank_area_y/4 # add TSV+base for full row cmd energy
        l["od-row-max"] = (tsv_area_y+bank_area_y)/2
        l["mwl"] = bank_x
        l["lwl"] = (self.mat_cols * self.isolation_cols_overhead) * tech.pitch_bl + swd_x/2
        l["bl"] = (self.mat_rows * self.isolation_rows_overhead) * tech.pitch_wl # do not include BLSA
        # col
        l["od-col-avg"] = tsv_area_y/2 + bank_area_y/4 + bank_x/2
        l["od-col-max"] = bank_x + (tsv_area_y+bank_area_y)/2 - bank_y/2
        l["csl"] = bank_y
        l["ldl"] = (self.mat_cols * self.isolation_cols_overhead) * tech.pitch_bl + swd_x/2
        l["mdl"] = bank_y
        l["bgbus+gbus"] = bank_area_y/2 # ASSUME group bgbus+gbus
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
        n["mwl"] = 1
        n["lwl"] = (self.mats * self.ecc_factor()) / self.ind_row_pages()
        n["bl"] = (self.mats * self.ecc_factor()) * (self.mat_cols)  / self.ind_row_pages()
        # per col cmd
        n["col"] = col
        n["csl"] = np.ceil((self.mats * self.ecc_factor()) / self.ind_pages()) # count 1 per MAT per cycle
        n["ldl"] = self.mdl_width_per_page() * self.ecc_factor()
        if self.mdl_over_mat:
            #repurpose ldls to mean the horizontal part of csl
            n["ldl"] = n["csl"]
        else:
            assert(n['ldl'] <= self.atom_size * self.ecc_factor())
        n["mdl"] = np.ceil(self.mdl_width_per_page() * self.ecc_factor())
        n["bgbus"] = np.ceil(self.bgbus_width() * self.md_ecc_factor()) 
        n["gbus"] = np.ceil(self.gbus_width() * self.md_ecc_factor())
        n["tsv"] = np.ceil(self.gbus_width() * self.md_ecc_factor() / self.gbus_tsv_sd)
        n["dq"] = np.ceil(self.dq_count() * 2 / (self.pch * self.channels * self.gbuses())) # md_ecc factored in by dq_count() # 2/gbuses #REPLACECONSTANT:?


        assert(n['mdl'] <= self.atom_size * self.ecc_factor())
        assert(n['bgbus'] <= self.atom_size * self.md_ecc_factor())
        assert(n['gbus'] <= self.atom_size * self.md_ecc_factor())
        assert(n['tsv'] <= self.atom_size * self.md_ecc_factor())

        return n


    def csl_mdl_pitch_ratios(self, tech:Tech):
        # Ensure the columns divide nicely over the MDLs/LDLs    
        assert(self.mat_cols % self.ldls_mdls == 0)
        
        if self.mdl_over_mat:
            # MDLs over MAT
            # CSLs over SWD            
            mdl_to_bl_pitch = self.mat_cols / (self.ldls_mdls*2) # differential
            csl_count_original = 2 * mdl_to_bl_pitch # = self.mat_cols / self.ldls_mdls
            pumps = self.pumps_per_atom()
            csl_count = pumps + np.log2(csl_count_original / pumps) # one-hot per pump rest binary

            # max(1, old_wire_count/new_wire_count)
            nonpump = np.log2(csl_count_original / pumps) + 2
            assert(2*self._ldls_mdls+2-nonpump > 0)
            csl_pitch_ratio = max([1, (2*self._ldls_mdls+2)/(csl_count)])
            mdl_pitch_ratio = mdl_to_bl_pitch / 8 # vs previous CSLs over MAT


        else:
            # Normal MAT
            # CSLs over MAT
            # MDLs over SWD
            csl_to_bl_pitch = self.ldls_mdls

            csl_pitch_ratio = csl_to_bl_pitch / 8
            mdl_pitch_ratio = max([1, (2*self._ldls_mdls+2)/(2*self.ldls_mdls+2) ])# 2 fx  # swd_width() in the dram file should have taken care of making the right swd width for overwide wires, hence 1

        return csl_pitch_ratio, mdl_pitch_ratio
    

    def bgbus_pitch_ratio(self, tech:Tech):
        # pitch ratio relative to baseline

        bank_x, bank_y, cell_area = self.bank_dims(tech)

        # number of bgbuses per pch
        n_bgbus = np.ceil(self.bgbus_width() * self.md_ecc_factor()) 

        pitch = bank_x / (n_bgbus * self.ind_pages() / (self.pages_per_bgbus_mux*self.mdl_bgbus_sd) * self.vert_bg/2 * np.pow(self.dbi_factor, self.dbi_for_bgbus))
        _pitch = self._bank_x(tech) / (self._n_bgbus * self.md_ecc_factor() * 1 / (self._pages_per_bgbus_mux*self._mdl_bgbus_sd) * self._vert_bg/2 * np.pow(self.dbi_factor, self.dbi_for_bgbus))
        
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
        _bank_y =  self._bank_y(tech)

        # calculate rc ratios
        csl_pitch_ratio, mdl_pitch_ratio = self.csl_mdl_pitch_ratios(tech)
        if self.mdl_over_mat:
            rc_ratio_csl = tech.rc_ratio(csl_pitch_ratio, tech._c_within_layer) # compare to 4blp MDLs
            rc_ratio_mdl = tech.rc_ratio(mdl_pitch_ratio, tech._c_within_layer_sparse) # compare to 8blp CSLs
        else:
            rc_ratio_csl = tech.rc_ratio(csl_pitch_ratio, tech._c_within_layer_sparse) # compare to 8blp CSLs
            rc_ratio_mdl = tech.rc_ratio(mdl_pitch_ratio, tech._c_within_layer) # compare to 4blp MDLs

        res_ratio = tech.pitch_mdl / tech.pitch_ldl # estimate resistance ratio as inverse of wire width ratio
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
        
        # add constant driver time
        return driver_time + (self._tck - driver_time) * numer / denom # scale against 2ns baseline


    def bandwidth(self, tech:Tech):
        # GB/s
        # no ECC
        core_tck = self.core_tck(tech) # ns
        dqs = self.dq_count()
        dq_speed_factor = self.dq_speed_factor()

        return dqs * dq_speed_factor / (core_tck * self.md_ecc_factor() * 8) #8 bits per byte
    

    def atom_time(self, tech):
        # window for atom on DQs (ns)
        return self.atom_size * self.md_ecc_factor() * self.core_tck(tech) / (self.dq_speed_factor() * (self.dq_count() / (self.pch * self.channels)))
    

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
        # ns
        # time from CAS to first data out
        _dies_stacked = self._ranks * self._channels / self._ch_per_die
        dies_stacked, die_width, non_tsv_y, cmd_tsv_y, data_tsv_y, other_tsv_y = self.calc_stack_dims(tech)
        die_y = non_tsv_y + cmd_tsv_y + data_tsv_y + other_tsv_y
        _die_y = self._die_y
        bank_x, bank_y, cell_area = self.bank_dims(tech)
        _bank_y = self._bank_y(tech)
        bgbus_pitch_ratio = self.bgbus_pitch_ratio(tech)
        bgbus_cap_ratio = tech.c_ratio(bgbus_pitch_ratio, tech._c_within_layer_sparse)
        base_cap_ratio = tech.c_ratio(bgbus_pitch_ratio * (self._bgbus_gbus_sd * self._gbus_tsv_sd)/(self.bgbus_gbus_sd * self.gbus_tsv_sd), tech._c_within_layer_sparse)

        _t_die_y = (self._tcl - self._tck - 2 * _dies_stacked * self.rc_rise_to(0.8) * tech._r_load * tech.scaled_cap_tsv()/1000) / 2 # assume baseline uses same TSVs
        tcl = (base_cap_ratio + bgbus_cap_ratio) * _t_die_y * (die_y-bank_y)/(_die_y-_bank_y) + 2 * dies_stacked * self.rc_rise_to(0.8) * tech._r_load * tech.scaled_cap_tsv()/1000 + self.pumps_per_atom() * self.core_tck(tech)

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
        _bank_x = self._bank_x(tech)
        tsig_scaled = tsig * (_bank_x / bank_x)

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
        # from memory controller standpoint, almost always makes sense to align
        return self.trcd(tech, for_trp=True)


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

        # Track energy on the power supply (IDD#/IPP#). Hence, when raising a voltage:
        # E = a * n * ((C/L)*L) * V_int * V_ext
        # Half stores in capacitor; the other half dissipates
        # When lowering, pull nothing; the half stored in the capacitor dissipates to ground

        e = {}

        n = self.wire_counts() # counts per command
        l = self.wire_lengths(tech)
        row, col = self.ch_cmd_bits()

        mdl_pumps = self.pumps_per_atom()
        tsv_pumps = mdl_pumps #* self.gbus_tsv_sd * self.bgbus_gbus_sd * self.mdl_bgbus_sd

        bgbus_pitch_ratio = self.bgbus_pitch_ratio(tech)
        bgbus_cap_ratio = tech.c_ratio(bgbus_pitch_ratio, tech._c_within_layer_sparse)
        base_cap_ratio = tech.c_ratio(bgbus_pitch_ratio * (self._bgbus_gbus_sd * self._gbus_tsv_sd)/(self.bgbus_gbus_sd * self.gbus_tsv_sd), tech._c_within_layer_sparse)

        vcorevdd = tech.vcore_int * tech.vdd
        vppvpp = tech.vpp_int * tech.vpp / tech.vpp_eff
        vddqlvdd = tech.vddql_int * tech.vdd

        # MDL-over-MAT binary CSL part
        csl_count_original = self.mat_cols/self.ldls_mdls
        binary_csls = np.log2(csl_count_original / self.pumps_per_atom())
        
        # capacitance ratios of CSL and MDL
        csl_pitch_ratio, mdl_pitch_ratio = self.csl_mdl_pitch_ratios(tech)
        if self.mdl_over_mat:
            c_ratio_csl = tech.c_ratio(csl_pitch_ratio, tech._c_within_layer) # compare to 4blp MDLs
            c_ratio_mdl = tech.c_ratio(mdl_pitch_ratio, tech._c_within_layer_sparse) # compare to 8blp CSLs
        else:
            c_ratio_csl = tech.c_ratio(csl_pitch_ratio, tech._c_within_layer_sparse) # compare to 8blp CSLs
            c_ratio_mdl = tech.c_ratio(mdl_pitch_ratio, tech._c_within_layer) # compare to 4blp MDLs

        # row
        e["row"] = n["row"]/2 * l["od-row-avg"] * tech.scale_cap(tech._c_ca) * vcorevdd  # data and cmd buses go to 1 half the time
        e["mwl"] = n["mwl"] * l["mwl"] * tech.scale_cap(tech._c_mwl) * vppvpp
        e["lwl"] = n["lwl"] * l["lwl"] * tech.scale_cap(tech._c_lwl) * vppvpp
        # heat of ACT/PRE
        e["heat-act-mwl"] = 0.5 * e["mwl"]
        e["heat-pre-mwl"] = 0.5 * e["mwl"]
        e["heat-act-lwl"] = 0.5 * e["lwl"]
        e["heat-pre-lwl"] = 0.5 * e["lwl"]

        # bitlines
        cap_bitline = tech.scale_cap_blsa(tech._c_bl_per_cell*self.mat_rows*self.isolation_rows_overhead + 2*tech.cell_leak*tech._c_cell + tech._c_blsa)
        ncv2 = n["bl"] * cap_bitline * vcorevdd
        e["bl-act"] = 0.75 * ncv2
        e["bl-pre"] = tech.bl_pre_beta * ncv2
        # heat of ACT/PRE
        e["heat-act-bl"] = 0.5 * ncv2
        e["heat-pre-bl"] = (0.25 + tech.bl_pre_beta) * ncv2

        # col
        e["col"] = n["col"]/2 * l["od-col-avg"] * tech.scale_cap(tech._c_ca) * vcorevdd # differential but still full swing  # data and cmd buses go to 1 half the time
        e["csl"] = (self.bank_clks_per_atom() * n["csl"]) * l["csl"] * (tech.scale_cap(tech._c_csl)*c_ratio_csl) * vcorevdd
        if self.mdl_over_mat:
            e["csl"] += (self.mdl_over_mat*binary_csls/2) * l["csl"] * tech.scale_cap(tech._c_csl) * vcorevdd # for MDL-over-MAT add half of binary CSLs
        e["ldl"] = self.bank_clks_per_atom() * n["ldl"] * l["ldl"] * tech.scale_cap(tech._c_ldl) * vcorevdd/2 # half due to precharge (fill vcore/2)
        e["mdl"] = self.bank_clks_per_atom() * n["mdl"] * l["mdl"] * (tech.scale_cap(tech._c_mdl)*c_ratio_mdl) * vcorevdd/2 # half due to precharge
        e["bgbus+gbus"] = self.bgbus_clks_per_atom() * n["bgbus"]/2 * l["bgbus+gbus"] * tech.scale_cap(tech._c_bus) * bgbus_cap_ratio * vcorevdd * np.pow(self.dbi_transition_factor_max(),self.dbi_for_bgbus) # data and cmd buses go to 1 half the time  # databus inversion decreases frequency of transition by ~80%
        e["dq"] = (n["dq"] * mdl_pumps)/2 * tech.c_dq * vddqlvdd * self.dbi_transition_factor_max() # data and cmd buses go to 1 half the time  # databus inversion decreases frequency of transition by ~80%
        e["row-dq"] = 0 #row * tech.c_dq * vcorevdd
        e["col-dq"] = 0 #col * tech.c_dq * vcorevdd # these are driven onto DRAM, do not draw DRAM power sources
        e["tsv"] = (n["tsv"] * tsv_pumps)/2 * l["tsv-n-layers-avg"] * tech.scaled_cap_tsv() * vddqlvdd * self.dbi_transition_factor_max() # data and cmd buses go to 1 half the time  # databus inversion decreases frequency of transition by ~80%
        e["row-tsv"] = row/2 * l["tsv-n-layers-avg"] * tech.scaled_cap_tsv() * vcorevdd # data and cmd buses go to 1 half the time
        e["col-tsv"] = col/2 * l["tsv-n-layers-avg"] * tech.scaled_cap_tsv() * vcorevdd # data and cmd buses go to 1 half the time
        e["base"] = (n["tsv"] * tsv_pumps)/2 * l["base"] * tech.scale_cap(tech._c_bus) * base_cap_ratio * vddqlvdd * self.dbi_transition_factor_max() # data and cmd buses go to 1 half the time  # databus inversion decreases frequency of transition by ~80%
        e["row-base"] = row/2 * l["base"] * tech.scale_cap(tech._c_ca) * vcorevdd # data and cmd buses go to 1 half the time
        e["col-base"] = col/2 * l["base"] * tech.scale_cap(tech._c_ca) * vcorevdd # data and cmd buses go to 1 half the time

        cmd_e = {}
        cmd_e["pre"] = e["row-dq"]/2 + e["row-base"]/2 + e["row-tsv"]/2 + e["row"]/2 + e["mwl"] + e["bl-pre"] # ASSUME MWL is inverted and raises in PRE to lower LWL # assume half as many bits for PRE cmd
        cmd_e["act"] = e["row-dq"] + e["row-base"] + e["row-tsv"] + e["row"] + e["lwl"] + e["bl-act"] # ASSUME MWL is inverted and lowers to raises LWL, so only LWL pulls current from VDD
        cmd_e["rd"] = e["col-dq"] + e["col-base"] + e["col-tsv"] + e["col"] + e["csl"] + e["ldl"] + e["mdl"] + e["bgbus+gbus"] + e["tsv"] + e["base"] + e["dq"]
        
        cmd_e["heat-pre"] = e["row-dq"]/2 + e["row-base"]/2 + e["row-tsv"]/2 + e["row"]/2 + e["heat-pre-mwl"] + e["heat-pre-lwl"] + e["heat-pre-bl"]
        cmd_e["heat-act"] = e["row-dq"] + e["row-base"] + e["row-tsv"] + e["row"] + e["heat-act-mwl"] + e["heat-act-lwl"] + e["heat-act-bl"]
        # heat is not currently calculated separately for DQs

        return cmd_e, e
