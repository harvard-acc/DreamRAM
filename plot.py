import pandas as pd
import matplotlib.pyplot as plt
import numpy as np, colorsys
import random
from matplotlib import colormaps
from matplotlib import colors
from matplotlib.lines import Line2D
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.ticker import LogLocator, FuncFormatter, FixedLocator
from matplotlib.cm import ScalarMappable
from paretoset import paretoset #pip install paretoset
import re
import os
import sys
import getopt


DATAVERSION = None

# colors for capacity levels for abstract results
cap_color_map = {
    16: "#563263",
    18: "#B23E53",
    24: "#F14C55",
    32: "#FEB2AB"
}

def _trim3(v: float) -> str:
        s = f"{v:.3f}"                 # round to 3 decimals max
        return s.rstrip("0").rstrip(".")  # drop trailing zeros and dot

def fmt_pow2_k(x, pos):
        # label only exact powers of two
        if x <= 0 or not np.isclose(np.log2(x), round(np.log2(x)), atol=1e-12):
            return ""
        return _trim3(x/1024) + "k" if x >= 1024 else _trim3(x)

def fmt_k_1000(x, pos):
    return _trim3(x/1000) + "k" if x >= 1000 else _trim3(x)


def mono_cmap_hex(hex_color="#6C7A96", light=0.95, dark=0.20, sat=None, n=256):
    """Single-hue colormap by sweeping lightness of `hex_color`."""
    r, g, b = colors.to_rgb(hex_color)
    h, l, s = colorsys.rgb_to_hls(r, g, b)
    if sat is None:
        sat = s
    Ls = np.linspace(light, dark, n)  # light → dark
    rgb = [colorsys.hls_to_rgb(h, L, sat) for L in Ls]
    return colors.LinearSegmentedColormap.from_list("mono_hex", rgb, N=n)

def safe_filename(s: str) -> str:
    s = s.replace('/', '-')  # kill path separators first
    # optionally collapse other weird chars to underscores
    return re.sub(r'[^\w\-.()+ ]+', '_', s)

def common_name(s):
    if s == 'capacity_gbytes':
        return 'Capacity (GB)'
    if s == 'total_area_mmmm':
        return 'Area (square mm)'
    if s == 'bw_gbytes':
        return 'Bandwidth (GB/s)'
    if s == 'worst_latency_ns':
        return 'Latency (ns)'
    if s == 'metric_e_per_bit_closed':
        return 'Energy per Bit (pJ/b)'
    if s == 'EDP':
        return 'EDP (pJ*ns/bit)'
    if s == 'storage_density':
        return 'Storage Density (GB/mm^2)'

def ideal(s):
    if s == 'capacity_gbytes':
        return 'max'
    if s == 'total_area_mmmm':
        return 'min'
    if s == 'bw_gbytes':
        return 'max'
    if s == 'worst_latency_ns':
        return 'min'
    if s == 'metric_e_per_bit_closed':
        return 'min'
    if s == 'EDP':
        return 'min'
    if s == 'storage_density':
        return 'max'


def color(n):
    # orange
    if n == 5:  
        # return '#de852fff'
        return '#FF7F0EFF'
    # yellow
    if n == 4:  
        # return '#bbbc3cff'
        return '#F2B701FF'
    # green
    if n == 3:  
        # return '#659c38ff'
        return '#2CA02CFF'
    # blue
    if n == 2:  
        return '#77bbcdff'
        # return '#1F77B4FF'
    # brown
    if n == 1: 
        return '#7d584dff'
        # return '#8C564BFF'
    if n == 0:
        return '#c7c7c7ff'

def filter(data, metric, value, ceiling=True, inclusive=True):
    if ceiling and inclusive:
        # drop all data[metric] > max_value (keep data[metric] <= max_value)
        new_data = data[data[metric] <= value]
    if (not ceiling) and inclusive:
        # drop all data[metric] < min_value (keep data[metric] >= min_value)
        new_data = data[data[metric] >= value]
    if ceiling and (not inclusive):
        # drop all data[metric] >= max_value (keep data[metric] < max_value)
        new_data = data[data[metric] < value]
    if (not ceiling) and (not inclusive):
        # drop all data[metric] <= min_value (keep data[metric] > min_value)
        new_data = data[data[metric] > value]

    new_data = new_data.reset_index(drop=True)

    return data, new_data

def nextcolor2():
    yield '#de852fff'

def nextcolor():
    yield '#7d584dff'
    yield '#77bbcdff'
    yield '#659c38ff'
    yield '#bbbc3cff'
    yield '#de852fff'

def getbest(data, d, tests):

    plot_isopower = True # isopower line, when (bandwidth, E/b)
    plot_isobw = True
    cap_cap = True

    

    if plot_isopower:
        #print(data['bw_gbytes'][0]*data['metric_e_per_bit_closed'][0])
        print("isopower")
        power = data['bw_gbytes'][0]*data['metric_e_per_bit_closed'][0]
        d.drop(d[d.bw_gbytes * d.metric_e_per_bit_closed > power].index, inplace=True)
    if plot_isobw:
        print("isobw")
        #print(data['bw_gbytes'][0])
        print(data['bw_gbytes'][0])
        d.drop(d[d.bw_gbytes < data['bw_gbytes'][0]].index, inplace=True)
    if cap_cap:
        print("cap_cap")
        #print(data['bw_gbytes'][0])
        print(data['capacity_gbytes'][0])
        d.drop(d[d.capacity_gbytes < data['capacity_gbytes'][0]].index, inplace=True)
    d.sort_values(by=['metric_e_per_bit_closed','bw_gbytes'], ascending=[True,False], inplace=True)

    header = data.columns.tolist()
    master_file = f'data/{DATAVERSION}/pareto/hbm3_{DATAVERSION}_iso.csv'

    maxes = {}
    print(tests)
    for test in tests:
        print(test)
        if test == 'capacity':
            objective_name = 'capacity_gbytes'
            objective_max = True
        elif test == 'bandwidth':
            objective_name = 'bw_gbytes'
            objective_max = True
        elif test == 'power':
            objective_name = 'worst_power_w'
            objective_max = False
        elif test == 'e_closed':
            objective_name = 'metric_e_per_bit_closed'
            objective_max = False
        else: 
            assert(False)
        if objective_max:
            print([d[d["user"] <= i+1][objective_name].idxmax() for i in range(5)])
            maxes[test] = [d[d["user"] <= i+1][objective_name].idxmax() for i in range(5)]
        else:
            print([d[d["user"] <= i+1][objective_name].idxmin() for i in range(5)])
            maxes[test] = [d[d["user"] <= i+1][objective_name].idxmin() for i in range(5)]

        i=4
        print('Objective:', test)
        print('user:', i+1)
        print('id:', d['id'][maxes[test][i]])
        print('Bandwidth:\t', round(d['bw_gbytes'][maxes[test][i]],6))
        print('Capacity: \t', round(d['capacity_gbytes'][maxes[test][i]],6))
        print('Energy:   \t', round(d['metric_e_per_bit_seq'][maxes[test][i]],6))
        print('Energy:   \t', round(d['metric_e_per_bit_closed'][maxes[test][i]],6))
        print('Power:    \t', round(d['worst_power_w'][maxes[test][i]],6))
        print('Latency:  \t', round(d['worst_latency_ns'][maxes[test][i]],6))
        print('Area:     \t', round(d['total_area_mmmm'][maxes[test][i]],6))
        print(d.loc[maxes[test][i]]) # output each config found
        print()

    return d, maxes


def plot_ew_bstract(data, y2, x1, c1, y2name, x1name, objective_name, data_og, zoom=True):

    dmin = np.min(data[objective_name])
    dmax = np.max(data[objective_name])

    x1 ='bw_gbytes'
    x1name = 'Bandwidth (GB/s)'

    y2 = 'metric_e_per_bit_closed'
    y2name = 'Energy (pJ/b)'
    
    fig, ax2 = plt.subplots(1,1,figsize=(10, 10))

    c1 = 'capacity_gbytes'
    colorname = 'Capacity (GB)'
    
    # data in gray
    if not zoom:
        sc1 = ax2.scatter(data[x1], data[y2], alpha=1, s=3, c=15+0*data[c1], edgecolors='none', norm=colors.Normalize(-0.5,19.5),cmap='tab20',zorder=0)

    data['worst_power_w'] = data['metric_e_per_bit_closed'] * data['bw_gbytes'] * 8e-3

    ds = {}
    dd = data.copy()
    d, maxes = getbest(data, dd, ['power','e_closed','bandwidth','capacity'])

    # which rank within each iso set to pick
    i = 4
    iso_sets = ["power", "e_closed", "bandwidth", "capacity"]

    # collect index labels (dedup while preserving order)
    labels = []
    for key in iso_sets:
        if key in maxes and len(maxes[key]) > i:
            labels.append(maxes[key][i])
    labels = list(dict.fromkeys(labels))  # remove duplicates, keep order

    # pull those rows by label
    best_df = d.loc[labels].copy()

    # (optional) keep the original index as a column for traceability
    best_df.insert(0, "original_index", best_df.index)

    # ensure output directory exists
    out_file = f"data/{DATAVERSION}/hbm3_{DATAVERSION}_usercolor_composite_cone_best.csv"
    # save
    data_out = pd.concat([data.iloc[[0]], best_df], ignore_index=True)
    data_out.to_csv(out_file, index=False)
    print(f"Saved {len(data_out)} best points to {out_file}")
    #exit()

    color_by_third_metric = True
    plot_isopower = True # isopower line, when (bandwidth, E/b)
    plot_isobw = True

    print(d[c1].unique())

    color_map = cap_color_map
    # color_map = {
    #     16: "#9B59B6",  # purple (amethyst)
    #     18: "#B46AA0",
    #     24: "#CC7A89",
    #     32: "#E07A5F"   # pinkish-red
    # }
    cap_colors = list(cap_color_map.keys()) #TODO check that 32 is actually the max capacity !!
    
    colors_binary = d[c1].astype(int).map(color_map)

    mask = {}
    for cap_color in cap_colors:
        mask[cap_color] = d[c1].astype(int) == cap_color

    for cap_color in cap_colors:
        ax2.scatter(
            d[x1][mask[cap_color]],
            d[y2][mask[cap_color]],
            c=color_map[cap_color],
            label=f"{cap_color} GB",
            s=5,
            alpha=1,
            edgecolors='none',
            zorder=1
        )


    def power_watts_gib(bandwidth_gibs, energy_pj_per_bit):
        """Convert bandwidth (GB/s, base-2 GiB) and energy (pJ/bit) to Watts."""
        k = 0.008589934  # conversion constant for GiB/s
        return k * bandwidth_gibs * energy_pj_per_bit
    
    # baseline
    ax2.plot(data[x1][0],data[y2][0],ms=15,marker='s',mfc='r',mec='r',zorder=4)

    def y_at_frac(ax, frac):
        ymin, ymax = ax.get_ylim()
        if ax.get_yscale() == "log":
            return 10**(np.log10(ymin) + frac*(np.log10(ymax) - np.log10(ymin)))
        return ymin + frac*(ymax - ymin)

    if plot_isopower:
 
        ax2.plot(1/np.linspace(np.min(data[y2])/(data[x1][0]*data[y2][0]), np.max(data[y2])/(data[x1][0]*data[y2][0])), np.linspace(np.min(data[y2]),np.max(data[y2])), color = 'k', linewidth='2', alpha=1,zorder=2)
        power_limit = power_watts_gib(data[x1][0], data[y2][0])
        print(f"Iso-power line: x*y = {power_limit:.4f} W")
        
        const = data[x1][0] * data[y2][0]
        ypos_d = y_at_frac(ax2, 0.12)             # low on the diagonal
        xpos_d = const / ypos_d

        dx, dy = -60, -60                         # same as arrow offset
        angle_deg = np.degrees(np.arctan2(dy, dx))

        # normalize the arrow direction vector
        length = np.hypot(dx, dy)
        ux, uy = dx/length, dy/length

        # perpendicular unit vector (rotate by 90°)
        px, py = -uy, ux

        # move the text 20 pts along the SAME direction (negative for top-right-ish)
        shift_along = 20   # negative to move opposite arrow head
        shift_perp  = 5

        text_dx = shift_along*ux + shift_perp*px
        text_dy = shift_along*uy + shift_perp*py


    if plot_isobw:
        ax2.plot(np.linspace(data[x1][0], data[x1][0]), np.linspace(np.min(data[y2]),np.max(data[y2])), color = 'k', linewidth='2', alpha=1,zorder=3)
        bw_limit = data[x1][0]
        print(f"Iso-bandwidth line at x = {bw_limit:.4f} GB/s")
        
    
        x0 = bw_limit
        # pick a low y on the current visible span (log-aware)
        ymin, ymax = ax2.get_ylim()
        if ax2.get_yscale() == "log":
            ypos = 10**(np.log10(ymin) + 0.05*(np.log10(ymax) - np.log10(ymin)))
        else:
            ypos = ymin + 0.05*(ymax - ymin)

    label_cfg = {
        "capacity":  dict(offset=(12, 0),  ha="left",  va="center", name=f"XX% Higher Capacity",),   # right of point
        "power":     dict(offset=(-12, 0), ha="right", va="center", name=f"XX% Lower Power"),   # left of point
        "e_closed":  dict(offset=(0, -12), ha="center",va="top", name=f"XX% Lower Energy"),      # below point
        "bandwidth": dict(offset=(12, 0),  ha="left",  va="center", name=f"XX% Higher Bandwidth"),   # right of point
    }
    iso_sets = ['power','e_closed','bandwidth','capacity']
    for key in maxes:
        n = nextcolor()
        i=4
        
        # get the row index of this "best" point
        idx = maxes[key][i]

        # pick its color based on capacity
        star_color = colors_binary.loc[idx]

        ax2.plot(
            d[x1].loc[idx],
            d[y2].loc[idx],
            ms=15,
            marker='*',
            mfc=star_color,   # face color = mapped color
            mec='black',          # white edge
            zorder=i+10
        )
        
        cfg = label_cfg[key]
        ax2.annotate(
            "",
            xy=(d[x1].loc[idx], d[y2].loc[idx]),
            xytext=cfg["offset"],
            textcoords="offset points",
            ha=cfg["ha"],
            va=cfg["va"],
            fontsize=16,
            fontweight="bold",    # <-- bold font
            color="black",
        )



    ax2.set_yscale('log')
    ax2.set_xscale('log')
        
    ax2.set_xlim((data_og[x1].min()/1.07, data_og[x1].max()*1.07)) 
    ax2.set_ylim((data_og[y2].min()/1.07, data_og[y2].max()*1.07))

    pow2_major = LogLocator(base=2, subs=(1.,))
    nice_energy = [0.25, 0.5, 1, 2, 4, 8]#[0.3, 0.5, 1, 2, 5, 10]
    
    ax2.xaxis.set_major_locator(pow2_major)
    ax2.xaxis.set_major_formatter(fmt_pow2_k)

    ax2.yaxis.set_major_locator(FixedLocator(nice_energy))
    ax2.yaxis.set_major_formatter(fmt_k_1000)

    ax2.tick_params(axis="both", which="major", labelsize=24, length=6, direction="out")
    ax2.tick_params(axis="both", which="minor", labelsize=24, length=6, direction="out")


    # ax2.set_title(f'{y2name} vs. {x1name}\nColor: {colorname}', fontsize=14)
    ax2.set_xlabel(x1name, fontsize=24)
    ax2.set_ylabel(y2name, fontsize=24)
    # ax2.grid(True)

    legend_elements = [
            Line2D([], [], color='black', marker='*', linestyle='None',
                markersize=15, label='Best by Metric')
    ]
    for key in list(cap_color_map.keys()):
        legend_elements += [Line2D([], [], color=cap_color_map[key], marker='o', linestyle='None', markersize=12, label=f'{key} GB')]

    legend_elements += [Line2D([], [], color='red', marker='s', linestyle='None', markersize=13, label='Baseline')]


    ax2.legend(handles=legend_elements, loc='upper right',
            frameon=True, fontsize=16)
    ax2.set_box_aspect(1)

    # leave room for labels and legend without shrinking the axes
    # fig.subplots_adjust(left=0.10, bottom=0.10, right=0.80, top=0.98)
    
    plt.savefig(f"plots/{DATAVERSION}/hbm3_{DATAVERSION}_abstract_results{'_zoom' if zoom else ''}_test.png", dpi=300) # if "No such file or directory", try making a plots/{DATAVERSION} directory

def helper_legend(ax2):
    handles = [
        Line2D([0], [0], marker='*', lw=0, mfc='black', mec='black',
            markeredgewidth=1.0, markersize=20, label="Best by Tier"),
        Line2D([0], [0], color=color(1), lw=5, label="A Pareto"), #1
        Line2D([0], [0], color=color(2), lw=5, label="B Pareto"),
        Line2D([0], [0], color=color(3), lw=5, label="C Pareto"),
        Line2D([0], [0], color=color(4), lw=5, label="D Pareto"),
        Line2D([0], [0], color=color(5), lw=5, label="E Pareto"),
        Line2D([0], [0], marker='s', lw=0, mfc='red', mec='red',
            markeredgewidth=2.5, markersize=15, label="Baseline"),        
    ]

    leg = ax2.legend(
        handles=handles,
        loc="lower right",
        fontsize=24,            # label font size
        # title="Legend",         # optional
        title_fontsize=14,      # title font size
        handlelength=2.2,       # length of line sample
        handletextpad=0.8,      # gap between handle and text
        labelspacing=0.3,       # vertical spacing between entries
        borderpad=0.1,          # padding inside the frame
        bbox_to_anchor=(1.0, 0.03),  # (x, y) in axes-fraction coords; y↑ moves it up
        borderaxespad=0.1,           # padding from the axes border
        frameon=True, fancybox=True, framealpha=0.95, facecolor="white", edgecolor="0.85"
    )

def helper(ax2, text):
    ax2.text(
        1.0, 0.0, f"{text}",
        transform=ax2.transAxes,
        ha="right", va="bottom",
        fontsize=24, fontweight="bold",
        zorder=100, clip_on=False,
    )

def helper_scatter_plot(ax2, data, y2, x1, c1, y2name, x1name, objective_name, data_og, cmap='tab20', plot_pareto=True, gray_data=False, lighter_data=False, plot_best=True, idealmarker=True, color_by_third_metric=True, pareto_color_by_third_metric=True, third_metric_cmap='Purples'):
    if y2name == 'EDP (pJ*ns/bit)' and x1name == 'Bandwidth (GB/s)':
        main_title = 'High Performance Edge'
    elif y2name == 'Latency (ns)' and x1name == 'Bandwidth (GB/s)':
        main_title = 'Server CPU'
    elif y2name == 'Energy per Bit (pJ/b)' and x1name == 'Bandwidth (GB/s)':
        main_title = 'Server GPU'
    elif y2name == 'Energy per Bit (pJ/b)' and x1name == 'Area (square mm)':
        main_title = 'Embedded IoT'

    dmin = np.min(data_og[objective_name])
    dmax = np.max(data_og[objective_name])

    minval, maxval, n = 0.1, 0.8, 256
    d = data.sort_values(by='user',ascending=False)
    if ideal(c1) == "max":
        cmap_obj = plt.get_cmap(third_metric_cmap)
        new_cmap = colors.LinearSegmentedColormap.from_list(
            "truncated",  # simple name
            cmap_obj(np.linspace(minval, maxval, n))
        )
        cmap_third_used = new_cmap
    else:
        cmap_obj = plt.get_cmap(third_metric_cmap)
        new_cmap = colors.LinearSegmentedColormap.from_list(
            "truncated",  # simple name
            cmap_obj(np.linspace(minval, maxval, n))
        )
        cmap_third_used = new_cmap.reversed()
    # sc2 = ax2.scatter(d[x1], d[y2], alpha=1, s=7, c=(d[c1]+lighter_data)*(1-gray_data)*(1-color_by_third_metric)+15*gray_data+d[objective_name]*color_by_third_metric, edgecolors='none', cmap=cmap if not color_by_third_metric else cmap_third_used, zorder=0, norm=colors.LogNorm(vmin=dmin, vmax=dmax) if color_by_third_metric else colors.Normalize(vmin=-0.5, vmax=19.5))
    # if color_by_third_metric:
    #     fig.colorbar(sc2, ax=ax2)
    if color_by_third_metric:
        sc2 = ax2.scatter(
            d[x1], d[y2], s=7, alpha=1,
            c=d[objective_name],                                  # ← only the metric
            cmap=cmap_third_used,
            norm=colors.LogNorm(vmin=max(dmin, 1e-12), vmax=dmax),
            edgecolors='none', zorder=0
        )
    else:
        sc2 = ax2.scatter(
            d[x1], d[y2], s=7, alpha=1,
            c=d[c1],
            cmap=cmap,
            norm=colors.Normalize(vmin=-0.5, vmax=19.5),
            edgecolors='none', zorder=0
        )
    ax2.plot(data_og[x1][0], data_og[y2][0], ms=9,marker='s',mfc='r',mec='r',zorder=40)

    # Calculate paretos
    if plot_pareto or plot_best:
        sets = [data[data["user"] <= i] for i in range(1, 6)]
        masks = [paretoset(sets[i][[x1,y2]], sense=[ideal(x1),ideal(y2)]) for i in range(5)]
        paretos = [sets[i][masks[i]] for i in range(5)]

    # Plot paretos
    header = data.columns.tolist()
    if plot_pareto:
        if data.equals(data_og):
            master_file = f'data/{DATAVERSION}/pareto/hbm3_{DATAVERSION}_{main_title}_data_og.csv'
        else: 
            master_file = f'data/{DATAVERSION}/pareto/hbm3_{DATAVERSION}_{main_title}_filtered.csv'
        pd.DataFrame(columns=header).to_csv(master_file, index=False)

        for i in reversed(range(5)):
           p = paretos[i].sort_values(by=x1)
           p.to_csv(master_file, mode="a", index=False, header=False)
           #    print(f"{main_title}: outputted the pareto csv")
           if pareto_color_by_third_metric:
               ax2.plot(p[x1], p[y2], c=color(i+1), ls='solid', lw=3.5, zorder=10+12-2*i)
               ax2.scatter(p[x1], p[y2], s=45, c=p[objective_name], cmap=cmap_third_used, edgecolors=color(i+1), linewidths=2.5, zorder=11+12-2*i, norm=colors.LogNorm(vmin=dmin, vmax=dmax))
           else:
               ax2.plot(p[x1], p[y2], c=color(i+1), ms='4', mfc=color(i+1), mec=color(i+1), marker='o', ls='solid', lw=3, zorder=11+12-2*i)

    # Plot best point in each pareto
    if plot_best:
        # Evaluate third metric
        maxes = []
        for i in range(5):
            try:
                s = paretos[i][objective_name]
                # uncomment next line if you want to ignore zeros:
                # s = s[s != 0]
                ix = s.idxmin()   # or idxmax() if that's your case
            except ValueError:
                # empty after filtering → no point for this user
                ix = None
            maxes.append(ix)


        # Plot
        if data.equals(data_og):
            set_name = f'data_og'
        else: 
            set_name = f'filtered'
        print(set_name)
        if pareto_color_by_third_metric:
            for i in reversed(range(5)): 
                if maxes[i] is None:
                    continue
                #ax2.scatter(paretos[i][x1][maxes[i]], paretos[i][y2][maxes[i]], s=25, marker='*', c=paretos[i][objective_name][maxes[i]], edgecolors=color(i+1), vmin=dmin, vmax=dmax, cmap=third_metric_cmap, zorder=4)
                ax2.plot(paretos[i][x1][maxes[i]],paretos[i][y2][maxes[i]], ms=20, marker='*', mfc=(0,0,0,0), mec='#FDF6E3FF', mew=2, zorder=40+6-i)
                ax2.plot(paretos[i][x1][maxes[i]],paretos[i][y2][maxes[i]], ms=17, marker='*', mfc=(0,0,0,0), mec='#D9C69AFF', mew=2, zorder=40+6-i)# #E6D9B6FF
                ax2.plot(paretos[i][x1][maxes[i]],paretos[i][y2][maxes[i]], ms=14, marker='*', mfc=(0,0,0,0), mec=color(i+1), mew=2, zorder=40+6-i)
                

                
                print(x1name, paretos[i][x1][maxes[i]], y2name, paretos[i][y2][maxes[i]], objective_name, paretos[i][objective_name][maxes[i]])
        else:
            for i in reversed(range(5)):
                if maxes[i] is None:
                    continue
                #ax2.scatter(paretos[i][x1][maxes[i]],paretos[i][y2][maxes[i]], s=12, marker='*', c=color(i+1), vmin=dmin, vmax=dmax, cmap='Purples')
                ax2.plot(paretos[i][x1][maxes[i]],paretos[i][y2][maxes[i]], ms=35, marker='*', mfc=color(i+1), mec=color(i+1), zorder=40+6-i)
                print(paretos[i][x1][maxes[i]], paretos[i][y2][maxes[i]], objective_name, paretos[i][objective_name][maxes[i]])
    
    # if energy vs. bandwidth, plot the isopower line
    # if y2=='metric_e_per_bit_closed' and x1=='bw_gbytes':
    #     ax2.plot(1/np.linspace(np.min(data[y2])/(data[x1][0]*data[y2][0]), np.max(data[y2])/(data[x1][0]*data[y2][0])), np.linspace(np.min(data[y2]),np.max(data[y2])), color = 'k', linewidth='2', alpha=1, zorder=100)


    ax2.plot(data_og[x1][0], data_og[y2][0], ms=9,marker='s',mfc='r',mec='r',zorder=40)

    return sc2

   
def create_scatter_plot(data, y2, x1, c1, y2name, x1name, objective_name, data_og, cmap='tab20', plot_pareto=True, gray_data=False, lighter_data=False, plot_best=True, idealmarker=True, color_by_third_metric=True, pareto_color_by_third_metric=True, third_metric_cmap='Purples'):

    # Create two subplots side by side, sharing y-axis
    fig, axes = plt.subplots(1, 2, figsize=(20, 10), sharex=True, sharey=True, constrained_layout=True)

    dmin = np.min(data[objective_name])
    dmax = np.max(data[objective_name])

    print(f"x: {x1}")
    print(f"y: {y2}")
    print(f"c: {c1}")

    print(f"Capacity min:{np.min(data['capacity_gbytes'])} max:{np.max(data['capacity_gbytes'])}")

    print(dmin)
    print(dmax)
    print(len(data))

    print(f"User 1: {(data['user'] == 1).sum()}")
    print(f"User 2: {(data['user'] == 2).sum()}")
    print(f"User 3: {(data['user'] == 3).sum()}")
    print(f"User 4: {(data['user'] == 4).sum()}")
    print(f"User 5: {(data['user'] == 5).sum()}")

    # data_og.head(1).to_csv("data/pareto/baseline.csv", index=False)
    # exit()
    plot_og = helper_scatter_plot(axes[0], data_og, y2, x1, c1, y2name, x1name, objective_name, data_og, 
                                    cmap='tab20', 
                                    plot_pareto=True, 
                                    gray_data=False, 
                                    lighter_data=False, 
                                    plot_best=True, 
                                    idealmarker=True, 
                                    color_by_third_metric=True, 
                                    pareto_color_by_third_metric=True, 
                                    third_metric_cmap=mono_cmap_hex(hex_color="#6C7A96"))
    plot_filter = helper_scatter_plot(axes[1], data, y2, x1, c1, y2name, x1name, objective_name, data_og, 
                                    cmap='tab20', 
                                    plot_pareto=True, 
                                    gray_data=False, 
                                    lighter_data=False, 
                                    plot_best=True, 
                                    idealmarker=True, 
                                    color_by_third_metric=True, 
                                    pareto_color_by_third_metric=True, 
                                    third_metric_cmap=mono_cmap_hex(hex_color="#6C7A96"))
    
    axes[0].set_yscale('log')
    axes[0].set_xscale('log')
    axes[0].set_xlim((data_og[x1].min()/1.07, data_og[x1].max()*1.07))
    axes[0].set_ylim((data_og[y2].min()/1.07, data_og[y2].max()*1.07))

    pow2_major = LogLocator(base=2, subs=(1.,))
    
    nice_latency = [30, 40, 60, 100]
    nice_edp = [30, 100, 300, 1000]
    nice_energy = [0.25, 0.5, 1, 2, 4, 8]
    nice_sd = [0.001, 0.003, 0.01, 0.02]
    nice_area = [50, 100, 200, 500, 1000, 2000]

    print(y2name)
    print(x1name)
    main_title = f"{y2name} vs. {x1name}"
    plot_x_log2 = True
    plot_y_log2 = False
    plot_cmap_log2 = False
    if y2name == 'EDP (pJ*ns/bit)' and x1name == 'Bandwidth (GB/s)':
        main_title = 'High Performance Edge'
        text = 'Filtered by 4 GB <= Capacity <= 16 GB'
        nice_y= nice_edp
        nice_c = nice_area
    elif y2name == 'Latency (ns)' and x1name == 'Bandwidth (GB/s)':
        main_title = 'Server CPU'
        text = 'Filtered by 8 GB <= Capacity'
        nice_y = nice_latency
        nice_c = nice_sd
    elif y2name == 'Energy per Bit (pJ/b)' and x1name == 'Bandwidth (GB/s)':
        main_title = 'Server GPU'
        text = 'Filtered by 8 GB <= Capacity'
        nice_y = nice_energy
        plot_cmap_log2 = True
        # helper_legend(axes[1])
    elif y2name == 'Energy per Bit (pJ/b)' and x1name == 'Area (square mm)':
        main_title = 'Embedded IoT'
        text = 'Filtered by Capacity <= 2 GB'
        nice_x = nice_area
        nice_y = nice_energy
        nice_c = nice_latency
        plot_x_log2 = False
    print(main_title)
    # helper(axes[1], text)
    
    # if color_by_third_metric or pareto_color_by_third_metric:
    #     main_title += f" — colored by {common_name(objective_name)}"
    # fig.suptitle(main_title, fontsize=32, fontweight="bold")

    if x1name == 'Area (square mm)':
        # x1name = r"Area ($\mathbf{mm^2}$)"
        x1name = r"Area ($mm^2$)"
    try:
        fig.supxlabel(x1name, fontsize=36)
        fig.supylabel(y2name, fontsize=36)
    except AttributeError:
        # Fallback for older Matplotlib: place text manually
        fig.text(0.5, 0.02, x1name, ha="center", va="center", fontsize=36)
        fig.text(0.02, 0.5, y2name, ha="center", va="center", rotation="vertical", fontsize=36)

    axes[1].tick_params(labelleft=False)

    sm = ScalarMappable(cmap=plot_filter.get_cmap(), norm=plot_filter.norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes[1], location="right", pad=0.02)
    
    for ax in axes:
        ax.tick_params(axis="both", which="major", labelsize=24, length=6, direction="out")
        ax.tick_params(axis="both", which="minor", labelsize=24, length=6, direction="out")

    if plot_x_log2:
        for ax in axes:
            ax.xaxis.set_major_locator(pow2_major)
            ax.xaxis.set_major_formatter(fmt_pow2_k)
    else:
        ax.xaxis.set_major_locator(FixedLocator(nice_x))
        ax.xaxis.set_major_formatter(fmt_k_1000)

    if plot_y_log2:
        print("plot log 2")
    else:
        ax.yaxis.set_major_locator(FixedLocator(nice_y))
        ax.yaxis.set_major_formatter(fmt_k_1000)
        
    if plot_cmap_log2:
        # cbar.ax.yaxis.set_major_locator(pow2_major)
        cbar.set_ticks([0.125, 0.25, 0.5, 1, 2, 4, 8, 16, 32, 64])  
        cbar.formatter = FuncFormatter(fmt_pow2_k)
        cbar.update_ticks()
    else:
        cbar.set_ticks(nice_c)  
        cbar.formatter = FuncFormatter(fmt_k_1000)
        cbar.update_ticks()
    if common_name(objective_name) == "Storage Density (GB/mm^2)":
        cbar.set_label(r"Storage Density (GB/$\mathrm{mm^2}$)", fontsize=36, labelpad=6)
    else:
        cbar.set_label(common_name(objective_name), fontsize=36, labelpad=6)
    cbar.ax.yaxis.set_label_position("right")
    cbar.ax.yaxis.set_ticks_position("right")
    cbar.ax.tick_params(labelsize=28, pad=6, labelright=True, labelleft=False)

    # plt.tight_layout()
    plt.savefig(f"plots/{DATAVERSION}/hbm3_{DATAVERSION}_{main_title}_filtered.png", dpi=300)

def create_design_space(data, y2, x1, y2name, x1name):

    og_data , filtered_data = filter(data=data, metric='user', value=5, ceiling=True)

    data = filtered_data
    
    fig, ax2 = plt.subplots(figsize=(10, 10)) #(12, 10) for energy-bandwidth design space fig 4

    color_map = {
        5: '#FF7F0EFF',  # orange
        4: '#F2B701FF',  # yellow
        3: '#2CA02CFF',  # green
        2: '#77bbcdff',  # blue
        1: '#7d584dff',  # brown
        0: '#c7c7c7ff',  # gray
    }

    colors = [color_map[m] for m in data['user']]
    user_int = data['user'].astype(int)
    
    # ax2.scatter(data[x1], data[y2], c=colors, s=7, edgecolor='none')
    order = [5, 4, 3, 2, 1]  # plot in this order

    for u in order:
        mask = (user_int == u)
        ax2.scatter(
            data[x1][mask],
            data[y2][mask],
            c=color_map[u],  
            s=7,
            edgecolors="none",
            zorder=1  # optional: increasing zorder with u
        )
    ax2.plot(data[x1][0],data[y2][0],ms=13,marker='s',mfc='r',mec='r',zorder=4)
    ax2.set_yscale('log')
    ax2.set_xscale('log')

    ax2.set_xlim((og_data[x1].min()/1.07, og_data[x1].max()*1.07)) 
    ax2.set_ylim((og_data[y2].min()/1.07, og_data[y2].max()*1.07))

    # pow2_major = LogLocator(base=2, subs=(1.,))
    # nice_energy = [0.3, 0.5, 1, 2, 5, 10]
    # ax2.xaxis.set_major_locator(pow2_major)
    # ax2.xaxis.set_major_formatter(fmt_pow2_k)
    # ax2.yaxis.set_major_locator(FixedLocator(nice_energy))
    # ax2.yaxis.set_major_formatter(fmt_k_1000)

    # pow10_major = LogLocator(base=10, subs=(1.,))
    # ax2.xaxis.set_major_locator(pow10_major)
    # ax2.xaxis.set_major_formatter(fmt_k_1000)
    # ax2.yaxis.set_major_locator(pow10_major)
    # ax2.yaxis.set_major_formatter(fmt_k_1000)

    ax2.tick_params(axis="both", which="major", labelsize=24, length=6, direction="out")
    ax2.tick_params(axis="both", which="minor", labelsize=24, length=6, direction="out")
    
    # ax2.set_title(f'{y2name} vs. {x1name}\nColor: {colorname}', fontsize=14)

    #if y2name == "Energy per Bit (pJ/b)":
    ax2.set_xlabel(x1name, fontsize=48)
    #if x1name == "Bandwidth (GB/s)":
    ax2.set_ylabel(y2name, fontsize=48)

    def convert(u):
        if u == 1:
            return "A"
        if u == 2:
            return "B"
        if u == 3:
            return "C"
        if u == 4:
            return "D"
        if u == 5:
            return "E"
    
    legend = False
    if legend:
        legend_elements = [
            Line2D([0], [0], marker='o', color='w',
                label=f'Tier {convert(u)}',
                markerfacecolor=color_map[u], markersize=15,
                markeredgecolor="none")
            for u in reversed(order)
        ]

        # add baseline symbol
        legend_elements.append(
            Line2D([0], [0], marker='s', color='w',
                label='Baseline',
                markerfacecolor='r', markeredgecolor='r', markersize=13)
        )

        # ax2.legend(handles=legend_elements,
        #         loc="upper left", 
        #         frameon=True, fontsize=14)
        fig.legend(
            handles=legend_elements,
            loc="upper left",
            bbox_to_anchor=(0.8, 0.97),   # right of axes, top-aligned
            frameon=True,
            fontsize=20
        )

    # make the **data box** square
    ax2.set_box_aspect(1)

    # leave room for labels and legend without shrinking the axes
    fig.subplots_adjust(left=0.20, bottom=0.20, right=0.97, top=0.97)

    #plt.tight_layout()
    #plt.savefig(f"plots/{DATAVERSION}/hbm3_{DATAVERSION}_full_design_space_test.png", dpi=300)
    plt.savefig(f"plots/{DATAVERSION}/hbm3_{DATAVERSION}_full_design_space_{y2name[0]}_{x1name[0]}.png", dpi=300)


def add_tiers_and_edp_to_csv(csv_file_path, csv_file):
    """
        Generate user tiers and create a new file that is csv_file_path - ".csv" + "usercolor_composite.csv" with the user tiers appended
    """
    try:
        # Read the CSV file
        data = pd.read_csv(csv_file_path)
        
        print("reading from", csv_file_path)
        print("generating tiers, edp, and storage density")

        #users
        # 1: bank is constant
        # 2: MAT is constant
        # 3: no proposals
        # 4: no DLOMAT
        # 5: all
        data['user'] = [5 for x in range(len(data['id']))]
        print('user filter 1/5')
        data['user'] -= [data['subarrays'][x]==data['subarrays'][0] and data['mats'][x]==data['mats'][0] and data['mat_rows'][x]==data['mat_rows'][0] and data['mat_cols'][x]==data['mat_cols'][0] and data['brvsa'][x]==data['brvsa'][0] and data['ha_layout'][x]==data['ha_layout'][0] and data['ha_double_ldls'][x]==data['ha_double_ldls'][0] and data['subchannels'][x]==data['subchannels'][0] and data['mdl_over_mat'][x]==data['mdl_over_mat'][0] and data['salp_groups'][x]==data['salp_groups'][0] and data['salp_all'][x]==data['salp_all'][0] and data['ldls_mdls'][x]==data['ldls_mdls'][0] and data['atom_size'][x]==data['atom_size'][0] for x in range(len(data['id']))]
        print('user filter 2/5')
        data['user'] -= [data['mat_rows'][x]==data['mat_rows'][0] and data['mat_cols'][x]==data['mat_cols'][0] and data['brvsa'][x]==data['brvsa'][0] and data['ha_layout'][x]==data['ha_layout'][0] and data['ha_double_ldls'][x]==data['ha_double_ldls'][0] and data['subchannels'][x]==data['subchannels'][0] and data['mdl_over_mat'][x]==data['mdl_over_mat'][0] and data['ldls_mdls'][x]==data['ldls_mdls'][0] and data['atom_size'][x]==data['atom_size'][0] for x in range(len(data['id']))]
        print('user filter 3/5')
        data['user'] -= [data['mat_rows'][x]==data['mat_rows'][0] and data['mat_cols'][x]==data['mat_cols'][0] and data['mdl_over_mat'][x]==data['mdl_over_mat'][0] and data['ldls_mdls'][x]==data['ldls_mdls'][0] and data['atom_size'][x]==data['atom_size'][0] for x in range(len(data['id']))]
        print('user filter 4/5')
        data['user'] -= [data['mdl_over_mat'][x]==data['mdl_over_mat'][0] for x in range(len(data['id']))]
        print(sum(data['user']==1))
        print(sum(data['user']==2))
        print(sum(data['user']==3))
        print(sum(data['user']==4))
        print(sum(data['user']==5))
    

        # map to the cmap tab20
        # 1 to 10
        # 2 to 18
        # 3 to 4
        # 4 to 16
        # 5 to 2
        # 16 to 15 (gray out)
        # given by a 5th-degree polynomial
        data['usercolor'] = data['user'].apply(lambda x: np.round(23371*x**5/72072 - 216955*x**4/24024 + 5566111*x**3/72072 - 6673741*x**2/24024 + 15357971*x/36036 - 621359/3003))

        # energy delay product
        data['EDP'] = data['metric_e_per_bit_closed'] * data['worst_latency_ns']
        data['storage_density'] = data['capacity_gbytes'] / (data['dies']*data['die_x_mm']*data['die_y_mm'])

        # save
        filename = csv_file_path.removesuffix(".csv") + "_usercolor_composite.csv"
        #data.to_csv(filename)
        data.to_csv(csv_file)

        print(f"appended user tiers etc and saved to {filename}\n")
        
    except FileNotFoundError:
        print(f"Error: The file '{csv_file_path}' was not found.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")





if __name__ == '__main__':

    # this sets the global variable at the top of the file
    DATAVERSION = '47b8b3d'

    args = sys.argv[1:]
    options = "hi:"
    long_options = ["input"]

    try:
        arguments, values = getopt.getopt(args, options, long_options)
        for currentArg, currentVal in arguments:
            if currentArg in ("-h", "--Help"):
                print("Usage:")
                print("python3 plot.py [-i INPUT_FILE_LABEL]")
                print("")
                exit()
            elif currentArg in ("-i", "--input"):
                print("Set data file label to:", currentVal)
                DATAVERSION = currentVal
            
    except getopt.error as err:
        print(str(err))



    csv_file = f'data/{DATAVERSION}/hbm3_{DATAVERSION}_usercolor_composite.csv'

    # on the first run, run these:
    if not os.path.isfile(f'data/{DATAVERSION}/hbm3_{DATAVERSION}_usercolor_composite.csv'):
        add_tiers_and_edp_to_csv(f'data/{DATAVERSION}/hbm3_{DATAVERSION}.csv', csv_file)
        os.makedirs(f'data/{DATAVERSION}/pareto', exist_ok=True)
        os.makedirs(f'plots/{DATAVERSION}', exist_ok=True)

    # Call the function to create and display the scatter plot.
    print(f'Loading data from {csv_file}')
    data = pd.read_csv(csv_file)
    print(data.keys())

    counts = data['capacity_gbytes'].value_counts().sort_index()

    # how many distinct values total
    n_unique = counts.shape[0]
    print("distinct values:", n_unique)

    # pretty print like: "1 1s, 2 2s, 2 3s"
    print(", ".join(f"{cnt} {val}s" for val, cnt in counts.items()))

    # exit()

    c = 'capacity_gbytes'
    a = 'total_area_mmmm'
    b = 'bw_gbytes'
    l = 'worst_latency_ns'
    e = 'metric_e_per_bit_closed'
    edp = 'EDP'
    sd = 'storage_density'

    # lists of axes:
    # y, x, c

    # application design scenarios
    list_axes = [[edp, b, a, 'av'],
                [l, b, sd, 'gpcpu'],
                [e, b, c, 'datacenter'],
                [e, a, l, 'iot']] 

    # for design tiers, uncomment this
    #list_axes = [[c,b],[a,b],[l,b],[e,b],[a,c],[l,c],[e,c],[l,a],[e,a],[e,l]]

    for axes in list_axes:

        try: # if third axis and scenario specified, plot design scenarios and abstract results (datacenter)
            if axes[3] == 'av':
                ceiling = True
                value = 16
                metric = c
            elif axes[3] == 'gpcpu':
                ceiling = False
                value = 8
                metric = c
            elif axes[3] == 'datacenter':
                ceiling = False
                value = 8
                metric = c
            elif axes[3] == 'iot':
                ceiling = True
                value = 2
                metric = c
            
            data, filtered_data = filter(data=data, metric=metric, value=value, ceiling=ceiling)
            #_ , filtered_data = filter(data=filtered_data, metric=metric, value=value, ceiling=True)
            if axes[3] == 'av':
                data, filtered_data = filter(data=filtered_data, metric=metric, value=4, ceiling=False)

            print("unique:")
            print(filtered_data["capacity_gbytes"].map('{:.4f}'.format).unique())


            if axes[0]=='e' and axes[1]=='b' and axes[2]=='c' and axes[3]=='datacenter':
                plot_ew_bstract(data, axes[0], axes[1], axes[2], common_name(axes[0]), common_name(axes[1]), axes[2], data_og=data, zoom=True)
            else:
                print("Skip abstract results plot for non-abstract axes")
            #exit()
            
            create_scatter_plot(filtered_data, axes[0], axes[1], axes[2], common_name(axes[0]), common_name(axes[1]), axes[2], data_og=data)

            
        except: # no third axis, use unfiltered tiers
            print("No third axis, using unfiltered tiers as color")
            create_scatter_plot(data, axes[0], axes[1], 'usercolor', common_name(axes[0]), common_name(axes[1]))
        
        create_design_space(data, axes[0], axes[1], common_name(axes[0]), common_name(axes[1]))
        