#!/usr/bin/env python3
# =============================================================================
# EARTHQUAKE EPICENTER LOCATION FROM SEISMIC INTENSITY - PLOTTING
#
# Author: Miroslav HALLO, Kyoto University
# E-mail: hallo.miroslav.2a@kyoto-u.ac.jp
# Tested with: Python 3.12.3, Jax 0.9.2, Matplotlib 3.10.8, NumPy 2.4.4,
#              Pandas 3.0.2, GeoPandas 1.1.3
# Description: Location of the earthquake epicenter and moment magnitude from
#              instrumental seismic intensity (historical or modern).
#              Japan: JMA Seismic Intensity Scale (Shindo), instrumental
#                     prediction is following Morikawa and Fujiwara (2013).
#                     Includes automatic Vs30 querying from J-SHIS derived
#                     database for sites without Vs30 measurements.
#              USA: Modified Mercalli Intensity (MMI), instrumental prediction
#                     by Atkinson et al. (2014) for western North America.
#              EU: European Macroseismic (EMS-98), instrumental prediction
#                     by Bindi et al. (2011) and Faenza and Michelini (2010).
#
# Copyright (C) 2026 Kyoto University
#
# This program is published under the GNU General Public License (GNU GPL).
#
# This program is free software: you can modify it and/or redistribute it
# or any derivative version under the terms of the GNU General Public
# License as published by the Free Software Foundation, either version 3
# of the License, or (at your option) any later version.
#
# This code is distributed in the hope that it will be useful, but WITHOUT
# ANY WARRANTY. We would like to kindly ask you to acknowledge the authors
# and don't remove their names from the code.
#
# You should have received a copy of the GNU General Public License along
# with this program. If not, see <http://www.gnu.org/licenses/>.
#
# =============================================================================

from pathlib import Path
from typing import List

import geopandas as gpd
import jax.numpy as jnp
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd

from config import INPUT, ScaleType


# =============================================================================
# CONSTANTS
# =============================================================================

# JMA intensity scale colors (RGB)
JMA_COLORS = {
    'INT_7':  [153/255, 51/255, 153/255],
    'INT_6P': [153/255, 0/255, 0/255],
    'INT_6M': [255/255, 51/255, 0/255],
    'INT_5P': [255/255, 140/255, 0/255],
    'INT_5M': [255/255, 255/255, 0/255],
    'INT_4':  [255/255, 250/255, 205/255],
    'INT_3':  [58/255, 95/255, 205/255],
    'INT_2':  [135/255, 206/255, 250/255],
    'INT_1':  [200/255, 250/255, 255/255],
}

# Historical JMA intensity scale colors (RGB)
JMA_HIST_COLORS = {
    'INT_S': [0/255, 0/255, 128/255],
    'INT_E': [58/255, 95/255, 205/255],
    'INT_e': [135/255, 206/255, 250/255],
}

# USGS ShakeMap MMI colors (RGB)
MMI_COLORS = {
    'I':    [1.0, 1.0, 1.0],
    'II':   [0.75, 0.85, 1.0],
    'III':  [0.63, 0.82, 1.0],
    'IV':   [0.5, 1.0, 1.0],
    'V':    [0.5, 1.0, 0.5],
    'VI':   [1.0, 1.0, 0.0],
    'VII':  [1.0, 0.67, 0.0],
    'VIII': [1.0, 0.0, 0.0],
    'IX':   [0.75, 0.0, 0.0],
    'X':    [0.5, 0.0, 0.0],
}

# Other colors (RGB)
OTHER_COLORS = {
    'NAN': [0.8, 0.8, 0.8],
}


# =============================================================================
# FUNCTIONS
# =============================================================================

# Prepare JMA intensity scale RGB color for all data points
def get_jma_color(row: pd.Series) -> List[float]:
    """
    Get RGB color for the JMA intensity, including historical scale e, E, S.

    Args:
        row (pd.Series): One row of Pandas table (must contain 'int', 'text').
    Returns:
        List[float]: A list of 3 floats [R, G, B] in range 0.0 to 1.0.
    """
    # Generic color in case of an error
    try:
        # Check for NaN
        if pd.isna(row['int']):
            return OTHER_COLORS['NAN']
        # Convert to float
        val = float(row['int'])
    except (ValueError, TypeError):
        return OTHER_COLORS['NAN']

    # JMA Shindo scale
    if val >= 6.5:
        color = JMA_COLORS['INT_7']
    elif val >= 6.0:
        color = JMA_COLORS['INT_6P']
    elif val >= 5.5:
        color = JMA_COLORS['INT_6M']
    elif val >= 5.0:
        color = JMA_COLORS['INT_5P']
    elif val >= 4.5:
        color = JMA_COLORS['INT_5M']
    elif val >= 3.5:
        color = JMA_COLORS['INT_4']
    elif val >= 2.5:
        color = JMA_COLORS['INT_3']
    elif val >= 1.5:
        color = JMA_COLORS['INT_2']
    else:
        color = JMA_COLORS['INT_1']

    # Historical scale
    if pd.notna(row['text']):
        note = str(row['text']).strip()
        if note == 'e':
            color = JMA_HIST_COLORS['INT_e']
        elif note == 'E':
            color = JMA_HIST_COLORS['INT_E']
        elif note == 'S':
            color = JMA_HIST_COLORS['INT_S']

    return color


# -----------------------------------------------------------------------------
# Prepare MMI intensity scale RGB color for all data points
def get_mmi_color(row: pd.Series) -> List[float]:
    """
    Get RGB color for the MMI intensity based on USGS ShakeMap standards.

    Args:
        row (pd.Series): One row of Pandas table (must contain 'int').
    Returns:
        List[float]: A list of 3 floats [R, G, B] in range 0.0 to 1.0.
    """
    # Generic color in case of an error
    try:
        # Check for NaN
        if pd.isna(row['int']):
            return OTHER_COLORS['NAN']
        # Convert to float
        val = float(row['int'])
    except (ValueError, TypeError):
        return OTHER_COLORS['NAN']

    # MMI intensity scale (standard USGS binning)
    if val >= 9.5:
        color = MMI_COLORS['X']
    elif val >= 8.5:
        color = MMI_COLORS['IX']
    elif val >= 7.5:
        color = MMI_COLORS['VIII']
    elif val >= 6.5:
        color = MMI_COLORS['VII']
    elif val >= 5.5:
        color = MMI_COLORS['VI']
    elif val >= 4.5:
        color = MMI_COLORS['V']
    elif val >= 3.5:
        color = MMI_COLORS['IV']
    elif val >= 2.5:
        color = MMI_COLORS['III']
    elif val >= 1.5:
        color = MMI_COLORS['II']
    else:
        color = MMI_COLORS['I']

    return color


# -----------------------------------------------------------------------------
# Plot cross-sections of the PDF
def plot_slices(pdf_3d: np.ndarray, gx: jnp.ndarray, gy: jnp.ndarray,
                gz: jnp.ndarray, loc_ix: int, loc_iy: int, loc_iz: int,
                output_path: Path) -> None:
    """
    Plots cross-sections of the PDF at the ML/MAP location.

    Args:
        pdf_3d (np.ndarray): Resultant PDF in the 3D model space.
        gx (jnp.ndarray): Vectorized (JAX) grid in x direction.
        gy (jnp.ndarray): Vectorized (JAX) grid in y direction.
        gz (jnp.ndarray): Vectorized (JAX) grid in z direction.
        loc_ix, loc_iy, loc_iz (int): Position (index) of the ML/MAP solution.
        output_path (Path): Full path for the output figure.
    Returns:
        None.
    """
    output_path = str(output_path)
    max_p = np.max(pdf_3d)
    xy_slice = pdf_3d[:, :, loc_iz].T  # Mw slice
    xz_slice = pdf_3d[:, loc_iy, :].T  # E-W vertical slice
    yz_slice = pdf_3d[loc_ix, :, :]    # N-S vertical slice

    # Create figure
    plt.rcParams.update({
        'font.size': 14,
        'xtick.labelsize': 14,
        'ytick.labelsize': 14,
        'axes.labelsize': 16,
        'axes.titlesize': 16,
    })
    fig = plt.figure(figsize=(12, 10), facecolor='w')
    fig.canvas.draw()

    # Subplot 1: Mw slice
    ax1 = fig.add_subplot(2, 2, 1)
    im1 = ax1.imshow(xy_slice, extent=[gx[0], gx[-1], gy[0], gy[-1]],
                     origin='lower', cmap='magma_r', vmin=0, vmax=max_p,
                     aspect='equal')
    ax1.plot(0, 0, 'g+', markersize=10, label='Reference')
    ax1.set_xlabel('Easting (km)')
    ax1.set_ylabel('Northing (km)')
    ax1.set_facecolor([0.9, 0.9, 0.9])
    pos1 = ax1.get_position()

    # Subplot 2: Vertical Slice N-S
    ax2 = fig.add_subplot(2, 2, 2, sharey=ax1)
    ax2.imshow(yz_slice, extent=[gz[0], gz[-1], gy[0], gy[-1]],
               origin='lower', cmap='magma_r', vmin=0, vmax=max_p,
               aspect='auto')
    ax2.set_xlabel('$M_w$')
    ax2.set_ylabel('Northing (km)')
    ax2.invert_xaxis()
    ax2.set_facecolor([0.9, 0.9, 0.9])
    pos2 = ax2.get_position()
    ax2.set_position([pos2.x0, pos1.y0, pos2.width, pos1.height])

    # Subplot 3: Vertical Slice E-W
    ax3 = fig.add_subplot(2, 2, 3, sharex=ax1)
    ax3.imshow(xz_slice, extent=[gx[0], gx[-1], gz[0], gz[-1]],
               origin='lower', cmap='magma_r', vmin=0, vmax=max_p,
               aspect='auto')
    ax3.set_xlabel('Easting (km)')
    ax3.set_ylabel('$M_w$')
    ax3.set_facecolor([0.9, 0.9, 0.9])
    pos3 = ax3.get_position()
    ax3.set_position([pos1.x0, pos3.y0, pos1.width, pos3.height])

    # Subplot 4: Legend and Info
    ax4 = fig.add_subplot(2, 2, 4)
    ax4.axis('off')
    info_text = (
        "Posterior PDF\n"
        "Cross-sections at ML/MAP\n\n"
        f"Slice at $M_w$ = {gz[loc_iz]:.2f}\n"
        f"N-S slice at Easting {gx[loc_ix]:.1f} km\n"
        f"E-W slice at Northing {gy[loc_iy]:.1f} km"
    )
    ax4.text(0, 0.5, info_text, transform=ax4.transAxes, fontsize=16,
             verticalalignment='center')

    # Colorbar
    cbar_ax = fig.add_axes([pos2.x0, pos3.y0 + 0.05, pos2.width * 0.8, 0.02])
    cb = fig.colorbar(im1, cax=cbar_ax, orientation='horizontal')
    cb.set_label('Probability')
    cb.formatter = ticker.ScalarFormatter(useMathText=True)
    cb.formatter.set_scientific(True)
    cb.formatter.set_powerlimits((0, 0))
    cb.update_ticks()

    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"[+] SUCCESS: Saved to {output_path}")
    plt.close(fig)


# -----------------------------------------------------------------------------
# Plot posterior marginal PDF
def plot_marginal_pdf(pdf_3d: np.ndarray, gx: np.ndarray, gy: np.ndarray,
                      gz: np.ndarray, loc_ml: list, loc_pm: list,
                      output_path: Path) -> None:
    """
    Plots marginal 2D PDFs.

    Args:
        pdf_3d (np.ndarray): Resultant PDF in the 3D model space.
        gx (jnp.ndarray): Vectorized (JAX) grid in x direction.
        gy (jnp.ndarray): Vectorized (JAX) grid in y direction.
        gz (jnp.ndarray): Vectorized (JAX) grid in z direction.
        loc_ml (List[float]): [x, y, Mw] of ML/MAP solution.
        loc_pm (List[float]): [x, y, Mw] of PM solution.
        output_path (Path): Full path for the output figure.
    Returns:
        None.
    """
    # Compute marginal PDF
    xy_2d = np.sum(pdf_3d, axis=2).T   # Sum over Mw
    xz_2d = np.sum(pdf_3d, axis=1).T   # Sum over Northing
    yz_2d = np.sum(pdf_3d, axis=0)     # Sum over Easting
    # Max marginal PDF
    max_p = np.max([np.max(xy_2d), np.max(xz_2d), np.max(yz_2d)])

    # Create figure
    plt.rcParams.update({
        'font.size': 14,
        'xtick.labelsize': 14,
        'ytick.labelsize': 14,
        'axes.labelsize': 16,
        'axes.titlesize': 16,
    })
    fig = plt.figure(figsize=(12, 10), facecolor='w')
    fig.canvas.draw()

    # Subplot 1: Mw
    ax1 = fig.add_subplot(2, 2, 1)
    im1 = ax1.imshow(xy_2d, extent=[gx[0], gx[-1], gy[0], gy[-1]],
                     origin='lower', cmap='magma_r', vmin=0, vmax=max_p,
                     aspect='equal')
    ax1.plot(0, 0, 'g+', markersize=10, label='Reference')
    ax1.plot(loc_ml[0], loc_ml[1], 'x', color='#cc3333', markersize=8,
             markeredgewidth=1.5)
    ax1.plot(loc_pm[0], loc_pm[1], 'o', color='#3333cc', markersize=8,
             markeredgewidth=1.5, markerfacecolor='none')
    ax1.set_xlabel('Easting (km)')
    ax1.set_ylabel('Northing (km)')
    ax1.set_facecolor([0.9, 0.9, 0.9])
    pos1 = ax1.get_position()

    # Subplot 2: Vertical N-S
    ax2 = fig.add_subplot(2, 2, 2, sharey=ax1)
    ax2.imshow(yz_2d, extent=[gz[0], gz[-1], gy[0], gy[-1]],
               origin='lower', cmap='magma_r', vmin=0, vmax=max_p,
               aspect='auto')
    ax2.plot(loc_ml[2], loc_ml[1], 'x', color='#cc3333', markersize=8,
             markeredgewidth=1.5)
    ax2.plot(loc_pm[2], loc_pm[1], 'o', color='#3333cc', markersize=8,
             markeredgewidth=1.5, markerfacecolor='none')
    ax2.set_xlabel('$M_w$')
    ax2.set_ylabel('Northing (km)')
    ax2.invert_xaxis()
    ax2.set_facecolor([0.9, 0.9, 0.9])
    pos2 = ax2.get_position()
    ax2.set_position([pos2.x0, pos1.y0, pos2.width, pos1.height])

    # Subplot 3: Vertical E-W
    ax3 = fig.add_subplot(2, 2, 3, sharex=ax1)
    ax3.imshow(xz_2d, extent=[gx[0], gx[-1], gz[0], gz[-1]],
               origin='lower', cmap='magma_r', vmin=0, vmax=max_p,
               aspect='auto')
    ax3.plot(loc_ml[0], loc_ml[2], 'x', color='#cc3333', markersize=8,
             markeredgewidth=1.5)
    ax3.plot(loc_pm[0], loc_pm[2], 'o', color='#3333cc', markersize=8,
             markeredgewidth=1.5, markerfacecolor='none')
    ax3.set_xlabel('Easting (km)')
    ax3.set_ylabel('$M_w$')
    ax3.set_facecolor([0.9, 0.9, 0.9])
    pos3 = ax3.get_position()
    ax3.set_position([pos1.x0, pos3.y0, pos1.width, pos3.height])

    # Subplot 4: Legend
    ax4 = fig.add_subplot(2, 2, 4)
    ax4.axis('off')
    ax4.set_xlim(0, 1)
    ax4.set_ylim(0, 1)
    ax4.plot(0.1, 0.8, 'x', color='#cc3333', markersize=8,
             markeredgewidth=1.5)
    ax4.text(0.2, 0.8, 'ML/MAP solution', verticalalignment='center',
             fontsize=16)
    ax4.plot(0.1, 0.7, 'o', color='#3333cc', markersize=8,
             markeredgewidth=1.5, markerfacecolor='none')
    ax4.text(0.2, 0.7, 'PM solution', verticalalignment='center', fontsize=16)
    ax4.text(0.1, 0.9, 'Posterior Marginal PDF', verticalalignment='center',
             fontsize=16)

    # Colorbar
    cbar_ax = fig.add_axes([pos2.x0, pos3.y0 + 0.05, pos2.width * 0.8, 0.02])
    cb = fig.colorbar(im1, cax=cbar_ax, orientation='horizontal')
    cb.set_label('Marginal Probability')
    cb.formatter = ticker.ScalarFormatter(useMathText=True)
    cb.formatter.set_scientific(True)
    cb.formatter.set_powerlimits((0, 0))
    cb.update_ticks()

    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"[+] SUCCESS: Saved to {output_path}")
    plt.close(fig)


# -----------------------------------------------------------------------------
# Plot data misfits
def plot_misfits(loc_ml: List[float], x_st: np.ndarray, y_st: np.ndarray,
                 i_misfit: np.ndarray, i_colors: np.ndarray, output_path
                 ) -> None:
    """
    Plots the intensity data misfits.

    Args:
        loc_ml (List[float]): [x, y, Mw] - ML/MAP solution [km].
        x_st, y_st (np.ndarray): All station coordinates [km].
        i_misfit (np.ndarray): Array of residuals (observed - synthetic).
        i_colors (np.ndarray): RGB colors for all station.
        output_path (Path): Full path for the output figure.
    Returns:
        None.
    """
    # Prepare epicentral distances
    r0 = np.sqrt((loc_ml[0] - x_st)**2 + (loc_ml[1] - y_st)**2)

    # Create figure
    plt.rcParams.update({
        'font.size': 11,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'axes.labelsize': 13,
        'axes.titlesize': 13,
    })
    fig, ax = plt.subplots(figsize=(8, 5), facecolor='w')
    ax.scatter(r0, i_misfit, c=i_colors, s=50, edgecolors='k',
               linewidths=0.5, zorder=3)
    ax.set_xscale('log')
    ax.axhline(0, color='0.6', linestyle='--', linewidth=1, zorder=1)
    ax.axhline(1, color='0.6', linestyle=':', linewidth=0.8, zorder=1)
    ax.axhline(-1, color='0.6', linestyle=':', linewidth=0.8, zorder=1)

    # Titles
    ax.set_title('The ML/MAP solution misfit', fontweight='normal')
    ax.set_xlabel('Epicentral distance (km)')
    ax.set_ylabel('Instrumental intensity misfits')

    # Box formating
    ax.set_axisbelow(False)
    ax.spines['top'].set_visible(True)
    ax.spines['right'].set_visible(True)
    ax.set_xlim(left=max(0.1, np.min(r0) * 0.8), right=np.max(r0) * 1.2)
    y_max = np.max(np.abs(i_misfit))
    ax.set_ylim(-y_max * 1.1, y_max * 1.1)

    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"[+] SUCCESS: Saved to {output_path}")
    plt.close(fig)


# -----------------------------------------------------------------------------
# Plot map with stations
def plot_station_map(data: gpd.GeoDataFrame, loc_ml: List[float],
                     loc_pm: List[float], loc_res_wgs: List[float],
                     loc_pm_wgs: List[float], gx: jnp.ndarray,
                     gy: jnp.ndarray, output_path: Path) -> None:
    """
    Plots the map of stations and the location results.

    Args:
        data (gpd.GeoDataFrame): Dataframe containing station coords & colors.
        loc_ml (List[float]): [x, y, Mw] of ML/MAP solution.
        loc_pm (List[float]): [x, y, Mw] of PM solution.
        loc_res_wgs (List[float]): [Lat, Lon, Mw] of ML/MAP solution.
        loc_pm_wgs (List[float]): [Lat, Lon, Mw] of PM solution.
        gx (jnp.ndarray): Vectorized (JAX) grid in x direction.
        gy (jnp.ndarray): Vectorized (JAX) grid in y direction.
        output_path (Path): Full path for the output figure.
    Returns:
        None.
    """
    # Create figure
    plt.rcParams.update({
        'font.size': 11,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'axes.labelsize': 13,
        'axes.titlesize': 13,
    })
    fig, ax = plt.subplots(figsize=(12, 5), facecolor='w')

    # Plot stations
    station_colors = np.array(data['color'].tolist())
    ax.scatter(data['x_km'], data['y_km'], c=station_colors, marker='s',
               linewidths=0.1, edgecolors='k', s=50, zorder=4)
    # ML/MAP solution
    ax.plot(loc_ml[0], loc_ml[1], 'x', color='#cc3333', markersize=10,
            markeredgewidth=1.0, label='ML/MAP sol.', zorder=6)
    # PM solution
    ax.plot(loc_pm[0], loc_pm[1], 'o', color='#3333cc', markersize=10,
            markeredgewidth=1.0, markerfacecolor='none', label='PM solution',
            zorder=6)
    # Search area
    grid_rect_x = [gx[0], gx[-1], gx[-1], gx[0], gx[0]]
    grid_rect_y = [gy[0], gy[0], gy[-1], gy[-1], gy[0]]
    ax.plot(grid_rect_x, grid_rect_y, color='#33cc33', linewidth=1.5,
            label='Search area', zorder=2)

    # Define proxy artists
    if INPUT.scale == ScaleType.JMA:  # Japan (JMA Shindo)
        has_special_flags = data['text'].isin(['e', 'E', 'S']).any()
        common_levels = [
            ("Int 7",  JMA_COLORS['INT_7']),
            ("Int 6+", JMA_COLORS['INT_6P']),
            ("Int 6-", JMA_COLORS['INT_6M']),
            ("Int 5+", JMA_COLORS['INT_5P']),
            ("Int 5-", JMA_COLORS['INT_5M']),
            ("Int 4",  JMA_COLORS['INT_4']),
        ]
        if has_special_flags:
            int_levels = common_levels + [
                ("S (hist)", JMA_HIST_COLORS['INT_S']),
                ("E (hist)", JMA_HIST_COLORS['INT_E']),
                ("e (hist)", JMA_HIST_COLORS['INT_e']),
            ]
        else:
            int_levels = common_levels + [
                ("Int 3", JMA_COLORS['INT_3']),
                ("Int 2", JMA_COLORS['INT_2']),
                ("Int 1", JMA_COLORS['INT_1']),
            ]
    elif INPUT.scale in [ScaleType.MMI, ScaleType.EMS98]:  # USA and EU
        int_levels = [
            ("X+",   MMI_COLORS['X']),
            ("IX",   MMI_COLORS['IX']),
            ("VIII", MMI_COLORS['VIII']),
            ("VII",  MMI_COLORS['VII']),
            ("VI",   MMI_COLORS['VI']),
            ("V",    MMI_COLORS['V']),
            ("IV",   MMI_COLORS['IV']),
            ("III",  MMI_COLORS['III']),
            ("II",   MMI_COLORS['II']),
            ("I",    MMI_COLORS['I']),
        ]

    # Plot proxy artists
    for label, color in int_levels:
        ax.scatter([], [], c=[color], marker='s', linewidths=0.1,
                   edgecolors='k', s=50, label=label)

    # Format
    ax.set_aspect('equal')
    xlims = ax.get_xlim()
    ylims = ax.get_ylim()
    dx = xlims[1] - xlims[0]
    dy = ylims[1] - ylims[0]
    new_xlim = [xlims[0] - 0.02 * dx, xlims[1] + 0.02 * dx]
    new_ylim = [ylims[0] - 0.02 * dy, ylims[1] + 0.02 * dy]
    ax.set_xlim(new_xlim)
    ax.set_ylim(new_ylim)

    # Text info
    ml_text = (f" ML/MAP solution\n"
               f" Lat{loc_res_wgs[0]:9.4f}\n"
               f" Lon{loc_res_wgs[1]:9.4f}\n"
               f" $M_w$  {loc_res_wgs[2]:4.1f}")

    pm_text = (f" PM solution\n"
               f" Lat{loc_pm_wgs[0]:9.4f}\n"
               f" Lon{loc_pm_wgs[1]:9.4f}\n"
               f" $M_w$  {loc_pm_wgs[2]:4.1f}")
    ax.text(new_xlim[0], new_ylim[0], ml_text, color='#cc3333',
            va='bottom', ha='left', fontsize=10, family='monospace')
    ax.text(new_xlim[0], new_ylim[1], pm_text, color='#3333cc',
            va='top', ha='left', fontsize=10, family='monospace')

    # Labels
    ax.set_title('Data points and location solution', fontweight='normal')
    ax.set_xlabel('Easting (km)')
    ax.set_ylabel('Northing (km)')
    ax.grid(True, linestyle=':', alpha=0.6)
    ax.legend(loc='upper left', bbox_to_anchor=(1.05, 1), borderaxespad=0.0,
              frameon=True, edgecolor='k', fontsize=10)

    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"[+] SUCCESS: Saved to {output_path}")
    plt.close(fig)
