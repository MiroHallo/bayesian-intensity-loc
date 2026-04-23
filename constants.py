#!/usr/bin/env python3
# =============================================================================
# EARTHQUAKE EPICENTER LOCATION FROM SEISMIC INTENSITY - CONSTANTS (COLORS)
#
# Author: Miroslav HALLO, Kyoto University
# E-mail: hallo.miroslav.2a@kyoto-u.ac.jp
# Tested with: Python 3.12.3
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

from types import SimpleNamespace
from typing import List, Union, Optional


# =============================================================================
# CONSTANTS
# =============================================================================

# JMA intensity scale colors (RGB)
JMA_COLORS = SimpleNamespace(
    INT_7=[153/255, 51/255, 153/255],
    INT_6P=[153/255, 0/255, 0/255],
    INT_6M=[255/255, 51/255, 0/255],
    INT_5P=[255/255, 140/255, 0/255],
    INT_5M=[255/255, 255/255, 0/255],
    INT_4=[255/255, 250/255, 205/255],
    INT_3=[58/255, 95/255, 205/255],
    INT_2=[135/255, 206/255, 250/255],
    INT_1=[200/255, 250/255, 255/255],
)

# Historical JMA intensity scale colors (RGB)
JMA_COLORS_HIST = SimpleNamespace(
    INT_S=[0/255, 0/255, 128/255],
    INT_E=[58/255, 95/255, 205/255],
    INT_e=[135/255, 206/255, 250/255],
)

# USGS ShakeMap MMI colors (RGB)
MMI_COLORS = SimpleNamespace(
    I=[1.0, 1.0, 1.0],
    II=[0.75, 0.85, 1.0],
    III=[0.63, 0.82, 1.0],
    IV=[0.5, 1.0, 1.0],
    V=[0.5, 1.0, 0.5],
    VI=[1.0, 1.0, 0.0],
    VII=[1.0, 0.67, 0.0],
    VIII=[1.0, 0.0, 0.0],
    IX=[0.75, 0.0, 0.0],
    X=[0.5, 0.0, 0.0],
)

# Other colors (RGB)
OTHER_COLORS = SimpleNamespace(
    NAN=[0.8, 0.8, 0.8],
)

# Legend for JMA scale (standard)
JMA_LEGEND = [
    ("Int 7", JMA_COLORS.INT_7),
    ("Int 6+", JMA_COLORS.INT_6P),
    ("Int 6-", JMA_COLORS.INT_6M),
    ("Int 5+", JMA_COLORS.INT_5P),
    ("Int 5-", JMA_COLORS.INT_5M),
    ("Int 4", JMA_COLORS.INT_4),
    ("Int 3", JMA_COLORS.INT_3),
    ("Int 2", JMA_COLORS.INT_2),
    ("Int 1", JMA_COLORS.INT_1),
]

# Legend for JMA scale (historical)
JMA_LEGEND_HIST = [
    ("Int 7", JMA_COLORS.INT_7),
    ("Int 6+", JMA_COLORS.INT_6P),
    ("Int 6-", JMA_COLORS.INT_6M),
    ("Int 5+", JMA_COLORS.INT_5P),
    ("Int 5-", JMA_COLORS.INT_5M),
    ("Int 4", JMA_COLORS.INT_4),
    ("S (hist)", JMA_COLORS_HIST.INT_S),
    ("E (hist)", JMA_COLORS_HIST.INT_E),
    ("e (hist)", JMA_COLORS_HIST.INT_e),
]

# Legend for MMI/EMS-98 scale
MMI_LEGEND = [
    ("X+", MMI_COLORS.X),
    ("IX", MMI_COLORS.IX),
    ("VIII", MMI_COLORS.VIII),
    ("VII", MMI_COLORS.VII),
    ("VI", MMI_COLORS.VI),
    ("V", MMI_COLORS.V),
    ("IV", MMI_COLORS.IV),
    ("III", MMI_COLORS.III),
    ("II", MMI_COLORS.II),
    ("I", MMI_COLORS.I),
]


# =============================================================================
# FUNCTIONS
# =============================================================================

# Get JMA intensity RGB color
def get_jma_color(val: Union[float, int, str, None],
                  note: Optional[str] = None) -> List[float]:
    """
    Get RGB color for the JMA intensity, including historical scale e, E, S.

    Args:
        val: JMA intensity value or None.
        note: Historical scale flag ('e', 'E', 'S') or None.
    Returns:
        List[float]: A list of 3 floats [R, G, B] in range 0.0 to 1.0.
    """
    try:
        if val is None:
            return OTHER_COLORS.NAN

        v = float(val)
        if v != v:  # NaN
            return OTHER_COLORS.NAN

        # JMA Shindo scale
        if v >= 6.5:
            color = JMA_COLORS.INT_7
        elif v >= 6.0:
            color = JMA_COLORS.INT_6P
        elif v >= 5.5:
            color = JMA_COLORS.INT_6M
        elif v >= 5.0:
            color = JMA_COLORS.INT_5P
        elif v >= 4.5:
            color = JMA_COLORS.INT_5M
        elif v >= 3.5:
            color = JMA_COLORS.INT_4
        elif v >= 2.5:
            color = JMA_COLORS.INT_3
        elif v >= 1.5:
            color = JMA_COLORS.INT_2
        else:
            color = JMA_COLORS.INT_1

        # Historical scale override
        if note:
            note_clean = str(note).strip()
            if note_clean == 'e':
                color = JMA_COLORS_HIST.INT_e
            elif note_clean == 'E':
                color = JMA_COLORS_HIST.INT_E
            elif note_clean == 'S':
                color = JMA_COLORS_HIST.INT_S

        return color

    except (ValueError, TypeError):
        return OTHER_COLORS.NAN


# -----------------------------------------------------------------------------
# Get MMI/EMS-98 intensity RGB color
def get_mmi_color(val: Union[float, int, str, None]) -> List[float]:
    """
    Get RGB color for the MMI intensity based on USGS ShakeMap standards.

    Args:
        val: MMI intensity value or None.
    Returns:
        List[float]: A list of 3 floats [R, G, B] in range 0.0 to 1.0.
    """
    try:
        if val is None:
            return OTHER_COLORS.NAN

        v = float(val)
        if v != v:  # NaN
            return OTHER_COLORS.NAN

        # MMI intensity scale (standard USGS binning)
        if v >= 9.5:
            color = MMI_COLORS.X
        elif v >= 8.5:
            color = MMI_COLORS.IX
        elif v >= 7.5:
            color = MMI_COLORS.VIII
        elif v >= 6.5:
            color = MMI_COLORS.VII
        elif v >= 5.5:
            color = MMI_COLORS.VI
        elif v >= 4.5:
            color = MMI_COLORS.V
        elif v >= 3.5:
            color = MMI_COLORS.IV
        elif v >= 2.5:
            color = MMI_COLORS.III
        elif v >= 1.5:
            color = MMI_COLORS.II
        else:
            color = MMI_COLORS.I

        return color

    except (ValueError, TypeError):
        return OTHER_COLORS.NAN
