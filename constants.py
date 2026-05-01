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
#              EU/Switzerland: European Macroseismic (EMS-98), PGV prediction
#                     by Bindi et al. (2011) for Mediterranean/Italy or
#                     by Cauzzi et al. (2015) for Switzerland, and conversion
#                     to instrumental intensity by Faenza and Michelini (2010).
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
    INT_7=[180/255, 0/255, 104/255],
    INT_6P=[165/255, 0/255, 33/255],
    INT_6M=[255/255, 40/255, 0/255],
    INT_5P=[255/255, 153/255, 0/255],
    INT_5M=[255/255, 230/255, 0/255],
    INT_4=[250/255, 230/255, 150/255],
    INT_3=[0/255, 65/255, 255/255],
    INT_2=[0/255, 170/255, 255/255],
    INT_1=[242/255, 242/255, 255/255],
)

# Historical JMA intensity scale colors (RGB)
JMA_COLORS_HIST = SimpleNamespace(
    INT_S=[0/255, 0/255, 128/255],
    INT_E=[58/255, 95/255, 205/255],
    INT_e=[135/255, 206/255, 250/255],
)

# USGS ShakeMap MMI/EMS-98 colors (RGB)
MMI_COLORS = SimpleNamespace(
    I=[255/255, 255/255, 255/255],
    II=[191/255, 204/255, 255/255],
    III=[160/255, 230/255, 255/255],
    IV=[128/255, 255/255, 255/255],
    V=[122/255, 255/255, 147/255],
    VI=[255/255, 255/255, 0/255],
    VII=[255/255, 200/255, 0/255],
    VIII=[255/255, 145/255, 0/255],
    IX=[255/255, 0/255, 0/255],
    X=[128/255, 0/255, 0/255],
)

# Swiss SED EMS-98 colors (RGB)
SED_COLORS = SimpleNamespace(
    I=[255/255, 255/255, 255/255],
    II=[191/255, 204/255, 255/255],
    III=[160/255, 230/255, 255/255],
    IV=[165/255, 245/255, 122/255],
    V=[255/255, 255/255, 0/255],
    VI=[255/255, 153/255, 0/255],
    VII=[255/255, 0/255, 0/255],
    VIII=[180/255, 0/255, 0/255],
    IX=[120/255, 0/255, 0/255],
)

# Other colors (RGB)
OTHER_COLORS = SimpleNamespace(
    NAN=[0.8, 0.8, 0.8],
)

# Legend for JMA scale (standard)
JMA_LEGEND = [
    ("Intensity", None),
    ("(JMA Shindo)", None),
    ("7", JMA_COLORS.INT_7),
    ("6+", JMA_COLORS.INT_6P),
    ("6-", JMA_COLORS.INT_6M),
    ("5+", JMA_COLORS.INT_5P),
    ("5-", JMA_COLORS.INT_5M),
    ("4", JMA_COLORS.INT_4),
    ("3", JMA_COLORS.INT_3),
    ("2", JMA_COLORS.INT_2),
    ("1", JMA_COLORS.INT_1),
]

# Legend for JMA scale (historical)
JMA_LEGEND_HIST = [
    ("Intensity", None),
    ("(JMA Shindo)", None),
    ("7", JMA_COLORS.INT_7),
    ("6+", JMA_COLORS.INT_6P),
    ("6-", JMA_COLORS.INT_6M),
    ("5+", JMA_COLORS.INT_5P),
    ("5-", JMA_COLORS.INT_5M),
    ("4", JMA_COLORS.INT_4),
    ("S (hist)", JMA_COLORS_HIST.INT_S),
    ("E (hist)", JMA_COLORS_HIST.INT_E),
    ("e (hist)", JMA_COLORS_HIST.INT_e),
]

# Legend for ShakeMap MMI/EMS-98 scale
MMI_LEGEND = [
    ("Intensity", None),
    ("(MMI/EMS-98)", None),
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

# Legend for SED EMS-98 scale
SED_LEGEND = [
    ("Intensity", None),
    ("(EMS-98, CH)", None),
    ("IX+", SED_COLORS.IX),
    ("VIII", SED_COLORS.VIII),
    ("VII", SED_COLORS.VII),
    ("VI", SED_COLORS.VI),
    ("V", SED_COLORS.V),
    ("IV", SED_COLORS.IV),
    ("III", SED_COLORS.III),
    ("II", SED_COLORS.II),
    ("I", SED_COLORS.I),
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
# Get ShakeMap MMI/EMS-98 intensity RGB color
def get_mmi_color(val: Union[float, int, str, None]) -> List[float]:
    """
    Get RGB color for MMI/EMS-98 intensity based on USGS ShakeMap standards.

    Args:
        val: MMI/EMS-98 intensity value or None.
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


# -----------------------------------------------------------------------------
# Get SED EMS-98 intensity RGB color
def get_sed_color(val: Union[float, int, str, None]) -> List[float]:
    """
    Get RGB color for the EMS-98 intensity based on SED standards.

    Args:
        val: EMS-98 intensity value or None.
    Returns:
        List[float]: A list of 3 floats [R, G, B] in range 0.0 to 1.0.
    """
    try:
        if val is None:
            return OTHER_COLORS.NAN

        v = float(val)
        if v != v:  # NaN
            return OTHER_COLORS.NAN

        # SED EMS-98 intensity scale
        if v >= 8.5:
            color = SED_COLORS.IX
        elif v >= 7.5:
            color = SED_COLORS.VIII
        elif v >= 6.5:
            color = SED_COLORS.VII
        elif v >= 5.5:
            color = SED_COLORS.VI
        elif v >= 4.5:
            color = SED_COLORS.V
        elif v >= 3.5:
            color = SED_COLORS.IV
        elif v >= 2.5:
            color = SED_COLORS.III
        elif v >= 1.5:
            color = SED_COLORS.II
        else:
            color = SED_COLORS.I

        return color

    except (ValueError, TypeError):
        return OTHER_COLORS.NAN
