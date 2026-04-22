#!/usr/bin/env python3
# =============================================================================
# EARTHQUAKE EPICENTER LOCATION FROM SEISMIC INTENSITY - SETTINGS
#
# Author: Miroslav HALLO, Kyoto University
# E-mail: hallo.miroslav.2a@kyoto-u.ac.jp
# Tested with: Python 3.12.3, NumPy 2.4.4
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

from dataclasses import dataclass
from enum import Enum
import sys
from typing import List

import numpy as np


# =============================================================================
# CLASSES
# =============================================================================

class ScaleType(Enum):
    JMA = "jma"      # Japan: JMA Seismic Intensity Scale (Shindo)
    MMI = "mmi"      # USA: Modified Mercalli Intensity (MMI)
    EMS98 = "ems98"  # EU: European Macroseismic Scale (EMS-98)


@dataclass
class ConfigClass:
    """
    Configuration container for earthquake epicenter location settings.

    Attributes:
        ref_lon, ref_lat: Reference point coordinates [deg].
        h_top: Top depth of the rupture [km].
        grid_x_set, grid_y_set, grid_z_set: Grid (start, step, end) [km].
        scale: Instrumental Seismic Intensity Scale (jma / mmi / ems98).
        input_file: Path to the input data file.
    """
    ref_lon: float
    ref_lat: float
    h_top: float
    grid_x_set: List[float]
    grid_y_set: List[float]
    grid_z_set: List[float]
    scale: ScaleType
    input_file: str

    def generate_grids(self):
        """Generates Grid arrays from (start, step, end) parameters."""
        def make_range(p):
            num = int(round((p[2] - p[0]) / p[1])) + 1
            return np.linspace(p[0], p[2], num)

        return (
            make_range(self.grid_x_set),
            make_range(self.grid_y_set),
            make_range(self.grid_z_set)
        )

    def __post_init__(self):
        """Validates configuration parameters and ensures correct types."""
        # Check float constants
        for field in ['ref_lon', 'ref_lat', 'h_top']:
            value = getattr(self, field)
            if not isinstance(value, (int, float)):
                print(f"[!] ERROR: {field} must be a number (float/int)")
                print(f"    Got: {type(value).__name__} ('{value}')")
                sys.exit(1)
            setattr(self, field, float(value))

        # Check grid search definitions: (start, step, end)
        for field in ['grid_x_set', 'grid_y_set', 'grid_z_set']:
            grid = getattr(self, field)
            if not isinstance(grid, (list, tuple)):
                print(f"[!] ERROR: {field} must be a list or tuple")
                sys.exit(1)
            if len(grid) != 3:
                print(f"[!] ERROR: {field} must have 3 elements")
                sys.exit(1)
            try:
                start, step, end = [float(x) for x in grid]
            except ValueError:
                print(f"[!] ERROR: All elements in {field} must be numbers")
                sys.exit(1)
            if end <= start:
                print(f"[!] ERROR: In {field}: 'end' must be > 'start'")
                sys.exit(1)
            if step <= 0:
                print(f"[!] ERROR: In {field}: 'step' must be > 0")
                sys.exit(1)
            diff = end - start
            if step > (diff / 2):
                print(f"[!] ERROR: In {field}: 'step' is too large")
                sys.exit(1)
            setattr(self, field, [start, step, end])

        # Intensity scale (str)
        if isinstance(self.scale, str):
            try:
                self.scale = ScaleType(self.scale.lower())
            except ValueError:
                valid_options = [s.value for s in ScaleType]
                print(f"[!] ERROR: Invalid intensity scale: '{self.scale}'")
                print(f"    Supported scales are: {valid_options}")
                sys.exit(1)

        # Intensity scale (ScaleType)
        if not isinstance(self.scale, ScaleType):
            print("[!] ERROR: Invalid intensity scale")
            print(f"    Got: {type(self.scale).__name__}")
            sys.exit(1)


# =============================================================================
# INPUT PARAMETERS
# =============================================================================

INPUT = ConfigClass(
    # Reference point location (center of computation)
    ref_lon=135.6,  # [deg] Longitude
    ref_lat=35.1,   # [deg] Latitude
    h_top=3.0,      # [km] Top depth of the rupture

    # Grid search definitions: (start, step, end)
    grid_x_set=[-25.0, 0.1, 25.0],  # [km] Easting grid
    grid_y_set=[-25.0, 0.1, 25.0],  # [km] Northing grid
    grid_z_set=[6.0, 0.01, 7.0],    # [-]  Moment magnitude grid

    # Instrumental seismic intensity scale to be used for the computation
    scale='JMA',
          # JMA - Japan: JMA Seismic Intensity Scale (Shindo)
          # MMI - USA: Modified Mercalli Intensity (MMI)
          # EMS98 - EU: European Macroseismic Scale (EMS-98)

    # Path to the ASCII (UTF-8) text file containing input data
    input_file='INPUT.txt',
               # The file must be whitespace-separated and follow column order:
               # N  Latitude  Longitude  Int  Sigma  Vs30  Note
               # -----------------------------------------------
               # N: Serial number of the observation point (Integer)
               # Latitude: Geographic latitude of the point in degrees (Float)
               # Longitude: Geographic longitude of the point in deg. (Float)
               # Int: Observed instrumental seismic intensity (Float)
               # Sigma: Standard deviation of the observed intensity (Float)
               # Vs30: Average S-wave velocity in upper 30 meters [m/s] (Float)
               # Note: Descriptive string or metadata (interpreted as comment)
               # -----------------------------------------------
               # *** Japan localization ***
               # JMA instrumental seismic intensity (macro to instrumental):
               # [4] 4.0, [5-] 4.75, [5+] 5.25, [6-] 5.75, [6+] 6.25, [7] 6.75
               # Vs30: If negative ( < 0.0) it will be automatically assigned
               #    from database with a processed subset of the J-SHIS
               #    by Hallo (2026). https://doi.org/10.5281/zenodo.19379171
               # Note: Includes special flags for historical events (e, E, S)
               #    to control plotting colors. Other than listed special flags
               #    will be interpreted as a comment.
               # -----------------------------------------------
               # *** USA localization ***
               # Instrumental Modified Mercalli Intensity (MMI)
               # [I] 1.0, [II] 2.0, [III] 3.0, [IV] 4.0, [V] 5.0, [VI] 6.0,
               # [VII] 7.0, [VIII] 8.0, [IX] 9.0, [X] 10.0, [XI] 11.0, [XII] 12
               # -----------------------------------------------
               # *** EU localization ***
               # Instrumental European Macroseismic Scale (EMS-98)
               # [I] 1.0, [II] 2.0, [III] 3.0, [IV] 4.0, [V] 5.0, [VI] 6.0,
               # [VII] 7.0, [VIII] 8.0, [IX] 9.0, [X] 10.0, [XI] 11.0, [XII] 12

)

# =============================================================================
