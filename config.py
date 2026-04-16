#!/usr/bin/env python3
# =============================================================================
# EARTHQUAKE EPICENTER LOCATION FROM SEISMIC INTENSITY - SETTINGS
#
# Author: Miroslav HALLO, Kyoto University
# E-mail: hallo.miroslav.2a@kyoto-u.ac.jp
# Tested with: Python 3.12.3, NumPy 2.4.4
# Description: Location of the earthquake epicenter and moment magnitude from
#              JMA instrumental seismic intensity (historical or modern). The
#              JMA intensity prediction is following Morikawa and Fujiwara
#              (2013). Includes automatic Vs30 querying from J-SHIS derived
#              Vs30 database for sites without Vs30 measurements.
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
# You should have received copy of the GNU General Public License along
# with this program. If not, see <http://www.gnu.org/licenses/>.
#
# =============================================================================

from dataclasses import dataclass
from typing import List

import numpy as np


# =============================================================================
# CLASSES
# =============================================================================

@dataclass
class ConfigClass:
    """
    Configuration container for earthquake epicenter location settings.

    Attributes:
        ref_lon, ref_lat: Reference point coordinates [deg].
        h_top: Top depth of the rupture [km].
        grid_x_set, grid_y_set, grid_z_set: Grid (start, step, end) [km].
        input_file: Path to the input data file.
    """
    ref_lon: float
    ref_lat: float
    h_top: float
    grid_x_set: List[float]
    grid_y_set: List[float]
    grid_z_set: List[float]
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


# =============================================================================
# INPUT PARAMETERS
# =============================================================================

INPUT = ConfigClass(
    # Reference point location (center of computation)
    ref_lon=135.6,  # [deg] Longitude
    ref_lat=35.1,   # [deg] Latitude
    h_top=3.0,      # [km] Top depth of the rupture

    # Grid search definitions: (start, step, end)
    grid_x_set=[-25.0, 0.05, 25.0],  # [km] Easting grid
    grid_y_set=[-25.0, 0.05, 25.0],  # [km] Northing grid
    grid_z_set=[6.0, 0.005, 7.0],    # [-]  Moment magnitude grid

    # Path to the ASCII (UTF-8) text file containing input data
    input_file='INPUT.txt',
    # The file must be whitespace-separated and follow this column order:
    # N  Latitude  Longitude  Int  Sigma  Vs30  Note
    # -----------------------------------------------
    # N: Serial number of the observation point (Integer)
    # Latitude: Geographic latitude of the point in decimal degrees (Float)
    # Longitude: Geographic longitude of the point in decimal degrees (Float)
    # Int: Observed JMA instrumental seismic intensity (Float), e.g.:
    #      [4] 4.0,  [5-] 4.75,  [5+] 5.25,  [6-] 5.75,  [6+] 6.25,  [7] 6.75
    # Sigma: Standard deviation (uncertainty) of the observed intensity (Float)
    #        It must be a positive number ( > 0.0).
    # Vs30: Average shear-wave velocity in the upper 30 meters [m/s] (Float)
    #       If negative ( < 0.0) it will be automatically assigned from SQLite
    #       database with a processed subset of the J-SHIS seismic hazard data
    #       by Hallo (2026). https://doi.org/10.5281/zenodo.19379171
    # Note: Descriptive string or metadata. Includes special flags for
    #       historical events (e, E, S) to control plotting colors. Other than
    #       listed special flags will be interpreted as a comment.
)

# =============================================================================
