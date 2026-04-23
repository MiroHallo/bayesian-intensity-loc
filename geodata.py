#!/usr/bin/env python3
# =============================================================================
# EARTHQUAKE EPICENTER LOCATION FROM SEISMIC INTENSITY - GEODATA PROCESSING
#
# Author: Miroslav HALLO, Kyoto University
# E-mail: hallo.miroslav.2a@kyoto-u.ac.jp
# Tested with: Python 3.12.3, GeoPandas 1.1.3, Pandas 3.0.2, Shapely 2.1.2,
#              SQLAlchemy 2.0.48
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

from typing import List, Optional
from pathlib import Path

import geopandas as gpd
import pandas as pd
from shapely.geometry import Point
from sqlalchemy.engine import Engine

import utils.jshis_sqlite_query as jshis


# =============================================================================
# CONSTANTS
# =============================================================================

DEFAULT_VS30 = 350.0


# =============================================================================
# FUNCTIONS
# =============================================================================

# Prepare input dictionary from the text file
def load_input_data(input_path: Path) -> pd.DataFrame:
    """
    Reads and validates the input file. Raises Error on failure.

    Args:
        input_path (Path): Full path of the input file.
    Returns:
        pd.DataFrame: Input table ['n' 'lat' 'lon' 'int' 'sig' 'vs30' 'text'].
    """
    if not input_path.is_file():
        raise FileNotFoundError(f"Input file '{input_path}' not found!")

    try:
        inp_data = pd.read_csv(
            input_path, sep=r'\s+', comment='#', header=None,
            names=['n', 'lat', 'lon', 'int', 'sig', 'vs30', 'text'],
            usecols=[0, 1, 2, 3, 4, 5, 6])

        # Numeric conversion
        numeric_cols = ['lat', 'lon', 'int', 'sig', 'vs30']
        for col in numeric_cols:
            if not pd.api.types.is_numeric_dtype(inp_data[col]):
                inp_data[col] = pd.to_numeric(inp_data[col], errors='coerce')

        # Validations
        if inp_data[numeric_cols].isna().any().any():
            raise ValueError("Non-numeric values or missing data in input!")

        if not (inp_data['sig'] > 0).all():
            raise ValueError("Sigma must be positive (> 0) for all entries.")

        return inp_data

    except Exception as e:
        raise RuntimeError(f"Error reading file: {e}")


# -----------------------------------------------------------------------------
# Convert geodata table into km
def prepare_geo(inp_data: pd.DataFrame,
                ref_lon: float, ref_lat: float) -> gpd.GeoDataFrame:
    """
    Prepare geodata table in local coordinates (Easting, Northing)

    Args:
        inp_data (pd.DataFrame): Input data table ['n' 'lat' 'lon' 'int' ...].
        ref_lon (float): Longitude of the reference point [deg].
        ref_lat (float): Latitude of the reference point [deg].
    Returns:
        gpd.GeoDataFrame: Geopandas table ['n' 'lat' 'lon' 'x_km' 'y_km' ...].
    """
    try:
        # Convert data to geopandas (WGS84 degrees)
        data = gpd.GeoDataFrame(
            inp_data,
            geometry=gpd.points_from_xy(inp_data['lon'], inp_data['lat']),
            crs="EPSG:4326",
        )

        # Find the best UTM Zone
        utm_crs = data.estimate_utm_crs()
        # Reprojects to meters using the UTM Zone
        data = data.to_crs(utm_crs)

        # Create a GeoSeries for the target location (WGS84 degrees)
        target_gs = gpd.GeoSeries(
            [Point(ref_lon, ref_lat)],
            crs="EPSG:4326",
        )

        # Reprojects to meters and extract the point
        target_pt = target_gs.to_crs(utm_crs).iloc[0]

        # Calculate relative difference to ref_lon and ref_lat [in km]
        data['x_km'] = (data.geometry.x - target_pt.x) / 1000.0
        data['y_km'] = (data.geometry.y - target_pt.y) / 1000.0

        # Save UTM Zone and reference point
        data.attrs['utm_crs'] = utm_crs
        data.attrs['target_pt'] = target_pt

        return data

    except Exception as e:
        raise RuntimeError(f"Geodata preparation failed: {e}")


# -----------------------------------------------------------------------------
# Convert location back to WGS84 (Lat/Lon)
def back_to_wgs84(loc_km: List[float], data: gpd.GeoDataFrame) -> List[float]:
    """
    Convert location (Easting, Northing, Mw) back to WGS84 (Lat, Lon, Mw)

    Args:
        loc_km (List[float]): [x, y, Mw] of the solution in km.
        data (gpd.GeoDataFrame): Table with transformation metadata in .attrs.
    Returns:
        List[float]: [Lat, Lon, Mw] of the solution in deg.
    """
    # Prepare UTM Zone and reference point
    target_pt = data.attrs['target_pt']
    utm_crs = data.attrs['utm_crs']

    # Convert back to WGS84 (Lat/Lon)
    res_utm_x = target_pt.x + (loc_km[0] * 1000.0)
    res_utm_y = target_pt.y + (loc_km[1] * 1000.0)

    res_gs = gpd.GeoSeries([Point(res_utm_x, res_utm_y)], crs=utm_crs)
    res_wgs = res_gs.to_crs("EPSG:4326").iloc[0]

    loc_wgs = [res_wgs.y, res_wgs.x, loc_km[2]]  # [Lat, Lon, Mw]

    return loc_wgs


# -----------------------------------------------------------------------------
# Process vs30 values in the geopandas table
def process_vs30(data: gpd.GeoDataFrame, input_path: Path) -> gpd.GeoDataFrame:
    """
    Checks for missing Vs30 values (<= 0) and attempts to fill them using a SQL
    database. Saves an updated input file if any values were changed.

    Args:
        data (gpd.GeoDataFrame): Input geopandas table with 'vs30' 'lon' 'lat'.
        input_path (Path): Full path of the input file.
    Returns:
        gpd.GeoDataFrame: Updated geopandas table with new Vs30 values.
    """
    bad_mask = data['vs30'] <= 0
    n_bad = bad_mask.sum()

    if n_bad == 0:
        return data

    print(f"[*] Found {n_bad} invalid Vs30 values")
    data.loc[bad_mask, 'vs30'] = DEFAULT_VS30

    # SQLite database file
    print("[*] Check or download SQLite database")
    jshis.download_database(jshis.DB_FILE)

    # SQL engine
    print("[*] Initialize SQL engine")
    engine = jshis.init_sql_engine(jshis.DB_FILE)
    if not engine:
        print("[!] SQL engine could not be initialized!")
        return data

    # Extract Vs30 data from the SQL database
    print("[*] Extracting missing Vs30 data from SQL")
    for idx, row in data[bad_mask].iterrows():
        val = single_vs30(row['lon'], row['lat'], engine)
        if val:
            data.at[idx, 'vs30'] = val
        else:
            print(f"[!] No SQL data found at {row['lon']}, {row['lat']}")

    if hasattr(engine, 'dispose'):
        engine.dispose()

    print("[+] SUCCESS: Vs30 values assigned from SQL database")

    # Save the updated table
    save_updated_input(data, input_path)

    return data


# -----------------------------------------------------------------------------
# Get Vs30 for one spatial point (lon, lat)
def single_vs30(lon: float, lat: float, engine: Engine) -> Optional[float]:
    """
    Get Vs30 from SQL for one spatial point.

    Args:
        lon (float): Longitude [deg].
        lat (float): Latitude [deg].
        engine (Engine): Active SQLAlchemy engine instance.
    Returns:
        float: Vs30 value [m/s] from the SQL database.
        None: If not found.
    """
    params = {
        'ref_lon': lon,
        'ref_lat': lat,
        'delta': jshis.DELTA
    }
    res = jshis.get_vs30(params, engine)
    if res:
        return res.vs30
    else:
        return None


# -----------------------------------------------------------------------------
# Save updated input table with new Vs30 values
def save_updated_input(data: gpd.GeoDataFrame, original_path: Path) -> None:
    """
    Save updated data table into formated text file.

    Args:
        data (gpd.GeoDataFrame): Input geopandas table.
        original_path (Path): Full path of the input file.
    Returns:
        None.
    """
    output_path = original_path.with_name(f"UPDATED_{original_path.name}")
    try:
        export_cols = ['n', 'lat', 'lon', 'int', 'sig', 'vs30', 'text']
        out_df = data[export_cols].copy()
        out_df['n'] = out_df['n'].astype(int)

        header = ("# UPDATED INPUT FILE WITH VS30 EXTRACTED FROM J-SHIS\n"
                  "#  N  Latitude  Longitude  Int  Sigma  Vs30  Note\n")

        formats = {
            'n': '{:>4d}'.format,
            'lat': '{:9.5f}'.format,
            'lon': '{:10.5f}'.format,
            'int': '{:5.2f}'.format,
            'sig': '{:5.2f}'.format,
            'vs30': '{:6.1f}'.format,
            'text': '{:<}'.format,
        }

        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(header)
            f.write(out_df.to_string(header=False, index=False,
                                     formatters=formats))
        print(f"[+] SUCCESS: Saved to {output_path}")

    except Exception as e:
        print(f"[!] ERROR saving file: {e}")
