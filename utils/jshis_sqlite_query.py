#!/usr/bin/env python3
# =============================================================================
# SQL DATA LOADER AND QUERY TOOL FOR VS30
#
# Author: Miroslav HALLO, Kyoto University
# E-mail: hallo.miroslav.2a@kyoto-u.ac.jp
# Tested with: Python 3.12.3, Pandas 3.0.2, GeoPandas 1.1.3
#              SQLAlchemy 2.0.48, Requests 2.33.1, Shapely 2.1.2
# Data source: https://doi.org/10.5281/zenodo.19379171
# Description: Optimized tool for spatial querying of J-SHIS derived Vs30
#              and Site Amplification data for Japan.
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
from pathlib import Path

import pandas as pd
import geopandas as gpd
import requests
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine
from shapely.geometry import Point


# =============================================================================
# CLASSES
# =============================================================================

@dataclass
class InputClass:
    ref_lon: float
    ref_lat: float
    delta: float = 0.02


@dataclass
class Vs30Class:
    lon: float
    lat: float
    vs30: float
    af: float
    dist_km: float


# =============================================================================
# GLOBAL PARAMETERS
# =============================================================================

# Input and Output files for standalone run
INPUT_FILE = 'example_targets.txt'
OUTPUT_FILE = 'example_results_vs30.txt'

# Filename of the SQLite database
DB_FILE = 'Vs30_Japan_JSHIS_derived.sqlite'

# Location search window [deg]
DELTA = 0.02


# =============================================================================
# FUNCTIONS
# =============================================================================

# Initialize SQL (SQLite) engine
def init_sql_engine(db_file: str) -> Engine | None:
    """
    Initialize SQL (SQLite) engine in read-only mode.

    Args:
        db_file (str): Text string containing filename of the SQLite database.
    Returns:
        engine (Engine): Active SQLAlchemy engine instance.
        None: If SQLite database file not found.
    """
    # Check if the SQLite database file exists
    full_db_path = Path(__file__).resolve().parent / db_file
    if not full_db_path.is_file():
        print(f"[!] ERROR: Database '{full_db_path.absolute()}' not found!")
        return None

    # Start SQL engine
    engine = create_engine(f"sqlite:///{full_db_path.absolute()}?mode=ro")
    return engine


# -----------------------------------------------------------------------------
# Get SQL database info
def get_db_info(engine: Engine) -> None:
    """
    Print database information from the db_info table.

    Args:
        engine (Engine): Active SQLAlchemy engine instance.
    Returns:
        None
    """
    with engine.connect() as conn:
        try:
            info_df = pd.read_sql_table("db_info", conn)
            for _, row in info_df.iterrows():
                print(f"{row['parameter'].upper()}: {row['value']}")
        except Exception:
            print("[!] Table 'db_info' not available!")


# -----------------------------------------------------------------------------
# Download SQLite database
def download_database(db_file: str) -> None:
    """
    Download SQLite from Zenodo repository DOI:10.5281/zenodo.19379171.

    Args:
        db_file (str): Text string containing filename of the SQLite database.
    Returns:
        None
    """
    # DOI for the SQLite database
    doi = "10.5281/zenodo.19379171"

    # Check if the SQLite database file exists
    full_db_path = Path(__file__).resolve().parent / db_file
    if full_db_path.is_file():
        return

    try:
        rdoi = requests.get(f"https://doi.org/{doi}", allow_redirects=True,
                            timeout=60)
        rdoi.raise_for_status()
        record_id = rdoi.url.rstrip('/').split('/')[-1]
        down_url = (f"https://zenodo.org/records/{record_id}/"
                    f"files/{db_file}?download=1")
        print(f"[*] Downloading from Zenodo (ID: {record_id}). Please wait...")
        with requests.get(down_url, stream=True) as r_down:
            r_down.raise_for_status()
            # Get total size
            total_size = int(r_down.headers.get('content-length', 0))
            downloaded = 0
            # Download
            with open(full_db_path, 'wb') as f:
                for chunk in r_down.iter_content(chunk_size=1024*1024):  # 1 MB
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)
                        if total_size > 0:
                            done = int(50 * downloaded / total_size)
                            print(f"\r    Progress: [{'=' * done}"
                                  f"{' ' * (50-done)}] "
                                  f"{downloaded/1024/1024:.1f} MB", end="")

        print(f"\n[+] Download complete: {full_db_path.absolute()}")

    except Exception as e:
        print(f"[!] ERROR: Failed to download SQlite database: {e}")
        if full_db_path.exists():
            full_db_path.unlink()


# -----------------------------------------------------------------------------
# Get Vs30 from SQL database
def get_vs30(input_params: dict, engine: Engine) -> Vs30Class | None:
    """
    Extract Vs30 data from SQL database.

    Args:
        input_params (dict): Dictionary with 'ref_lon', 'ref_lat', 'delta'.
        engine (Engine): Active SQLAlchemy engine instance.
    Returns:
        Vs30Class: Object containing 'lon', 'lat', 'vs30', 'af', 'dist_km'.
        None: If no data is found in the search window.
    """
    # Creates input object
    inp = InputClass(**input_params)

    # Prepare parameters for SQL query
    sql_params = {
        'lon_min': inp.ref_lon - inp.delta,
        'lon_max': inp.ref_lon + inp.delta,
        'lat_min': inp.ref_lat - inp.delta,
        'lat_max': inp.ref_lat + inp.delta,
    }

    # Define SQL query
    query = text("""
        SELECT * FROM vs30_data
        WHERE lon BETWEEN :lon_min AND :lon_max
        AND lat BETWEEN :lat_min AND :lat_max
    """)

    # Read SQL query
    with engine.connect() as conn:
        db_data = pd.read_sql(query, conn, params=sql_params)
    if db_data.empty:
        return None

    # Convert data to geopandas (WGS84 degrees)
    geo_data = gpd.GeoDataFrame(
        db_data,
        geometry=gpd.points_from_xy(db_data['lon'], db_data['lat']),
        crs="EPSG:4326",
    )
    # Find the best UTM Zone
    utm_crs = geo_data.estimate_utm_crs()
    # Reprojects to meters using the UTM Zone
    geo_data_metric = geo_data.to_crs(utm_crs)

    # Create a GeoSeries for the target location (WGS84 degrees)
    target_gs = gpd.GeoSeries(
        [Point(inp.ref_lon, inp.ref_lat)],
        crs="EPSG:4326",
    )
    # Reprojects to meters and extract the point
    target_pt = target_gs.to_crs(utm_crs).iloc[0]

    # Calculate distances (kilometers)
    geo_data_metric['dist_km'] = (
        geo_data_metric.geometry.distance(target_pt) / 1000.0
    )

    # Find the closest
    closest_idx = geo_data_metric['dist_km'].idxmin()
    res_data = geo_data_metric.loc[closest_idx]

    # Results
    res = Vs30Class(
        lon=res_data['lon'],
        lat=res_data['lat'],
        vs30=res_data['vs30'],
        af=res_data['af'],
        dist_km=res_data['dist_km'],
    )
    return res


# -----------------------------------------------------------------------------
# Main function
def main():
    """
    Main function for standalone execution.

    - Download SQLite database DB_FILE (if not exists).
    - Initialize SQL engine.
    - Read target locations from the INPUT_FILE.
    - Extract Vs30 values.
    - Save results into the OUTPUT_FILE.
    """
    print("-" * 50)

    # SQLite database file
    print("[*] Check or download SQLite database")
    download_database(DB_FILE)

    # SQL engine
    print("[*] Initialize SQL engine")
    engine = init_sql_engine(DB_FILE)
    if not engine:
        print("[!] SQL engine could not be initialized!")
        return

    print("[*] Database Info")
    get_db_info(engine)

    # Prepare input dictionary of target locations (from the text file)
    print("[*] Read input file")
    input_path = Path(INPUT_FILE)
    if not input_path.is_file():
        print(f"[!] ERROR: Input file '{INPUT_FILE}' not found!")
        return
    try:
        targets = pd.read_csv(input_path, sep=r'\s+', comment='#', header=None,
                              names=['ref_lon', 'ref_lat'], usecols=[0, 1])
    except Exception as e:
        print(f"[!] ERROR reading file: {e}")
        return

    # Extract Vs30 data from the SQL database
    print("[*] Extract Vs30 data")
    results_list = []
    for _, row in targets.iterrows():
        input_params = {
            'ref_lon': row['ref_lon'],
            'ref_lat': row['ref_lat'],
            'delta': DELTA,
        }

        res = get_vs30(input_params, engine)

        res_row = {
            'target_lon': row['ref_lon'],
            'target_lat': row['ref_lat'],
            'lon': res.lon if res else None,
            'lat': res.lat if res else None,
            'vs30': res.vs30 if res else None,
            'af': res.af if res else None,
            'dist_km': res.dist_km if res else None,
        }
        results_list.append(res_row)

    # Save Vs30 data results in the text file
    print("[*] Save results in file")
    if results_list:
        out = pd.DataFrame(results_list)
        header_text = (
            f"# RESULTS OF VS30 FOR TARGET LOCATIONS\n"
            f"# Generated on: {pd.Timestamp.now()}\n"
            f"# Target_Lon Target_Lat Lon Lat Vs30(m/s) AF(-) Dist(km)\n"
        )
        out_format = {
            'target_lon': '{:.4f}'.format,
            'target_lat': '{:.4f}'.format,
            'lon':        '{:.4f}'.format,
            'lat':        '{:.4f}'.format,
            'vs30':       '{:.1f}'.format,
            'af':         '{:.4f}'.format,
            'dist_km':    '{:.3f}'.format,
        }
        out_text = out.to_string(index=False, header=False, justify='left',
                                 na_rep='NaN', formatters=out_format)
        with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
            f.write(header_text)
            f.write(out_text)
        print(f"[+] SUCCESS: Results saved to '{OUTPUT_FILE}'")
    else:
        print("[!] No matching data found for any input coordinates!")


# -----------------------------------------------------------------------------
# Entry point
if __name__ == "__main__":
    main()
