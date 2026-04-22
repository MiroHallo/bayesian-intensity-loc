#!/usr/bin/env python3
# =============================================================================
# EARTHQUAKE EPICENTER LOCATION FROM SEISMIC INTENSITY
#
# Author: Miroslav HALLO, Kyoto University
# E-mail: hallo.miroslav.2a@kyoto-u.ac.jp
# Tested with: Python 3.12.3, Jax 0.9.2, Matplotlib 3.10.8, NumPy 2.4.4,
#              Pandas 3.0.2, GeoPandas 1.1.3, psutil 7.2.2, Requests 2.33.1,
#              Shapely 2.1.2, SQLAlchemy 2.0.49
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

from datetime import datetime
from pathlib import Path

import geopandas as gpd
import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
import psutil
from shapely.geometry import Point

from config import INPUT, ScaleType
import utils.jshis_sqlite_query as jshis
import plotting as putil


# =============================================================================
# FUNCTIONS
# =============================================================================

# Forward problem - Japan (vectorized over stations, JAX)
@jax.jit
def forward_jma_intensity(
        mw: float, r: jnp.ndarray, h_top: float, vs30: jnp.ndarray
        ) -> tuple[jnp.ndarray, float]:
    """
    JMA instrumental seismic intensity (Morikawa and Fujiwara, 2013).

    Args:
        mw (float): Moment magnitude.
        r (jnp.ndarray): Distance array [km].
        h_top (float): Top depth of the rupture [km].
        vs30 (jnp.ndarray): Vs30 array [m/s].
    Returns:
        tuple: jnp.ndarray: Predicted JMA instrumental seismic intensity.
               float: Model uncertainty 1 sigma.
    """
    # Coefficients from Morikawa and Fujiwara (2013)
    mw01, mw02 = 8.2, 16.0
    a1, b1, c1, d1, e1 = -0.0321, -0.003736, 6.9301, 0.005078, 0.5
    sig_mod = 0.3493
    ps, vsmax, v0 = -0.5898, 1900.0, 350.0

    # Compute support terms
    r = jnp.sqrt(r**2 + h_top**2)
    mw1 = jnp.minimum(mw, mw01)
    vs30_limited = jnp.minimum(vs30, vsmax)
    gs = ps * jnp.log10(vs30_limited / v0)
    term_log = jnp.log10(r + (d1 * 10**(e1 * mw1)))

    # Main equation for JMA seismic intensity
    intensity = 2.0 * (a1 * ((mw1 - mw02)**2) + b1 * r + c1 - term_log + gs)

    return intensity, sig_mod


# -----------------------------------------------------------------------------
# Forward problem - USA (vectorized over stations, JAX)
@jax.jit
def forward_mmi_intensity(
        mw: float, r: jnp.ndarray, h_top: float, vs30: jnp.ndarray
        ) -> tuple[jnp.ndarray, float]:
    """
    Modified Mercalli Intensity (MMI) prediction (Atkinson et al., 2014).

    Args:
        mw (float): Moment magnitude.
        r (jnp.ndarray): Distance array [km].
        h_top (float): Top depth of the rupture [km].
        vs30 (jnp.ndarray): Vs30 array [m/s].
    Returns:
        tuple: jnp.ndarray: Predicted MMI instrumental seismic intensity.
               float: Model uncertainty 1 sigma.
    """
    # Coefficients from Atkinson et al. (2014)
    c1, c2, c3, c4, c5, c6 = 0.309, 1.864, -1.672, -0.00219, 1.77, -0.383
    h_sat = 14.0  # Saturation depth [km]
    v0 = 450.0  # Reference Vs30 [m/s]
    sig_mod = 0.5

    # Compute support terms
    refe = jnp.sqrt(r**2 + jnp.maximum(h_sat, h_top)**2)
    b = jnp.maximum(0.0, jnp.log10(refe / 50.0))
    amp_gain = -1.5 * jnp.log10(jnp.maximum(vs30, 150.0) / v0)

    # Intensity Prediction Equation (IPE for WNA)
    intensity = (c1 + c2 * mw + c3 * jnp.log10(refe) + c4 * refe
                 + c5 * b + c6 * mw * jnp.log10(refe)) + amp_gain

    return intensity, sig_mod


# -----------------------------------------------------------------------------
# Forward problem - EU (vectorized over stations, JAX)
@jax.jit
def forward_ems98_intensity(
        mw: float, r: jnp.ndarray, h_top: float, vs30: jnp.ndarray
        ) -> tuple[jnp.ndarray, float]:
    """
    European Macroseismic Scale (EMS-98) prediction.
    Based on Bindi et al. (2011) and Faenza and Michelini (2010) conversion.

    Args:
        mw (float): Moment magnitude.
        r (jnp.ndarray): Distance array [km].
        h_top (float): Top depth of the rupture [km].
        vs30 (jnp.ndarray): Vs30 array [m/s].
    Returns:
        tuple: jnp.ndarray: Predicted EMS-98 instrumental intensity.
               float: Model uncertainty 1 sigma.
    """
    # Coefficients for PGV on A-class rock (Bindi et al., 2011; table 5)
    e1, c1, c2, h, c3 = 2.305, -1.517, 0.326, 7.879, 0.0
    b1, b2 = 0.236, -0.00686
    r_ref, m_ref, m_h, b3 = 1.0, 5.0, 6.75, 0.0
    sig_mod_pgv = 0.332

    # Compute support terms
    refe = jnp.sqrt(r**2 + jnp.maximum(h, h_top)**2)
    fd = (c1 + c2 * (mw - m_ref)) * jnp.log10(refe/r_ref) - c3 * (refe-r_ref)
    fm = jnp.where(
        mw <= m_h,
        b1 * (mw - m_h) + b2 * ((mw - m_h)**2),
        b3 * (mw - m_h)
    )

    # Simplified site amplification
    v0 = 800.0  # Eurocode 8 reference rock
    amp_pgv = 0.3 * jnp.log10(v0 / jnp.maximum(vs30, 150.0))

    # log10(PGV) [cm/s] using European GMPE (Bindi et al., 2011)
    log_pgv = e1 + fd + fm + amp_pgv

    # Convert log10(PGV) to EMS-98 (Faenza and Michelini, 2010)
    intensity = 5.11 + 2.35 * log_pgv
    sig_mod_int = 0.35

    # Uncertainty (combined GMPE + GMICE)
    sig_mod = jnp.sqrt(sig_mod_pgv**2 + sig_mod_int**2)

    return intensity, sig_mod


# -----------------------------------------------------------------------------
# Likelihood function (one point of the grid, JAX)
@jax.jit
def eval_log_L(
        x_src: float, y_src: float, mw_src: float,
        x_st: jnp.ndarray, y_st: jnp.ndarray, vs30_st: jnp.ndarray,
        int_obs: jnp.ndarray, sig_obs: jnp.ndarray, h_top: float
        ) -> jnp.ndarray:
    """
    Evaluates the log-likelihood for a single grid point (hypothetical source).

    Args:
        x_src (float): Source Easting coordinate in the local grid [km].
        y_src (float): Source Northing coordinate in the local grid [km].
        mw_src (float): Source moment magnitude.
        x_st (jnp.ndarray): Array of station Easting coordinates [km].
        y_st (jnp.ndarray): Array of station Northing coordinates [km].
        vs30_st (jnp.ndarray): Array of station Vs30 values [m/s].
        int_obs (jnp.ndarray): Array of observed instrumental intensities.
        sig_obs (jnp.ndarray): Array of observational uncertainties (1 sigma).
        h_top (float): Top depth of the rupture [km].
    Returns:
        jnp.ndarray: Scalar log-likelihood value for the given source.
    """
    # Distance from the rupture top to all station
    r = jnp.sqrt((x_src - x_st)**2 + (y_src - y_st)**2)

    # Forward problem (predict synthetic intensities)
    if INPUT.scale == ScaleType.JMA:  # Japan (JMA Shindo)
        int_syn, sig_mod = forward_jma_intensity(mw_src, r, h_top, vs30_st)
    elif INPUT.scale == ScaleType.MMI:  # USA (Modified Mercalli)
        int_syn, sig_mod = forward_mmi_intensity(mw_src, r, h_top, vs30_st)
    elif INPUT.scale == ScaleType.EMS98:  # EU (European Macroseismic EMS-98)
        int_syn, sig_mod = forward_ems98_intensity(mw_src, r, h_top, vs30_st)

    # Combined variance (sum of squares of independent uncertainties)
    variance = sig_obs**2 + sig_mod**2 + 1e-9

    # Normalized residuals
    diff = (int_obs - int_syn)**2 / variance

    # Log-likelihood calculation (suma over all stations)
    # Log(L) = sum ( -0.5 * diff - log(sqrt(2 * pi * variance)) )
    log_L = jnp.sum(-0.5 * diff - jnp.log(jnp.sqrt(2.0 * jnp.pi * variance)))

    return log_L


# -----------------------------------------------------------------------------
# Check memory requirements by grid search and JAX vectorization
def check_memory_requirements(gx: jnp.ndarray, gy: jnp.ndarray, gz:
                              jnp.ndarray, x_st: jnp.ndarray) -> None:
    """
    Checks if the grid search dimensions fit into the available system RAM and
    estimates the memory load for JAX operations.

    Args:
        gx (jnp.ndarray): Vectorized (JAX) grid in x direction.
        gy (jnp.ndarray): Vectorized (JAX) grid in y direction.
        gz (jnp.ndarray): Vectorized (JAX) grid in z direction.
        x_st (jnp.ndarray): Vectorized (JAX) GeoPandas data.
    Returns:
        None.
    """
    try:
        backend = jax.lib.xla_bridge.get_backend().platform
    except Exception:
        backend = 'unknown'

    # Final 3D matrix size (X * Y * Z, float32 = 4 bytes per element)
    num_elements_3d = len(gx) * len(gy) * len(gz)
    res_size_gb = (num_elements_3d * 4) / 1e9

    # Peak memory for a single JAX slice (X * Y * number of stations)
    num_elements_2d_batch = len(gx) * len(gy) * len(x_st)
    jax_slice_gb = (num_elements_2d_batch * 4) / 1e9

    # Get available system RAM
    available_ram_gb = psutil.virtual_memory().available / 1e9

    print("[*] Memory Usage Forecast")
    print(f"    Grid dimensions:       {len(gx)} x {len(gy)} x {len(gz)}")
    print(f"    Number of stations:    {len(x_st)}")
    print(f"    Final 3D matrix (RAM): {res_size_gb:.2f} GB")
    print(f"    JAX peak per slice:    {jax_slice_gb:.2f} GB")
    print(f"    Available System RAM:  {available_ram_gb:.2f} GB")

    # Check RAM (3D matrix)
    if res_size_gb > available_ram_gb * 0.9:
        raise MemoryError(
            f"Insufficient RAM! The 3D grid requires {res_size_gb:.2f} GB!"
        )

    # Check RAM (JAX slice)
    if backend == 'cpu' and jax_slice_gb > available_ram_gb * 0.9:
        raise MemoryError(
            f"Insufficient RAM! The JAX slice requires {jax_slice_gb:.2f} GB!"
        )

    # Check JAX memory footprint
    if jax_slice_gb > 8.0:
        print("    ADVISORY: Consider reducing X-Y grid resolution")

    # Check JAX memory footprint
    if jax_slice_gb > 64.0:
        raise MemoryError(
            f"Slice size ({jax_slice_gb:.1f} GB) exceeds hard limit of 64 GB!"
        )

    print("[*] SUCCESS: Memory check passed")


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
    for idx, row in data[bad_mask].iterrows():
        data.at[idx, 'vs30'] = 350.0

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
        input_params = {
            'ref_lon': row['lon'],
            'ref_lat': row['lat'],
            'delta': jshis.DELTA,
        }
        res = jshis.get_vs30(input_params, engine)
        if res:
            data.at[idx, 'vs30'] = res.vs30
        else:
            print(f"[!] No SQL data found at {row['lon']}, {row['lat']}")

    if hasattr(engine, 'dispose'):
        engine.dispose()

    print("[+] SUCCESS: Vs30 values assigned from SQL database")

    # Save the updated table
    output_path = input_path.with_name(f"UPDATED_{input_path.name}")
    try:
        export_cols = ['n', 'lat', 'lon', 'int', 'sig', 'vs30', 'text']
        out_df = pd.DataFrame(data[export_cols])
        out_df['n'] = out_df['n'].astype(int)
        header_text = (
            "# UPDATED INPUT FILE WITH VS30 EXTRACTED FROM J-SHIS\n"
            "#  N  Latitude  Longitude  Int  Sigma  Vs30  Note\n"
        )
        out_format = {
            'n': '{:>4d}'.format,
            'lat': '{:9.5f}'.format,
            'lon': '{:10.5f}'.format,
            'int': '{:5.2f}'.format,
            'sig': '{:5.2f}'.format,
            'vs30': '{:6.1f}'.format,
            'text': '{:<}'.format,
        }
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(header_text)
            f.write(out_df.to_string(header=False, index=False,
                                     formatters=out_format, justify='left'))
        print(f"[+] SUCCESS: Updated file saved to: {output_path}")
    except Exception as e:
        print(f"[!] ERROR saving updated file: {e}")

    return data


# -----------------------------------------------------------------------------
# Main function
def main() -> None:
    """
    Main function for the location from instrumental seismic intensity.

    - Read input seismic intensity and Vs30 from the INPUT.input_file.
    - Prepare input data for the grid.
    - Prepare Vs30 for all sites.
    - Vectorize and evaluate probability in 3D model space (JAX).
    - Find ML/MAP and PM solutions with uncertainties.
    - Plot and save results.
    """
    print("-" * 50)
    print(f"[*] Seismic intensity scale: {INPUT.scale.value.upper()}")

    # Get the directory where the current script is located
    proj_root = Path(__file__).parent.resolve()

    # Create directory for results
    res_dir = proj_root / "results"
    res_dir.mkdir(parents=True, exist_ok=True)

    # Prepare timestamp (Format: yyyyMMdd_HHmmss) and outfile name pattern
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    outfile = f"{timestamp}_loc"

    # -------------------------------------------
    # Prepare input dictionary from the text file
    print("[*] Read input file")
    input_path = Path(INPUT.input_file)
    if not input_path.is_file():
        print(f"[!] ERROR: Input file '{INPUT.input_file}' not found!")
        return
    try:
        inp_data = pd.read_csv(
            input_path, sep=r'\s+', comment='#', header=None,
            names=['n', 'lat', 'lon', 'int', 'sig', 'vs30', 'text'],
            usecols=[0, 1, 2, 3, 4, 5, 6])
        # Check if all have numeric values
        numeric_cols = ['lat', 'lon', 'int', 'sig', 'vs30']
        for col in numeric_cols:
            if not pd.api.types.is_numeric_dtype(inp_data[col]):
                inp_data[col] = pd.to_numeric(inp_data[col], errors='coerce')
        if inp_data[numeric_cols].isna().any().any():
            print("[!] ERROR: Non-numeric values or missing data found!")
            return
        # Check if sig > 0
        if not (inp_data['sig'] > 0).all():
            print("[!] ERROR: Sigma must be positive (> 0).")
            return
    except Exception as e:
        print(f"[!] ERROR reading file: {e}")
        return

    print("[*] Prepare input data")
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
        [Point(INPUT.ref_lon, INPUT.ref_lat)],
        crs="EPSG:4326",
    )
    # Reprojects to meters and extract the point
    target_pt = target_gs.to_crs(utm_crs).iloc[0]

    # Calculate relative difference to ref_lon and ref_lat [in km]
    data['x_km'] = (data.geometry.x - target_pt.x) / 1000.0
    data['y_km'] = (data.geometry.y - target_pt.y) / 1000.0

    # Get seismic intensity color scale
    if INPUT.scale == ScaleType.JMA:  # Japan (JMA Shindo)
        data['color'] = data.apply(putil.get_jma_color, axis=1)
    elif INPUT.scale in [ScaleType.MMI, ScaleType.EMS98]:  # USA and EU
        data['color'] = data.apply(putil.get_mmi_color, axis=1)

    # -------------------------------------------
    # Check and Fill Vs30 using the external database
    data = process_vs30(data, input_path)

    # -------------------------------------------
    # Vectorization (JAX)
    print("[*] Vectorize data for JAX")

    # Vectorizing GeoPandas data for JAX
    x_st = jnp.array(data['x_km'].values, dtype=jnp.float32)
    y_st = jnp.array(data['y_km'].values, dtype=jnp.float32)
    vs30_st = jnp.array(data['vs30'].values, dtype=jnp.float32)
    int_obs = jnp.array(data['int'].values, dtype=jnp.float32)
    sig_obs = jnp.array(data['sig'].values, dtype=jnp.float32)

    # Vectorizing grid for JAX
    grid_x, grid_y, grid_z = INPUT.generate_grids()
    gx = jnp.array(grid_x, dtype=jnp.float32)
    gy = jnp.array(grid_y, dtype=jnp.float32)
    gz = jnp.array(grid_z, dtype=jnp.float32)
    h_top = jnp.array(INPUT.h_top, dtype=jnp.float32)

    # Run memory check
    check_memory_requirements(gx, gy, gz, x_st)

    # Vectorizing the likelihood function over the 2D grid (x, y)
    # in_axes defines which arguments are mapped (0) and which fixed (None)
    grid_search_2d = jax.jit(
        jax.vmap(
            jax.vmap(eval_log_L,
                     in_axes=(None, 0, None, None, None, None, None, None,
                              None)),
            in_axes=(0, None, None, None, None, None, None, None, None)
        ))

    # -------------------------------------------
    # Grid search for PDF (JAX)
    print("[*] Evaluate probability in 3D model space (Parallel-JAX)")
    print_step = max(1, len(gz) // 10)

    # Prepare log_PDF 3D array
    log_pdf_3d = np.full((len(gx), len(gy), len(gz)), -1e19, dtype=np.float32)

    for i, mw_val in enumerate(gz):
        # Loop over Mw
        slice_2d = grid_search_2d(
           gx, gy, mw_val, x_st, y_st, vs30_st, int_obs, sig_obs, h_top
           )
        # Transfer back to NumPy
        log_pdf_3d[:, :, i] = np.array(slice_2d.block_until_ready())
        # Print progress
        if i % print_step == 0 or i == len(gz) - 1:
            percent = (i + 1) / len(gz) * 100
            print(f"    Progress: {percent:5.1f}%")

    # Convert Log-Likelihood to PDF
    max_log = np.max(log_pdf_3d)
    pdf_3d = np.exp(log_pdf_3d - max_log)

    # Normalize total probability to 1.0
    pdf_3d = pdf_3d / np.sum(pdf_3d)

    print("[*] SUCCESS: 3D PDF evaluation finished")

    # -------------------------------------------
    # Process the PDF to find the solution
    print("[*] Estimating ML/MAP solution")

    # Maximum Likelihood (ML) / Maximum a Posteriori (MAP) solution
    idx = np.unravel_index(np.argmax(pdf_3d), pdf_3d.shape)
    loc_ix, loc_iy, loc_iz = int(idx[0]), int(idx[1]), int(idx[2])

    # The ML/MAP solution
    loc_res = [gx[loc_ix], gy[loc_iy], gz[loc_iz]]  # [Easting, Northing, Mw]

    # Convert back to WGS84 (Lat/Lon)
    res_utm_x = target_pt.x + (loc_res[0] * 1000.0)
    res_utm_y = target_pt.y + (loc_res[1] * 1000.0)
    res_gs = gpd.GeoSeries([Point(res_utm_x, res_utm_y)],
                           crs=utm_crs)
    res_wgs = res_gs.to_crs("EPSG:4326").iloc[0]
    loc_res_wgs = [res_wgs.y, res_wgs.x, loc_res[2]]  # [Lat, Lon, Mw]

    # -------------------------------------------
    # ML/MAP solution uncertainty (Marginal PDFs and Gaussian Fitting)
    print("[*] Estimating solution uncertainties from marginal PDFs")
    loc_xyz_sigma = [0.0, 0.0, 0.0]

    # Marginal PDFs
    # Marginal for Easting: Sum over Northing (axis 1) and Mw (axis 2)
    marginal_x = np.sum(pdf_3d, axis=(1, 2))
    # Marginal for Northing: Sum over Easting (axis 0) and Mw (axis 2)
    marginal_y = np.sum(pdf_3d, axis=(0, 2))
    # Marginal for Mw: Sum over Easting (axis 0) and Northing (axis 1)
    marginal_mw = np.sum(pdf_3d, axis=(0, 1))

    # Gaussian fit
    # Fit 2nd order polynomial to log(marginal): ln(y) = ax^2 + bx + c
    # The standard deviation sigma = sqrt(-1 / (2 * a))
    grid_list = [gx, gy, gz]
    marginal_list = [marginal_x, marginal_y, marginal_mw]

    for i in range(3):
        curr_grid = np.array(grid_list[i])
        curr_pdf = marginal_list[i]
        # Use only points where PDF is significantly positive to avoid log(0)
        mask = curr_pdf > (np.max(curr_pdf) * 0.03)
        if np.sum(mask) > 3:
            try:
                lny = np.log(curr_pdf[mask])
                coeffs = np.polyfit(curr_grid[mask], lny, 2)
                if coeffs[0] < 0:  # Must be concave down
                    sigma = np.sqrt(-1.0 / (2.0 * coeffs[0]))
                    loc_xyz_sigma[i] = sigma
                else:
                    loc_xyz_sigma[i] = np.nan
            except np.linalg.LinAlgError:
                loc_xyz_sigma[i] = np.nan
        else:
            loc_xyz_sigma[i] = np.nan

    # -------------------------------------------
    # Compute Posterior Mean solution
    print("[*] Compute Posterior Mean (PM) solution")
    total_p = np.sum(pdf_3d)
    x_mean = np.sum(gx * marginal_x) / total_p
    y_mean = np.sum(gy * marginal_y) / total_p
    mw_mean = np.sum(gz * marginal_mw) / total_p

    # Posterior Mean solution (PM)
    loc_pm = [x_mean, y_mean, mw_mean]  # [Easting, Northing, Mw]

    # Convert back to WGS84 (Lat/Lon)
    pm_utm_x = target_pt.x + (loc_pm[0] * 1000.0)
    pm_utm_y = target_pt.y + (loc_pm[1] * 1000.0)
    pm_gs = gpd.GeoSeries([Point(pm_utm_x, pm_utm_y)],
                          crs=utm_crs)
    pm_wgs = pm_gs.to_crs("EPSG:4326").iloc[0]
    loc_pm_wgs = [pm_wgs.y, pm_wgs.x, loc_pm[2]]  # [Lat, Lon, Mw]

    # -------------------------------------------
    # Save ML / MAP / PM solutions
    print("[*] Save results to a text file")
    header = "# SOLUTION FOR THE EARTHQUAKE EPICENTER LOCATION"
    map_header = "# Maximum Likelihood (ML) / Maximum a Posteriori (MAP)"
    map_cols = ("# Latitude, Longitude, Mw, Easting, Northing, E_sigma, "
                "N_sigma, Mw_sigma, E_2sigma, N_2sigma, Mw_2sigma [km]")
    pm_header = "# Posterior Mean solution (PM)"
    pm_cols = "# Latitude, Longitude, Mw, Easting, Northing [km]"

    # Prepare strings for ML / MAP
    map_row = (
        f"{loc_res_wgs[0]:10.5f} {loc_res_wgs[1]:10.5f} "
        f"{loc_res_wgs[2]:5.2f} {loc_res[0]:8.3f} {loc_res[1]:8.3f} "
        f"{loc_xyz_sigma[0]:8.3f} {loc_xyz_sigma[1]:8.3f} "
        f"{loc_xyz_sigma[2]:8.3f} {loc_xyz_sigma[0]*2:8.3f} "
        f"{loc_xyz_sigma[1]*2:8.3f} {loc_xyz_sigma[2]*2:8.3f}"
    )

    # Prepare strings for PM
    pm_row = (
        f"{loc_pm_wgs[0]:10.5f} {loc_pm_wgs[1]:10.5f} "
        f"{loc_pm_wgs[2]:5.2f} {loc_pm[0]:8.3f} {loc_pm[1]:8.3f}"
    )

    # Save into text file
    output_path = res_dir / f"{outfile}.txt"
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(f"{header}\n")
            f.write(f"# {'-' * 68}\n")
            f.write(f"{map_header}\n")
            f.write(f"{map_cols}\n")
            f.write(f"{map_row}\n")
            f.write(f"# {'-' * 68}\n")
            f.write(f"{pm_header}\n")
            f.write(f"{pm_cols}\n")
            f.write(f"{pm_row}\n")
        print(f"[+] SUCCESS: Results saved to: {output_path}")
    except IOError as e:
        print(f"[!] ERROR: Could not save results to file: {e}")

    # -------------------------------------------
    # Evaluate misfit for ML/MAP solution
    print("[*] Evaluating misfit for the ML/MAP solution")

    # loc_res contains [Easting, Northing, Mw]
    mw_bes = loc_res[2]

    # Calculate distances from the best source to all stations [km]
    r_bes = np.sqrt((loc_res[0] - x_st)**2 + (loc_res[1] - y_st)**2)

    # Compute synthetic intensities for all stations
    if INPUT.scale == ScaleType.JMA:  # Japan (JMA Shindo)
        int_synth, _ = forward_jma_intensity(mw_bes, r_bes, h_top, vs30_st)
    elif INPUT.scale == ScaleType.MMI:  # USA (Modified Mercalli)
        int_synth, _ = forward_mmi_intensity(mw_bes, r_bes, h_top, vs30_st)
    elif INPUT.scale == ScaleType.EMS98:  # EU (European Macroseismic EMS-98)
        int_synth, _ = forward_ems98_intensity(mw_bes, r_bes, h_top, vs30_st)

    # Compute misfit (Observed - Synthetic)
    int_misfit = int_obs - int_synth

    # Add these results back to our GeoDataFrame
    data['int_synth'] = np.array(int_synth)
    data['int_misfit'] = np.array(int_misfit)

    # -------------------------------------------
    # Plot results
    print("[*] Plot results")

    output_path = res_dir / f"{outfile}_map.png"
    putil.plot_station_map(data, loc_res, loc_pm, loc_res_wgs, loc_pm_wgs, gx,
                           gy, output_path)

    output_path = res_dir / f"{outfile}_pdf_cross_section.png"
    putil.plot_slices(pdf_3d, gx, gy, gz, loc_ix, loc_iy, loc_iz, output_path)

    output_path = res_dir / f"{outfile}_pdf_marginal.png"
    putil.plot_marginal_pdf(pdf_3d, gx, gy, gz, loc_res, loc_pm, output_path)

    output_path = res_dir / f"{outfile}_misfit.png"
    station_colors = np.array(data['color'].tolist())
    putil.plot_misfits(
        loc_ml=loc_res,
        x_st=data['x_km'].values,
        y_st=data['y_km'].values,
        i_misfit=data['int_misfit'].values,
        i_colors=station_colors,
        output_path=output_path
    )

    print("[*] SUCCESS: All done")


# -----------------------------------------------------------------------------
# Entry point
if __name__ == "__main__":
    main()
