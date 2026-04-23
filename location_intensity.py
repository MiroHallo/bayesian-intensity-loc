#!/usr/bin/env python3
# =============================================================================
# EARTHQUAKE EPICENTER LOCATION FROM SEISMIC INTENSITY
#
# Author: Miroslav HALLO, Kyoto University
# E-mail: hallo.miroslav.2a@kyoto-u.ac.jp
# Tested with: Python 3.12.3, Jax 0.9.2, NumPy 2.4.4, psutil 7.2.2
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

import jax
import jax.numpy as jnp
import numpy as np
import psutil

from config import INPUT, ScaleType
from constants import (get_jma_color, get_mmi_color,
                       JMA_LEGEND, JMA_LEGEND_HIST, MMI_LEGEND)
from geodata import (load_input_data, prepare_geo, back_to_wgs84, process_vs30)
from plotting import (plot_slices, plot_marginal_pdf,
                      plot_misfits, plot_station_map)


# =============================================================================
# FUNCTIONS
# =============================================================================

# Forward problem - JMA Shindo (vectorized over stations, JAX)
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
# Forward problem - MMI (vectorized over stations, JAX)
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
    h_top = jnp.maximum(h_sat, h_top)
    v0 = 450.0  # Reference Vs30 [m/s]
    sig_mod = 0.5

    # Compute support terms
    refe = jnp.sqrt(r**2 + h_top**2)
    b = jnp.maximum(0.0, jnp.log10(refe / 50.0))
    amp_gain = -1.5 * jnp.log10(jnp.maximum(vs30, 150.0) / v0)

    # Intensity Prediction Equation (IPE for WNA)
    intensity = (c1 + c2 * mw + c3 * jnp.log10(refe) + c4 * refe
                 + c5 * b + c6 * mw * jnp.log10(refe)) + amp_gain

    return intensity, sig_mod


# -----------------------------------------------------------------------------
# Forward problem - EMS-98 (vectorized over stations, JAX)
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
    e1, c1, c2, c3 = 2.305, -1.517, 0.326, 0.0
    h = 7.879  # Saturation depth [km]
    h_top = jnp.maximum(h, h_top)
    b1, b2 = 0.236, -0.00686
    r_ref, m_ref, m_h, b3 = 1.0, 5.0, 6.75, 0.0
    sig_mod_pgv = 0.332

    # Compute support terms
    refe = jnp.sqrt(r**2 + h_top**2)
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
    # Prepare input data from the text file
    print("[*] Read input file")
    try:
        input_path = Path(INPUT.input_file)
        inp_data = load_input_data(input_path)
    except (FileNotFoundError, ValueError, RuntimeError) as e:
        print(f"[!] ERROR: {e}")
        return

    print("[*] Prepare input data")
    try:
        data = prepare_geo(inp_data, INPUT.ref_lon, INPUT.ref_lat)
    except Exception as e:
        print(f"[!] ERROR: {e}")
        return

    # Get seismic intensity color scale
    if INPUT.scale == ScaleType.JMA:  # JMA Shindo
        data['color'] = data.apply(
            lambda row: get_jma_color(row['int'], row.get('text')),
            axis=1
        )
    elif INPUT.scale in [ScaleType.MMI, ScaleType.EMS98]:  # MMI/EMS-98
        data['color'] = data['int'].apply(get_mmi_color)

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
    try:
        check_memory_requirements(gx, gy, gz, x_st)
    except (MemoryError) as e:
        print(f"[!] ERROR: {e}")
        return

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
    loc_res_wgs = back_to_wgs84(loc_res, data)  # [Lat, Lon, Mw]

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
    loc_pm_wgs = back_to_wgs84(loc_pm, data)  # [Lat, Lon, Mw]

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

    # Prepare color legend
    if INPUT.scale == ScaleType.JMA:  # JMA Shindo
        has_special_flags = data['text'].isin(['e', 'E', 'S']).any()
        if has_special_flags:
            legend_levels = JMA_LEGEND_HIST
        else:
            legend_levels = JMA_LEGEND
    elif INPUT.scale in [ScaleType.MMI, ScaleType.EMS98]:  # MMI/EMS-98
        legend_levels = MMI_LEGEND

    output_path = res_dir / f"{outfile}_map.png"
    plot_station_map(data, loc_res, loc_pm, loc_res_wgs, loc_pm_wgs, gx,
                     gy, legend_levels, output_path)

    output_path = res_dir / f"{outfile}_pdf_cross_section.png"
    plot_slices(pdf_3d, gx, gy, gz, loc_ix, loc_iy, loc_iz, output_path)

    output_path = res_dir / f"{outfile}_pdf_marginal.png"
    plot_marginal_pdf(pdf_3d, gx, gy, gz, loc_res, loc_pm, output_path)

    output_path = res_dir / f"{outfile}_misfit.png"
    station_colors = np.array(data['color'].tolist())
    plot_misfits(
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
