from dataclasses import dataclass, field, fields, replace
from typing import Dict, List, Optional, Sequence
import numpy as np
import pandas as pd
import xarray as xr 
import matplotlib.pyplot as plt
from scipy.optimize import dual_annealing
import soil_moisture_model
from soil_moisture_model import SoilWaterParams, simulate_soil_water_balance, params_as_dict
import sys
import os

if len(sys.argv) > 1:
    data_dir = sys.argv[1]
else:
    data_dir = "./"

icosStations = pd.read_csv("/mnt/shared/pyrealm2/ICOS_stations_processed.csv")
icosStations = icosStations[icosStations['Site type'].isin([
    'croplands','grasslands','forest','open shrublands',
    'savannas','closed shrublands','evergreen needleleaf forests',
    'evergreen broadleaf forests','deciduous needleleaf forests',
    'deciduous needleleaf forests','mixed forests'
])]

# Use icosSites if it exists, otherwise fall back to icosStations
sites_df = icosSites if "icosSites" in globals() else icosStations

# Load existing best_params if available
existing_sites = set()
best_params_file = f"{data_dir}/best_params_by_site.parquet"
if os.path.exists(best_params_file):
    existing_sites = set(pd.read_parquet(best_params_file).index)

all_results = []
best_params_records = []
failed_sites = []
failed_swc = []
failed_calibration = []

base_params = SoilWaterParams()
all_param_names = [f.name for f in fields(SoilWaterParams)]

# Bounds for all SoilWaterParams fields + lai_to_biomass_multiplier (last)
param_bounds = {
    "initial_soil_moisture_mm": (5.0, 250.0),
    "base_soil_water_capacity_mm": (60.0, 400.0),
    "base_drainage_coeff": (0.01, 0.30),
    "base_runoff_fraction_max": (0.02, 0.95),
    "runoff_exponent": (0.5, 4.0),
    "field_capacity_fraction": (0.40, 0.95),
    "reference_clay_pct": (5.0, 60.0),
    "awc_mm_per_clay_pct": (0.0, 3.0),
    "drainage_clay_sensitivity": (0.0, 0.06),
    "runoff_clay_sensitivity": (0.0, 0.06),
    "stress_clay_sensitivity": (-0.01, 0.02),
    "temp_base_c": (-5.0, 8.0),
    "pet_coeff_mm_per_degday": (0.1, 1.5),
    "reference_elevation_m": (-100.0, 2500.0),
    "elev_pet_sensitivity_per_km": (-0.10, 0.30),
    "kc_min": (0.05, 1.00),
    "kc_max": (0.50, 2.00),
    "biomass_half_sat_gC_m2": (20.0, 800.0),
    "interception_max_mm": (0.0, 8.0),
    "interception_biomass_half_sat_gC_m2": (20.0, 800.0),
    "uptake_max_factor": (1.0, 2.5),
    "uptake_biomass_half_sat_gC_m2": (20.0, 800.0),
    "freeze_trigger_weeks": (1.0, 12.0),
    "thaw_trigger_temp_c": (-5.0, 5.0),
    "frozen_infiltration_fraction": (0.01, 0.80),
    "frozen_drainage_multiplier": (0.01, 1.00),
    "frozen_uptake_multiplier": (0.01, 1.00),
    "frozen_pet_multiplier": (0.01, 1.00),
}
bounds = [param_bounds[n] for n in all_param_names] + [(20.0, 1000.0)]  # lai multiplier last

def vector_to_full_params(x):
    vals = {n: float(v) for n, v in zip(all_param_names, x[:-1])}
    vals["freeze_trigger_weeks"] = int(np.clip(np.round(vals["freeze_trigger_weeks"]), 1, 20))
    return replace(base_params, **vals), float(x[-1])

for site_name in sites_df["Id"].dropna().unique():
    # Skip if already calibrated
    if site_name in existing_sites:
        print(f"Skipping {site_name} (already calibrated)")
        continue
    
    try:
        # ---------- Site/model inputs ----------
        D_site = xr.open_dataset(f"/mnt/shared/pyrealm2/inputData/{site_name}_weekly_final_variables.nc")
        site_meta = sites_df.loc[sites_df["Id"] == site_name].iloc[0]
        elevation = float(site_meta["Elevation above sea"])
        plant_type = site_meta["plant_type"] if "plant_type" in site_meta.index else np.nan
        clay_pct = float(D_site.attrs["clay_fraction"])

        # ---------- Observed SWC ----------
        try:
            D_flux = pd.read_csv(
                f"/mnt/shared/pyrealm2/ICOS_ETC_ARCHIVE/tmp_{site_name}/ICOSETC_{site_name}_FLUXNET_HH_L2.csv"
            )
            site_csv_df = D_flux.copy()
            site_csv_df["TIMESTAMP"] = pd.to_datetime(
                site_csv_df["TIMESTAMP_START"].astype(str), format="%Y%m%d%H%M"
            )
            site_csv_df["SWC_F_MDS_1"] = site_csv_df["SWC_F_MDS_1"].replace(-9999, np.nan)

            swc_weekly_site = (
                site_csv_df.assign(
                    time=site_csv_df["TIMESTAMP"].dt.to_period("W-FRI").dt.end_time.dt.normalize()
                )
                .groupby("time", as_index=False)["SWC_F_MDS_1"]
                .mean()
            )
        except Exception as e:
            failed_swc.append((site_name, str(e)))
            swc_weekly_site = pd.DataFrame(columns=["time", "SWC_F_MDS_1"])

        # ---------- Objective ----------
        def objective(x):
            try:
                p, lai_mult = vector_to_full_params(x)
                if p.kc_max <= p.kc_min:
                    return 1e9

                biomass = D_site.modis_lai[:, 0, 0].data * lai_mult
                sim_in = pd.DataFrame({
                    "time": D_site.time.data,
                    "temp_c": D_site.temperature_celcius[:, 0, 0].data,
                    "precip_mm": D_site.precipitation_mm[:, 0, 0].data,
                    "biomass_gC_m2": biomass,
                })

                out_tmp = simulate_soil_water_balance(
                    sim_in, date_col="time", elevation_m=elevation, clay_pct=clay_pct, params=p
                )
                out_tmp["time"] = pd.to_datetime(out_tmp["time"])

                merged = (
                    out_tmp[["time", "soil_moisture_mm"]]
                    .merge(swc_weekly_site, on="time", how="inner")
                    .dropna(subset=["soil_moisture_mm", "SWC_F_MDS_1"])
                )
                if len(merged) < 20:
                    return 1e8

                rmse_val = np.sqrt(np.mean((merged["soil_moisture_mm"] - merged["SWC_F_MDS_1"]) ** 2))
                return float(rmse_val)
            except Exception:
                return 1e10

        try:
            opt = dual_annealing(
                objective,
                bounds=bounds,
                maxiter=120,
                seed=42,
                no_local_search=False
            )
            best_x = opt.x
            best_rmse = float(opt.fun)
        except Exception as e:
            failed_calibration.append((site_name, str(e)))
            continue

        # ---------- Run best simulation ----------
        best_params, best_lai_mult = vector_to_full_params(best_x)
        biomass_best = D_site.modis_lai[:, 0, 0].data * best_lai_mult
        example_site = pd.DataFrame({
            "time": D_site.time.data,
            "temp_c": D_site.temperature_celcius[:, 0, 0].data,
            "precip_mm": D_site.precipitation_mm[:, 0, 0].data,
            "biomass_gC_m2": biomass_best,
        })

        out_site = simulate_soil_water_balance(
            example_site,
            date_col="time",
            elevation_m=elevation,
            clay_pct=clay_pct,
            params=best_params
        )
        out_site["time"] = pd.to_datetime(out_site["time"])
        out_site = out_site.merge(swc_weekly_site, on="time", how="left")

        out_site["site_name"] = site_name
        out_site["plant_type"] = plant_type
        out_site["lai_to_biomass_multiplier"] = best_lai_mult
        out_site["calib_rmse"] = best_rmse
        all_results.append(out_site)

        rec = {
            "site_name": site_name,
            "plant_type": plant_type,
            "calib_rmse": best_rmse,
            "lai_to_biomass_multiplier": best_lai_mult,
        }
        rec.update(params_as_dict(best_params))
        best_params_records.append(rec)
        print(f"Completed calibration for site: {site_name}")

    except Exception as e:
        failed_sites.append((site_name, str(e)))

    if all_results:
        results_all_sites = (
            pd.concat(all_results, ignore_index=True)
            .set_index(["time", "site_name", "plant_type"])
            .sort_index()
        )
        results_all_sites.to_parquet(f"{data_dir}/results_all_sites.parquet")
    else:
        print("Warning: No successful calibrations to save.")

    if best_params_records:
        best_params_by_site = pd.DataFrame(best_params_records).set_index("site_name").sort_index()
        best_params_by_site.to_parquet(f"{data_dir}/best_params_by_site.parquet")
    else:
        print("Warning: No best parameters to save.")

print(f"Failed sites: {len(failed_sites)}, Failed SWC: {len(failed_swc)}, Failed calibration: {len(failed_calibration)}")
