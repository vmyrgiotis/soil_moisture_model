from dataclasses import dataclass, field, fields, replace
from typing import Dict, List, Optional, Sequence
import numpy as np
import pandas as pd


@dataclass
class SoilWaterParams:
    initial_soil_moisture_mm: float = 50.0

    base_soil_water_capacity_mm: float = 160.0
    base_drainage_coeff: float = 0.10
    base_runoff_fraction_max: float = 0.45
    runoff_exponent: float = 2.0
    field_capacity_fraction: float = 0.70

    reference_clay_pct: float = 25.0
    awc_mm_per_clay_pct: float = 1.2
    drainage_clay_sensitivity: float = 0.015
    runoff_clay_sensitivity: float = 0.010
    stress_clay_sensitivity: float = 0.003

    temp_base_c: float = 0.0
    pet_coeff_mm_per_degday: float = 0.55

    reference_elevation_m: float = 0.0
    elev_pet_sensitivity_per_km: float = 0.08

    kc_min: float = 0.30
    kc_max: float = 1.20
    biomass_half_sat_gC_m2: float = 150.0

    interception_max_mm: float = 3.0
    interception_biomass_half_sat_gC_m2: float = 120.0

    uptake_max_factor: float = 1.35
    uptake_biomass_half_sat_gC_m2: float = 140.0

    freeze_trigger_weeks: int = 4
    thaw_trigger_temp_c: float = 0.0
    frozen_infiltration_fraction: float = 0.15
    frozen_drainage_multiplier: float = 0.20
    frozen_uptake_multiplier: float = 0.15
    frozen_pet_multiplier: float = 0.25


CALIBRATABLE_PARAMS = [
    "initial_soil_moisture_mm",
    "base_soil_water_capacity_mm",
    "base_drainage_coeff",
    "base_runoff_fraction_max",
    "runoff_exponent",
    "field_capacity_fraction",
    "reference_clay_pct",
    "awc_mm_per_clay_pct",
    "drainage_clay_sensitivity",
    "runoff_clay_sensitivity",
    "stress_clay_sensitivity",
    "temp_base_c",
    "pet_coeff_mm_per_degday",
    "reference_elevation_m",
    "elev_pet_sensitivity_per_km",
    "kc_min",
    "kc_max",
    "biomass_half_sat_gC_m2",
    "interception_max_mm",
    "interception_biomass_half_sat_gC_m2",
    "uptake_max_factor",
    "uptake_biomass_half_sat_gC_m2",
    "frozen_infiltration_fraction",
    "frozen_drainage_multiplier",
    "frozen_uptake_multiplier",
    "frozen_pet_multiplier",
]


def params_to_vector(params: SoilWaterParams, names: Sequence[str] = CALIBRATABLE_PARAMS) -> np.ndarray:
    return np.array([getattr(params, n) for n in names], dtype=float)


def vector_to_params(x: Sequence[float], base: SoilWaterParams, names: Sequence[str] = CALIBRATABLE_PARAMS) -> SoilWaterParams:
    updates = {n: float(v) for n, v in zip(names, x)}
    return replace(base, **updates)


def params_as_dict(params: SoilWaterParams) -> Dict[str, float]:
    return {f.name: getattr(params, f.name) for f in fields(params)}


def biomass_to_kc(biomass_gC_m2, kc_min, kc_max, half_sat):
    biomass = np.maximum(np.asarray(biomass_gC_m2, dtype=float), 0.0)
    frac = biomass / (biomass + half_sat + 1e-12)
    return kc_min + (kc_max - kc_min) * frac


def biomass_interception_capacity_mm(biomass_gC_m2, max_interception_mm, half_sat):
    biomass = np.maximum(np.asarray(biomass_gC_m2, dtype=float), 0.0)
    frac = biomass / (biomass + half_sat + 1e-12)
    return max_interception_mm * frac


def biomass_uptake_factor(biomass_gC_m2, max_factor, half_sat):
    biomass = np.maximum(np.asarray(biomass_gC_m2, dtype=float), 0.0)
    frac = biomass / (biomass + half_sat + 1e-12)
    return 1.0 + (max_factor - 1.0) * frac


def pressure_from_elevation_kpa(elevation_m):
    z = float(elevation_m)
    return 101.3 * ((293.0 - 0.0065 * z) / 293.0) ** 5.26


def elevation_pet_multiplier(elevation_m, reference_elevation_m=0.0, sensitivity_per_km=0.08):
    dz_km = (float(elevation_m) - reference_elevation_m) / 1000.0
    mult = 1.0 + sensitivity_per_km * dz_km
    return float(np.clip(mult, 0.75, 1.35))


def temperature_pet_weekly(temp_c, elevation_m, coeff_mm_per_degday, temp_base_c=0.0,
                           reference_elevation_m=0.0, elev_pet_sensitivity_per_km=0.08):
    temp = np.asarray(temp_c, dtype=float)
    degree_days = np.maximum(temp - temp_base_c, 0.0) * 7.0
    base_pet = coeff_mm_per_degday * degree_days
    elev_mult = elevation_pet_multiplier(
        elevation_m,
        reference_elevation_m=reference_elevation_m,
        sensitivity_per_km=elev_pet_sensitivity_per_km
    )
    return base_pet * elev_mult


def clay_modified_properties(clay_pct, params: SoilWaterParams):
    clay = float(np.clip(clay_pct, 0.0, 80.0))
    dclay = clay - params.reference_clay_pct

    soil_water_capacity = params.base_soil_water_capacity_mm + params.awc_mm_per_clay_pct * dclay
    soil_water_capacity = float(np.clip(soil_water_capacity, 60.0, 320.0))

    drainage_coeff = params.base_drainage_coeff * np.exp(-params.drainage_clay_sensitivity * dclay)
    drainage_coeff = float(np.clip(drainage_coeff, 0.015, 0.25))

    runoff_fraction_max = params.base_runoff_fraction_max * np.exp(params.runoff_clay_sensitivity * dclay)
    runoff_fraction_max = float(np.clip(runoff_fraction_max, 0.05, 0.90))

    stress_exponent = 1.4 + params.stress_clay_sensitivity * dclay
    stress_exponent = float(np.clip(stress_exponent, 1.0, 2.2))

    return soil_water_capacity, drainage_coeff, runoff_fraction_max, stress_exponent


def simulate_soil_water_balance(
    data: pd.DataFrame,
    elevation_m: float,
    clay_pct: float,
    params: SoilWaterParams = SoilWaterParams(),
    date_col: Optional[str] = None,
    temp_col: str = "temp_c",
    precip_col: str = "precip_mm",
    biomass_col: str = "biomass_gC_m2",
):
    df = data.copy()

    T = df[temp_col].to_numpy(dtype=float)
    P = np.maximum(df[precip_col].to_numpy(dtype=float), 0.0)
    B = np.maximum(df[biomass_col].to_numpy(dtype=float), 0.0)

    atm_pressure_kpa = pressure_from_elevation_kpa(elevation_m)

    base_pet = temperature_pet_weekly(
        T,
        elevation_m,
        coeff_mm_per_degday=params.pet_coeff_mm_per_degday,
        temp_base_c=params.temp_base_c,
        reference_elevation_m=params.reference_elevation_m,
        elev_pet_sensitivity_per_km=params.elev_pet_sensitivity_per_km
    )

    kc = biomass_to_kc(B, params.kc_min, params.kc_max, params.biomass_half_sat_gC_m2)
    interception_capacity = biomass_interception_capacity_mm(
        B, params.interception_max_mm, params.interception_biomass_half_sat_gC_m2
    )
    uptake_factor = biomass_uptake_factor(
        B, params.uptake_max_factor, params.uptake_biomass_half_sat_gC_m2
    )

    Smax_const, drain_const, runoff_const, stress_exp_const = clay_modified_properties(clay_pct, params)

    n = len(df)
    S = np.zeros(n)
    runoff = np.zeros(n)
    drainage = np.zeros(n)
    et_actual = np.zeros(n)
    et_potential = np.zeros(n)
    stress = np.zeros(n)
    intercepted_mm = np.zeros(n)
    throughfall_mm = np.zeros(n)
    freeze_weeks_below_zero = np.zeros(n, dtype=int)
    frozen_flag = np.zeros(n, dtype=bool)

    S_prev = params.initial_soil_moisture_mm
    consecutive_freezing_weeks = 0
    is_frozen = False

    for i in range(n):
        if T[i] < 0.0:
            consecutive_freezing_weeks += 1
        else:
            consecutive_freezing_weeks = 0

        if consecutive_freezing_weeks > params.freeze_trigger_weeks:
            is_frozen = True

        if is_frozen and T[i] >= params.thaw_trigger_temp_c:
            is_frozen = False
            consecutive_freezing_weeks = 0

        freeze_weeks_below_zero[i] = consecutive_freezing_weeks
        frozen_flag[i] = is_frozen

        Smax_i = Smax_const
        stress_i_exp = stress_exp_const

        if is_frozen:
            drain_i = drain_const * params.frozen_drainage_multiplier
            runoff_i_max = min(0.98, runoff_const + (1.0 - params.frozen_infiltration_fraction) * 0.5)
            pet_i = base_pet[i] * params.frozen_pet_multiplier
            uptake_i = 1.0 + (uptake_factor[i] - 1.0) * params.frozen_uptake_multiplier
        else:
            drain_i = drain_const
            runoff_i_max = runoff_const
            pet_i = base_pet[i]
            uptake_i = uptake_factor[i]

        intercepted_mm[i] = min(P[i], interception_capacity[i])
        throughfall_mm[i] = P[i] - intercepted_mm[i]

        S_prev = np.clip(S_prev, 0.0, Smax_i)
        rel_sat = S_prev / Smax_i if Smax_i > 0 else 0.0

        runoff_frac = runoff_i_max * (rel_sat ** params.runoff_exponent)
        runoff_input = throughfall_mm[i]

        if is_frozen:
            infiltration = params.frozen_infiltration_fraction * runoff_input
            runoff[i] = runoff_input - infiltration
        else:
            runoff[i] = runoff_frac * runoff_input
            infiltration = runoff_input - runoff[i]

        available_before_et = min(Smax_i, S_prev + infiltration)
        stress[i] = np.clip((available_before_et / Smax_i) ** stress_i_exp, 0.0, 1.0)

        et_potential[i] = pet_i * kc[i] * uptake_i
        et_actual[i] = min(et_potential[i] * stress[i], available_before_et)

        after_et = available_before_et - et_actual[i]
        fc_store = params.field_capacity_fraction * Smax_i

        if after_et > fc_store:
            drainage[i] = drain_i * (after_et - fc_store)
        else:
            drainage[i] = 0.0

        S_now = np.clip(after_et - drainage[i], 0.0, Smax_i)
        S[i] = S_now
        S_prev = S_now

    df["site_elevation_m"] = float(elevation_m)
    df["site_clay_pct"] = float(clay_pct)
    df["atm_pressure_kpa"] = atm_pressure_kpa
    df["base_pet_mm"] = base_pet
    df["kc"] = kc
    df["biomass_uptake_factor"] = uptake_factor
    df["interception_capacity_mm"] = interception_capacity
    df["intercepted_mm"] = intercepted_mm
    df["throughfall_mm"] = throughfall_mm
    df["freeze_weeks_below_zero"] = freeze_weeks_below_zero
    df["frozen_soil"] = frozen_flag
    df["et_potential_mm"] = et_potential
    df["water_stress"] = stress
    df["et_actual_mm"] = et_actual
    df["runoff_mm"] = runoff
    df["drainage_mm"] = drainage
    df["soil_water_capacity_mm"] = Smax_const
    df["drainage_coeff"] = drain_const
    df["runoff_fraction_max"] = runoff_const
    df["stress_exponent"] = stress_exp_const
    df["soil_moisture_mm"] = S
    df["relative_soil_moisture"] = S / Smax_const

    if date_col is not None and date_col in df.columns:
        cols = [date_col] + [c for c in df.columns if c != date_col]
        df = df[cols]

    return df
