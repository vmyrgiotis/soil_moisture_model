# Soil Moisture Model

## Overview
This model simulates weekly soil moisture dynamics, accounting for precipitation, temperature, biomass, soil properties (clay content), and site elevation. It is designed for ecohydrological and agricultural applications, providing estimates of soil water balance components and plant water stress.

## Model Parameters
The model is parameterized via the `SoilWaterParams` dataclass. Key parameters and their units:

| Parameter                        | Description                                 | Units         | Typical Value |
|-----------------------------------|---------------------------------------------|---------------|--------------|
| initial_soil_moisture_mm         | Initial soil moisture                       | mm            | 50           |
| base_soil_water_capacity_mm      | Base soil water holding capacity            | mm            | 160          |
| base_drainage_coeff              | Drainage coefficient                        | fraction      | 0.10         |
| base_runoff_fraction_max         | Max runoff fraction                         | fraction      | 0.45         |
| runoff_exponent                  | Runoff exponent                             | -             | 2.0          |
| field_capacity_fraction          | Field capacity as fraction of capacity      | fraction      | 0.70         |
| reference_clay_pct               | Reference clay percentage                   | %             | 25           |
| awc_mm_per_clay_pct              | AWC per % clay                              | mm/%          | 1.2          |
| drainage_clay_sensitivity        | Drainage sensitivity to clay                | 1/%           | 0.015        |
| runoff_clay_sensitivity          | Runoff sensitivity to clay                  | 1/%           | 0.010        |
| stress_clay_sensitivity          | Stress exponent sensitivity to clay         | 1/%           | 0.003        |
| temp_base_c                      | Base temperature for PET                    | °C            | 0.0          |
| pet_coeff_mm_per_degday          | PET coefficient                             | mm/°C·day     | 0.55         |
| reference_elevation_m            | Reference elevation                         | m             | 0.0          |
| elev_pet_sensitivity_per_km      | PET sensitivity to elevation                | /km           | 0.08         |
| kc_min, kc_max                   | Min/max crop coefficient                    | -             | 0.30/1.20    |
| biomass_half_sat_gC_m2           | Half-saturation for biomass effect          | gC/m²         | 150          |
| interception_max_mm              | Max canopy interception                     | mm            | 3.0          |
| interception_biomass_half_sat_gC_m2 | Half-sat for interception effect         | gC/m²         | 120          |
| uptake_max_factor                | Max root uptake factor                      | -             | 1.35         |
| uptake_biomass_half_sat_gC_m2    | Half-sat for uptake effect                  | gC/m²         | 140          |
| freeze_trigger_weeks             | Weeks below 0°C to trigger freeze           | weeks         | 4            |
| thaw_trigger_temp_c              | Thaw temperature threshold                  | °C            | 0.0          |
| frozen_infiltration_fraction     | Infiltration fraction when frozen           | fraction      | 0.15         |
| frozen_drainage_multiplier       | Drainage multiplier when frozen             | fraction      | 0.20         |
| frozen_uptake_multiplier         | Uptake multiplier when frozen               | fraction      | 0.15         |
| frozen_pet_multiplier            | PET multiplier when frozen                  | fraction      | 0.25         |

## Model Outputs
The main outputs (per week) are columns in the returned DataFrame:

| Output Column            | Description                                 | Units   |
|-------------------------|---------------------------------------------|---------|
| soil_moisture_mm        | Soil moisture content                       | mm      |
| relative_soil_moisture  | Soil moisture as fraction of capacity       | -       |
| et_actual_mm            | Actual evapotranspiration                   | mm      |
| et_potential_mm         | Potential evapotranspiration                | mm      |
| runoff_mm               | Runoff                                      | mm      |
| drainage_mm             | Drainage                                    | mm      |
| intercepted_mm          | Precip intercepted by canopy                | mm      |
| throughfall_mm          | Precipitation reaching soil                  | mm      |
| water_stress            | Plant water stress (0-1)                    | -       |
| frozen_soil             | Soil frozen flag                            | bool    |

## Units
- All water fluxes and storages are in millimeters (mm).
- Biomass is in grams of carbon per square meter (gC/m²).
- Temperature in degrees Celsius (°C).
- Clay content in percent (%).
- Elevation in meters (m).

## Repository
https://github.com/vmyrgiotis/soil_moisture_model

