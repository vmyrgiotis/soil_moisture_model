"""
Data production pipeline for EO and meteorological forcing datasets.

This module can be imported and executed as a script. It sources, processes,
and saves a set of Earth observation and meteorological variables for a
single flux-tower site, defined by a point coordinate and a date range.
All outputs are aggregated to a weekly time step and written to NetCDF.

Data Sources & Variables
------------------------
1. Sentinel-1 GRD — Copernicus Data Space (OpenEO federation)
   * SAR sigma-0 backscatter (VV, VH) in dB
   * Output: <SITE>_openeo_s1bs.nc

2. Sentinel-2 L2A — Copernicus Data Space (OpenEO federation)
   * Surface reflectance bands B01-B12, scene classification (SCL),
     and viewing/solar geometry angles (VZA, VAA, SZA, SAA)
   * GPR-derived LAI (m2 m-2) and FAPAR (-) with uncertainty
     estimated using pre-trained Gaussian Process Regression models
     from the grounded-eo repository
   * Output: <SITE>_openeo_s2_gpr_bands.nc,
             <SITE>_fapar_lai_gpr_openeo.nc

3. MODIS — NASA AppEEARS API
   * Gross Primary Production — Gpp_500m (MOD17A2HGF.061, g C m-2 8-day)
   * Leaf Area Index — Lai_500m (MCD15A2H.061, m2 m-2)
   * FPAR — Fpar_500m (MCD15A2H.061, -)
   * Land cover type — LC_Type5 (MCD12Q1.061)
   * Output directory: eo/<SITE>_LC_LAI_FPAR_GPP/

4. ERA5 reanalysis — local files from metData/Europe
   * 2 m air temperature (t2m, K)
   * 2 m dewpoint temperature (d2m, K)
   * Surface pressure (sp, Pa)
   * Surface solar radiation downwards (ssrd, J m-2)
   * Total precipitation (tp, m)
   * Total cloud cover (tcc, -)
   * Volumetric soil water layer 1 (swvl1, m3 m-3)
    * Input path pattern: metData/Europe/{year}_{variable}

5. NOAA GML Mauna Loa daily CO2 — https://gml.noaa.gov
   * Atmospheric CO2 concentration (ppm), gap-filled with linear
     interpolation and aggregated to the chosen time step

Derived / Processed Variables (saved to final NetCDF)
------------------------------------------------------
All variables are aggregated to a weekly (7-day) time step.

Variable              Units           Description
--------              -----           -----------
lai_obs               m2 m-2          GPR LAI from Sentinel-2
fapar                 -               GPR FAPAR from Sentinel-2
temp_degC             degC            ERA5 2 m air temperature
vpd_Pa                Pa              Vapour pressure deficit (ERA5 t2m + d2m)
ppfd_umol_m2_s1       umol m-2 s-1    PPFD derived from SSRD x FAPAR
patm_Pa               Pa              ERA5 surface pressure
precip_mm             mm              ERA5 total precipitation
co2_ppm               ppm             NOAA Mauna Loa atmospheric CO2
sf                    -               Stomatal factor (placeholder)

Final output: eo/<SITE>_weekly_final_variables.nc

Script arguments
----------------
--wrk_dir : str
    Absolute working directory that contains output folders (e.g. "eo", "met").
--site : str
    Site identifier used in output naming.
--lat : float
    Site latitude in decimal degrees.
--lon : float
    Site longitude in decimal degrees.
--start_date : str
    Simulation start date in YYYY-MM-DD format.
--end_date : str
    Simulation end date in YYYY-MM-DD format.

Example
-------
python data_production.py --wrk_dir /mnt/shared/pyrealm --site LAIG --lat 53.12 --lon -2.34 --start_date 2015-01-01 --end_date 2025-06-30
"""

# =============================================================================
# IMPORTS
# Standard library
# =============================================================================
import os
import sys
import json
import time
import warnings
import pathlib
from glob import glob
from pprint import pprint

# Suppress all warnings globally
warnings.filterwarnings('ignore')

# Numerical / data
import numpy as np
import pandas as pd
import geopandas as gpd
import xarray as xr
import rasterio
import joblib

# HTTP clients
import requests as r
from httpx import Client

# Remote-sensing / EO data access
import openeo       # Sentinel-2/1 via OpenEO federation

# Visualisation
import seaborn as sns
import matplotlib.pyplot as plt
import argparse
import subprocess
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import rasterio
from rasterio.transform import Affine
import rioxarray

# =============================================================================
# COMMAND-LINE ARGUMENTS
# Run as:  python data_production.py --wrk_dir /mnt/shared/pyrealm --site LAIG --lat 53 --lon -2
# All arguments are required; no default values are assumed.
# =============================================================================
parser = argparse.ArgumentParser(
    description='Run the data production pipeline.',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
parser.add_argument('--wrk_dir', type=str, required=True,
                    help='Absolute path to the working directory (e.g. /mnt/shared/pyrealm/)')
parser.add_argument('--site', type=str, required=True,
                    help='Site name identifier (e.g. LAIG, EastBush, BB_CA)')
parser.add_argument('--lat', type=float, required=True,
                    help='Site latitude in decimal degrees (e.g. 53.12)')
parser.add_argument('--lon', type=float, required=True,
                    help='Site longitude in decimal degrees (e.g. -2.34)')
parser.add_argument('--start_date', type=str, required=True,
                    help='Start date in YYYY-MM-DD format (e.g. 2015-01-01)')
parser.add_argument('--end_date', type=str, required=True,
                    help='End date in YYYY-MM-DD format (e.g. 2025-06-30)')

WRK_DIR = None
SITE_NAME = None
LAT = None
LON = None
START_DATE = None
END_DATE = None
start_date_iso = None
end_date_iso = None
lat = None
lon = None
W = None
S = None
E = None
N = None
square_geojson = None
square_feature_collection = None


def initialize_runtime_context(args):
    """Populate runtime globals and build the 250 m square AOI.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed command-line arguments with attributes:
        `wrk_dir`, `site`, `lat`, `lon`, `start_date`, `end_date`.

    Notes
    -----
    This function updates module-level globals used by the runtime pipeline,
    including bounding-box coordinates and GeoJSON geometries.
    """
    global WRK_DIR, SITE_NAME, LAT, LON, START_DATE, END_DATE
    global start_date_iso, end_date_iso, lat, lon, W, S, E, N
    global square_geojson, square_feature_collection

    WRK_DIR = args.wrk_dir
    SITE_NAME = args.site
    LAT = args.lat
    LON = args.lon
    START_DATE = args.start_date
    END_DATE = args.end_date

    # Parse and validate expected YYYY-MM-DD inputs.
    start_date_obj = pd.to_datetime(START_DATE, format='%Y-%m-%d')
    end_date_obj = pd.to_datetime(END_DATE, format='%Y-%m-%d')
    if end_date_obj < start_date_obj:
        raise ValueError('end_date must be after or equal to start_date')

    start_date_iso = start_date_obj.strftime('%Y-%m-%d')
    end_date_iso = end_date_obj.strftime('%Y-%m-%d')

    lat = LAT
    lon = LON
    meters_per_deg_lat = 111320
    meters_per_deg_lon = 40075000 * np.cos(np.deg2rad(lat)) / 360
    half_side_m = 100  # half of 100 m
    delta_lat = half_side_m / meters_per_deg_lat
    delta_lon = half_side_m / meters_per_deg_lon

    square_coords = [
        [lon - delta_lon, lat + delta_lat],  # top-left
        [lon + delta_lon, lat + delta_lat],  # top-right
        [lon + delta_lon, lat - delta_lat],  # bottom-right
        [lon - delta_lon, lat - delta_lat],  # bottom-left
        [lon - delta_lon, lat + delta_lat],  # close polygon
    ]
    square_geojson = {
        "type": "Feature",
        "geometry": {
            "type": "Polygon",
            "coordinates": [square_coords],
        },
        "properties": {},
    }
    square_feature_collection = {
        "type": "FeatureCollection",
        "features": [square_geojson],
    }
    
    # Reusable bounds (west, south, east, north) for services that require bbox.
    W = lon - delta_lon
    S = lat - delta_lat
    E = lon + delta_lon
    N = lat + delta_lat

    # # Clone grounded-eo repository if not already present
    grounded_eo_path = os.path.join(WRK_DIR, 'grounded-eo')
    if not os.path.exists(grounded_eo_path):
        subprocess.run(['git', 'clone', 'https://github.com/luke-a-brown/grounded-eo.git', grounded_eo_path], check=True)

# =============================================================================
# HELPER FUNCTIONS
# Utility functions used throughout the pipeline. Defined here so they are
# available before any data-download or model code runs.
# =============================================================================
# --- Elevation fetching utility --- 

def get_elevation(lat, lon):
    """
    Query Open-Elevation API to get elevation (in meters) for a given latitude and longitude.
    Returns elevation in meters, or None if not found.
    """
    url = "https://api.open-elevation.com/api/v1/lookup"
    params = {"locations": f"{lat},{lon}"}
    try:
        response = r.get(url, params=params)
        response.raise_for_status()
        results = response.json().get("results", [])
        if results:
            return results[0]["elevation"]
    except Exception as e:
        print(f"Error fetching elevation: {e}")
    return None


# --- Daylength utility ---

def daylength(dayOfYear, lat):
    """Computes the length of the day (the time between sunrise and
    sunset) given the day of the year and latitude of the location.
    Accepts dayOfYear as a scalar or numpy array.
    Uses the Brock model for the computations.

    Parameters
    ----------
    dayOfYear : int or np.ndarray
        The day of the year. 1 corresponds to 1st of January
        and 365 to 31st December (on a non-leap year).
    lat : float
        Latitude of the location in degrees. Positive values
        for north and negative for south.

    Returns
    -------
    d : float or np.ndarray
        Daylength in hours.
    """
    dayOfYear = np.asarray(dayOfYear)
    latInRad = np.deg2rad(lat)
    declinationOfEarth = 23.45 * np.sin(np.deg2rad(360.0 * (283.0 + dayOfYear) / 365.0))
    tan_lat = np.tan(latInRad)
    tan_dec = np.tan(np.deg2rad(declinationOfEarth))
    x = -tan_lat * tan_dec

    # Use numpy for vectorized computation
    d = np.where(
        x <= -1.0, 24.0,
        np.where(
            x >= 1.0, 0.0,
            2.0 * np.rad2deg(np.arccos(x)) / 15.0
        )
    )
    return d

# --- Modern % C utility ---

def modernTS(firstYear,lastYear): # Define the start and end years and months

    """ 
    MODERN %  RADIOCARBON 
    Reproduces the curve of % modern C as in Fig 5 in 
    https://www.rothamsted.ac.uk/sites/default/files/Documents/RothC_description.pdf
    """
    
    start_year, start_month = 1860, 1
    end_year, end_month = 2022, 12
    # Create a list of tuples representing year-month pairs
    year_month_pairs = [(year, month) for year in range(start_year, end_year + 1) 
                          for month in range(1, 13) 
                          if not (year == end_year and month > end_month)]
    # Extract year and month into separate lists
    years, months = zip(*year_month_pairs)
    # Create a list for the "modern" values
    modern_values = []
    # Set initial "modern" value and year thresholds
    modern_value = 100
    threshold_year_1955 = 1955
    threshold_year_1962 = 1962
    threshold_year_2000 = 2000
    # Populate the "modern" values based on year thresholds
    for year in years:
      if year <= threshold_year_1955:
          modern_values.append(modern_value)
      elif year <= threshold_year_1962:
          modern_values.append(200)
      elif year <= threshold_year_2000:
          # Linear decline from 200 to 110 over 38 years
          modern_values.append(200 - (year - threshold_year_1962) * (90 / 38))
      else:
          modern_values.append(110)
    # Create the DataFrame
    modernDF = pd.DataFrame({'year': years, 'month': months, 'modern': modern_values})
    return modernDF[modernDF.year.isin(np.arange(firstYear,lastYear+1))]

# --- Soil properties fetching utility ---

def fetch_soil_properties(gdf, depths=None, properties=None):
    """
    Fetch soil properties from the openepi.io API for the centroid of a GeoDataFrame.

    Parameters:
        gdf: GeoDataFrame with geometry column.
        depths: list of depth strings (default: ["0-5cm", "0-30cm", "5-15cm", "15-30cm", "30-60cm", "60-100cm", "100-200cm"])
        properties: list of property strings (default: ["ocs", "clay"])

    Returns:
        soil_df: DataFrame with columns ['property', 'depth', 'unit', 'value']
    """
    
    if depths is None:
        depths = ["0-5cm", "0-30cm", "5-15cm", "15-30cm", "30-60cm", "60-100cm", "100-200cm"]
    if properties is None:
        properties = ["ocs", "clay"]

    with Client() as client:
        response = client.get(
            url="https://api.openepi.io/soil/property",
            params={
                "lat": float(gdf.geometry.centroid.y.mean()),
                "lon": float(gdf.geometry.centroid.x.mean()),
                "depths": depths,
                "properties": properties,
                "values": "mean",
            },
        )
        json_data = response.json()

        records = []
        for prop in json_data["properties"]["layers"]:
            name = prop["name"]
            unit = prop["unit_measure"]["mapped_units"]
            for d in prop["depths"]:
                label = d["label"]
                value = d["values"]["mean"]
                records.append({
                    "property": name,
                    "depth": label,
                    "unit": unit,
                    "value": value
                })
        soil_df = pd.DataFrame(records)
    return soil_df



# Calculate VPD (Vapor Pressure Deficit) in Pa using t2m and d2m
# # t2m and d2m are in Kelvin

def calc_vpd(t2m, d2m):
    """Compute vapor pressure deficit (VPD) from air and dew-point temperature.

    Parameters
    ----------
    t2m : array-like
        Air temperature in Kelvin.
    d2m : array-like
        Dew-point temperature in Kelvin.

    Returns
    -------
    array-like
        Vapor pressure deficit in Pa.
    """

    t2m = t2m - 273.15
    d2m = d2m - 273.15

    ## relative humidity (http:/andrew.rsmas.miami.edu/bmcnoldy/Humidity.html)
    rh = 100 * (np.exp((17.625 * d2m) / (243.04 + d2m)) / np.exp((17.625 * t2m) / (243.04 + t2m)))
    ## vapor pressure deficit (http:/cronklab.wikidot.com/calculation-of-vapour-pressure-deficit)
    vpd = (1 - (rh / 100)) * (610.7 * 10 ** (7.5 * t2m / (237.3 + t2m)))

    return vpd

# --- GPR LAI/FAPAR retrieval utility ---

# https://github.com/vmyrgiotis/grounded-eo.git
# Calcuate FPAR and LAI using pre-trained GPR models 
# and Sentinel-2 bands + angles + scene classification

def compute_gpr_metric(
    b1, b2, b3, b4, b5, b6, b7, b8, b8a, b9, b11, b12,
    scl, vza, vaa, sza, saa,
    variable='LAI',
    scale_factor=10000,
    model_dir='/mnt/shared/pyrealm2/grounded-eo/models'
):
    """Estimate LAI or FAPAR from Sentinel-2 bands using pre-trained GPR models.

    Parameters
    ----------
    b1..b12 : np.ndarray
        Sentinel-2 reflectance bands.
    scl : np.ndarray
        Scene classification layer; only vegetation/soil pixels are predicted.
    vza, vaa, sza, saa : np.ndarray
        Viewing and solar angles in degrees.
    variable : str, default 'LAI'
        Target to estimate: 'LAI' or 'FAPAR'.
    scale_factor : int, default 10000
        Reflectance scaling factor applied to input bands.
    model_dir : str, default 'grounded-eo/models/'
        Directory containing `lai.pkl` and `fapar.pkl` models.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Predicted mean and standard deviation arrays with original spatial shape.
    """
    # Scale reflectance bands
    b1 = b1 / scale_factor
    b2 = b2 / scale_factor
    b3 = b3 / scale_factor
    b4 = b4 / scale_factor
    b5 = b5 / scale_factor
    b6 = b6 / scale_factor
    b7 = b7 / scale_factor
    b8 = b8 / scale_factor
    b8a = b8a / scale_factor
    b9 = b9 / scale_factor
    b11 = b11 / scale_factor
    b12 = b12 / scale_factor

    # Load GPR model
    if variable.lower() == 'lai':
        gpr = joblib.load(f'{model_dir}/lai.pkl')
        upper_lim = 10
    elif variable.lower() == 'fapar':
        gpr = joblib.load(f'{model_dir}/fapar.pkl')
        upper_lim = 1
    else:
        raise ValueError("variable must be 'LAI' or 'FAPAR'")

    # Calculate cosine of angles
    cos_vza = np.cos(np.radians(vza))
    cos_sza = np.cos(np.radians(sza))
    cos_raa = np.cos(np.radians(vaa - saa))

    # Get shape
    shape = b1.shape

    # Construct GPR input array
    inputs = np.array([
        b1.flatten(), b2.flatten(), b3.flatten(), b4.flatten(),
        b5.flatten(), b6.flatten(), b7.flatten(), b8.flatten(),
        b8a.flatten(), b9.flatten(), b11.flatten(), b12.flatten(),
        cos_sza.flatten(), cos_vza.flatten(), cos_raa.flatten()
    ]).T

    # Mask non-vegetated/non-soil pixels using scene classification
    mask = np.isin(scl.flatten(), [4, 5])

    # Drop NaNs before calling model
    valid_mask = mask.copy()
    # Find rows with any NaN in the input features
    nan_mask = np.any(np.isnan(inputs), axis=1)
    valid_mask = valid_mask & (~nan_mask)

    # Prepare output arrays
    gpr_mean = np.full(inputs.shape[0], np.nan, dtype=np.float32)
    gpr_std = np.full(inputs.shape[0], np.nan, dtype=np.float32)

    # Only predict for valid pixels
    if np.any(valid_mask):
        pred_mean, pred_std = gpr.predict(inputs[valid_mask], return_std=True)
        # Restrict to minima/maxima
        pred_mean[pred_mean < 0] = 0
        pred_mean[pred_mean > upper_lim] = upper_lim
        gpr_mean[valid_mask] = pred_mean
        gpr_std[valid_mask] = pred_std

    # Reshape to original shape
    gpr_mean = gpr_mean.reshape(shape)
    gpr_std = gpr_std.reshape(shape)

    return gpr_mean, gpr_std


def run_pipeline(wrk_dir: str, site: str, lat: float, lon: float, start_date: str, end_date: str, site_type: str):
    """
    Run the full end-to-end data production workflow.

    Steps:
    1. Initialize runtime context and AOI.
    2. Download Sentinel-1 backscatter data (SAR sigma0) via OpenEO.
    3. Download Sentinel-2 bands and compute GPR LAI/FAPAR.
    4. Download MODIS LAI/GPP/land cover data via AppEEARS.
    5. Discover and merge local ERA5 meteorological data.
    6. Download and process NOAA Mauna Loa CO2 time series.
    7. Derive meteorological variables (VPD, PPFD, etc.).
    8. Save final processed variables to NetCDF.

    Parameters
    ----------
    wrk_dir : str
        Absolute working directory used for input/output paths.
    site : str
        Site identifier used in file naming.
    lat : float
        Site latitude in decimal degrees.
    lon : float
        Site longitude in decimal degrees.
    start_date : str
        Simulation start date in YYYY-MM-DD format.
    end_date : str
        Simulation end date in YYYY-MM-DD format.
    site_type : str
        Site type classification (e.g., 'croplands', 'grasslands', 'forest', etc.)
    """
    args = argparse.Namespace(
        wrk_dir=wrk_dir,
        site=site,
        lat=lat,
        lon=lon,
        start_date=start_date,
        end_date=end_date,
    )

    def log_step_success(step_number: int, message: str):
        print(f"[SUCCESS] Step {step_number}: {message}")


    # Step 1: Initialize runtime context and AOI
    initialize_runtime_context(args)
    log_step_success(0, f"Runtime context initialized for site={SITE_NAME} with AOI bounds W={W}, S={S}, E={E}, N={N}.")

    # Use weekly aggregation consistently throughout the pipeline.
    agg = '7D'
    # Set aggregation label early so it's always defined
    agg_norm = str(agg).strip().lower()
    if agg_norm in {'1', '1d'}:
        agg_label = 'daily'
    elif agg_norm in {'7', '7d'}:
        agg_label = 'weekly'
    elif agg_norm in {'30', '30d'}:
        agg_label = 'monthly'
    else:
        agg_label = f'agg_{agg_norm}'

    # =============================================================================
    # SECTION 1 — SENTINEL-1 BACKSCATTER DOWNLOAD (OpenEO)
    # Downloads SAR sigma0 backscatter (VV, VH) for the site AOI via the
    # Copernicus Data Space OpenEO federation. Skipped if the file already exists.
    # Output: <SITE_NAME>_openeo_s1bs.nc
    # =============================================================================
    # Step 2: Download Sentinel-1 backscatter data
    band_file = os.path.join(WRK_DIR, 'eo', f'{SITE_NAME}_openeo_s1bs.nc')
    try:
        if not os.path.exists(band_file):
            connection = openeo.connect("openeofed.dataspace.copernicus.eu")
            connection.authenticate_oidc()
            datacube = connection.load_collection(
                "SENTINEL1_GRD",
                spatial_extent={"west": W, "south": S, "east": E, "north": N, "crs": "EPSG:4326"},
                temporal_extent=["2017-01-01", end_date_iso],
                bands=["VV", "VH"],
                properties=[openeo.collection_property("polarisation") == "VV&VH"]
            )
            s1_scatter = datacube.sar_backscatter(coefficient="sigma0-ellipsoid", elevation_model="COPERNICUS_30", noise_removal=True)
            s1bs = s1_scatter.apply(lambda x: 10 * x.log(base=10))
            # s1bs.download(band_file, format="NetCDF")
            job = s1bs.execute_batch(out_format="netCDF")
            job.get_result().download_file(band_file)            
        else:
            warnings.warn(f"{band_file} already exists. Skipping download.")
        log_step_success(1, f"Sentinel-1 backscatter data is ready at {band_file}.")
    except Exception as e:
        print(f"[ERROR] Sentinel-1 sourcing failed: {e}")
        log_step_success(1, f"Sentinel-1 step skipped due to error.")

    # SECTION 2 — SENTINEL-2 BAND DOWNLOAD
    # Step 3: Download Sentinel-2 bands and compute GPR LAI/FAPAR
    band_file = os.path.join(WRK_DIR, 'eo', f'{SITE_NAME}_openeo_s2_gpr_bands.nc')
    try:
        if not os.path.exists(band_file):
            connection = openeo.connect("openeofed.dataspace.copernicus.eu")
            connection.authenticate_oidc()
            bands = ["B01", "B02", "B03", "B04", "B05", "B06", 
                     "B07", "B08", "B8A", "B09", "B11", "B12", 
                     "SCL", "VZA", "VAA", "SZA", "SAA"]
            datacube = connection.load_collection(
                "SENTINEL2_L2A",
                spatial_extent={"west": W, "south": S, "east": E, "north": N},
                temporal_extent=["2017-01-01", end_date_iso],
                bands=bands,
            )
            # datacube.download(band_file, format="NetCDF")
            job = datacube.execute_batch(out_format="netCDF")
            job.get_result().download_file(band_file)            
        else:
            warnings.warn(f"{band_file} already exists. Skipping download.")
            
    except Exception as e:
        print(f"[ERROR] Sentinel-2 sourcing failed: {e}")
        log_step_success(2, f"Sentinel-2 step skipped due to error.")
    
    # GPR LAI/FAPAR COMPUTATION 
    out_nc = os.path.join(WRK_DIR, 'eo', f'{SITE_NAME}_fapar_lai_gpr_openeo.nc')
    if not os.path.exists(out_nc):
        data = xr.open_dataset(os.path.join(WRK_DIR, 'eo',  f'{SITE_NAME}_openeo_s2_gpr_bands.nc'))
        b1 = data.B01.values
        b2 = data.B02.values
        b3 = data.B03.values
        b4 = data.B04.values
        b5 = data.B05.values
        b6 = data.B06.values
        b7 = data.B07.values
        b8 = data.B08.values
        b8a = data.B8A.values
        b9 = data.B09.values
        b11 = data.B11.values
        b12 = data.B12.values
        scl = data.SCL.values
        vza = data.VZA.values
        vaa = data.VAA.values
        sza = data.SZA.values
        saa = data.SAA.values
        fapar_gpr, fapar_gpr_std = compute_gpr_metric(
            b1, b2, b3, b4, b5, b6, b7, b8, b8a, b9, b11, b12,
            scl, vza, vaa, sza, saa,
            variable='FAPAR',
            scale_factor=10000,
        )
        lai_gpr, lai_gpr_std = compute_gpr_metric(
            b1, b2, b3, b4, b5, b6, b7, b8, b8a, b9, b11, b12,
            scl, vza, vaa, sza, saa,
            variable='LAI',
            scale_factor=10000,
        )
        times = data['t']
        lats = data['y']
        lons = data['x']
        fapar_lai_gpr_ds = xr.Dataset(
            {
                "fapar_gpr": (("t", "y", "x"), fapar_gpr),
                "fapar_gpr_std": (("t", "y", "x"), fapar_gpr_std),
                "lai_gpr": (("t", "y", "x"), lai_gpr),
                "lai_gpr_std": (("t", "y", "x"), lai_gpr_std),
            },
            coords={
                "t": times,
                "y": lats,
                "x": lons,
            },
            attrs={"description": "GPR-derived FAPAR, LAI and uncertainty from Sentinel-2"}
        )
        fapar_lai_gpr_ds.to_netcdf(out_nc)
        print(f"Saved FAPAR and LAI GPR dataset to {out_nc}")
    else:
        warnings.warn(f"{out_nc} already exists. Skipping computation and save.")
    log_step_success(2, f"Sentinel-2 bands and GPR LAI/FAPAR outputs are ready ({band_file}, {out_nc}).")
    
    # SECTION 3 — MODIS LAI / GPP / LAND COVER DOWNLOAD (AppEEARS)
    # Step 4: Download MODIS LAI/GPP/land cover data
    IN_DIR = os.path.join(WRK_DIR, 'eo')
    API_URL = 'https://appeears.earthdatacloud.nasa.gov/api/'
    USER = "vmyrg"
    PASSWORD = "CU5-HxX-HyT-AfU"
    TASK_NAME = f'{SITE_NAME}_LC_LAI_FPAR_GPP'
    OUTPUT_FORMAT = 'netcdf4'
    try:
        if not os.path.exists(IN_DIR):
            os.makedirs(IN_DIR)
        os.chdir(IN_DIR)
        if os.path.exists(os.path.join(IN_DIR, TASK_NAME)):
            print(f"Data already exists at {os.path.join(IN_DIR, TASK_NAME)}. Skipping download and processing.")
        else:
            token_response = r.post(f'{API_URL}login', auth=(USER, PASSWORD)).json()
            token = token_response['token']
            headers = {'Authorization': f'Bearer {token}'}
            product_response = r.get(f'{API_URL}product').json()
            print(f"AρρEEARS currently supports {len(product_response)} products.")
            products = {p['ProductAndVersion']: p for p in product_response}
            layers = []
            X_product = 'MOD17A2HGF.061'
            X_layers = ['Gpp_500m']
            for layer in X_layers:
                layers.append({'product': X_product, 'layer': layer})
            X_product = 'MCD15A2H.061'
            layers.append({'product': X_product, 'layer': 'Lai_500m'})
            X_product = 'MCD15A2H.061'
            X_layers = ['Fpar_500m']
            for layer in X_layers:
                layers.append({'product': X_product, 'layer': layer})
            X_product = 'MCD12Q1.061'
            X_layers = ['LC_Type5','QC']
            for layer in X_layers:
                layers.append({'product': X_product, 'layer': layer})
            nps_json = square_feature_collection
            projections = r.get(f'{API_URL}spatial/proj').json()
            projs = {p['Name']: p for p in projections}
            projection_name = projs['geographic']['Name']
            task = {
                'task_type': 'area',
                'task_name': TASK_NAME,
                'params': {
                'dates': [{'startDate': pd.to_datetime(start_date_iso).strftime('%m-%d-%Y'), 
                           'endDate': pd.to_datetime(end_date_iso).strftime('%m-%d-%Y')}],
                'layers': layers,
                'output': {
                    'format': {'type': OUTPUT_FORMAT},
                    'projection': projection_name
                },
                'geo': nps_json,
                }
            }
            task_response = r.post(f'{API_URL}task', json=task, headers=headers).json()
            if "task_id" in task_response:
                task_id = task_response["task_id"]
                print("Task submitted:", task_id)
            else:
                print("Task submission failed:", task_response.get("message", task_response))
                raise RuntimeError(f"Task submission failed: {task_response.get('message', task_response)}")
            def wait_for_task_completion(api_url, task_id, headers):
                while True:
                    resp = r.get(f'{api_url}task/{task_id}', headers=headers).json()
                    if 'status' in resp:
                        status = resp['status']
                        print("Task status:", status)
                        if status == 'done':
                            break
                    else:
                        print("Unexpected response:", resp)
                        break
                    time.sleep(20)
            wait_for_task_completion(API_URL, task_id, headers)
            dest_dir = os.path.join(IN_DIR, TASK_NAME)
            os.makedirs(dest_dir, exist_ok=True)
            bundle = r.get(f'{API_URL}bundle/{task_id}', headers=headers).json()
            files = {f['file_id']: f['file_name'] for f in bundle['files']}
            for file_id, file_name in files.items():
                dl = r.get(f'{API_URL}bundle/{task_id}/{file_id}', headers=headers, stream=True)
                filename = file_name.split('/')[1] if file_name.endswith('.tif') else file_name
                filepath = os.path.join(dest_dir, filename)
                with open(filepath, 'wb') as f:
                    for chunk in dl.iter_content(chunk_size=8192):
                        f.write(chunk)
            print(f'Downloaded files from AppEEARS can be found at: {dest_dir}')
        log_step_success(3, f"AppEEARS MODIS data is available in {os.path.join(IN_DIR, TASK_NAME)}.")
    except Exception as e:
        print(f"[ERROR] MODIS sourcing failed: {e}")
        log_step_success(3, f"MODIS step skipped due to error.")
    
    
    # =============================================================================
    # SECTION 4 — ERA5 LOCAL INPUT DISCOVERY (metData/Europe)
    # Step 5: Discover and merge local ERA5 meteorological data
    # Reads pre-downloaded ERA5 files from metData/Europe using the pattern
    # {year}_{variable} and collects matching NetCDF files.
    # =============================================================================
    start_year = pd.to_datetime(START_DATE, format='%Y-%m-%d').year
    end_year = pd.to_datetime(END_DATE, format='%Y-%m-%d').year

    def _collect_era5_files(years, variables):
        files = []
        missing = []
        for y in years:
            for var in variables:
                stem = f"{y}_{var}"
                direct_path = era5_base / stem

                if direct_path.is_file():
                    matches = [str(direct_path)]
                elif direct_path.is_dir():
                    matches = sorted(glob(str(direct_path / '**/*.nc'), recursive=True))
                else:
                    matches = sorted(glob(str(era5_base / f'{stem}*.nc')))

                if matches:
                    files.extend(matches)
                else:
                    missing.append(stem)
        return files, missing

    era5_base_candidates = [
        pathlib.Path(WRK_DIR) / 'metData' / 'Europe',
        pathlib.Path(WRK_DIR).parent / 'metData' / 'Europe',
        pathlib.Path('/mnt/shared/metData/Europe'),
    ]
    era5_base = next((p for p in era5_base_candidates if p.exists()), None)
    if era5_base is None:
        candidate_list = ', '.join(str(p) for p in era5_base_candidates)
        raise FileNotFoundError(
            f"Could not find ERA5 local data directory. Checked: {candidate_list}"
        )

    instant_vars = [
        '2m_temperature',
        '2m_dewpoint_temperature',
        'surface_pressure',
    ]
    accum_vars = [
        'surface_solar_radiation_downwards',
        'total_precipitation',
        'total_cloud_cover',
    ]


    years = list(range(start_year, end_year + 1))
    instant_files, missing_instant = _collect_era5_files(years, instant_vars)
    accum_files, missing_accum = _collect_era5_files(years, accum_vars)

    if missing_instant or missing_accum:
        missing_all = missing_instant + missing_accum
        missing_preview = ', '.join(missing_all[:10])
        suffix = ' ...' if len(missing_all) > 10 else ''
        raise FileNotFoundError(
            f"Missing ERA5 local inputs under {era5_base}: {missing_preview}{suffix}"
        )

    log_step_success(4, f"ERA5 local inputs discovered in {era5_base} for {start_year}-{end_year}.")

    # =============================================================================
    # SECTION 5 — ERA5 LOCAL MERGE
    # Step 5 (continued): Merge local ERA5 NetCDF files
    # Merges local ERA5 NetCDF files into ds_accum and ds_instant and trims to
    # [start_date_iso, end_date_iso].
    # =============================================================================
    ds_accum = xr.open_mfdataset(sorted(set(accum_files)), combine='by_coords')
    ds_instant = xr.open_mfdataset(sorted(set(instant_files)), combine='by_coords')

    if 'time' in ds_accum.coords and 'valid_time' not in ds_accum.coords:
        ds_accum = ds_accum.rename({'time': 'valid_time'})
    if 'time' in ds_instant.coords and 'valid_time' not in ds_instant.coords:
        ds_instant = ds_instant.rename({'time': 'valid_time'})

    ds_accum = ds_accum.sel(valid_time=slice(start_date_iso, end_date_iso),)
    ds_instant = ds_instant.sel(valid_time=slice(start_date_iso, end_date_iso))
    ds_accum = ds_accum.sel(latitude=[lat],longitude=[lon], method="nearest")
    ds_instant = ds_instant.sel(latitude=[lat],longitude=[lon], method="nearest")
    # copy ds_instant to use for sunshine duration calculation later, 
    tcc_df = ds_accum.to_dataframe().reset_index().copy()
    tcc_df = tcc_df.reset_index()    
    ds_accum = ds_accum.to_dataframe().reset_index().resample('D', on='valid_time').sum(numeric_only=True)
    ds_instant = ds_instant.to_dataframe().reset_index().resample('D', on='valid_time').mean(numeric_only=True)
    
    log_step_success(
        5,
        f"ERA5 local datasets merged ({len(set(accum_files))} accum files, {len(set(instant_files))} instant files).",
    )
    
    
    # =============================================================================
    # SECTION 6 — ATMOSPHERIC CO2 TIME SERIES (NOAA Mauna Loa)
    # Step 6: Download and process NOAA Mauna Loa CO2 time series
    # Downloads daily CO2 measurements from NOAA GML, filters to the simulation
    # period, gap-fills with linear interpolation, and resamples to the chosen
    # time step (agg). Result: 1-D array `co2` in ppm.
    # =============================================================================
    co2_url = "https://gml.noaa.gov/webdata/ccgg/trends/co2/co2_daily_mlo.txt"
    co2_df = pd.read_csv(
        co2_url,
        delim_whitespace=True,
        comment='#',
        header=None,
        names=['year', 'month', 'day', 'decimal_date', 'daily_avg', 'interp', 'trend', 'days']
    )
    
    # Filter for valid daily average values
    co2_df = co2_df[co2_df['daily_avg'] > 0]
    # Create a datetime column
    co2_df['date'] = pd.to_datetime(co2_df[['year', 'month', 'day']])
    # Merge year, month, and day to a date and set as index
    co2_df = co2_df[(co2_df.year >= start_year) & (co2_df.year <= end_year)].copy()
    co2_df['date'] = pd.to_datetime(co2_df[['year', 'month', 'day']])
    co2_df.set_index('date', inplace=True)
    # Expand index to all days in 2020
    full_index = pd.date_range(start=start_date_iso, end=end_date_iso, freq='D')
    co2_df = co2_df.reindex(full_index)
    # Gapfill daily_avg using linear interpolation, then back/forward fill for any remaining NaNs
    co2_df['daily_avg'] = co2_df['daily_avg'].interpolate(method='linear').bfill().ffill()
    co2 = np.array(co2_df['daily_avg'].resample(agg).mean())
    
    log_step_success(6, f"CO2 time series prepared with {len(co2)} records.")
    
    
    # =============================================================================
    # SECTION 7 — DERIVED METEOROLOGICAL VARIABLES
    # Step 7: Derive meteorological variables (VPD, PPFD, etc.)
    # Calculates VPD (Pa) from ERA5 2 m temperature and dewpoint, and PPFD
    # (umol photons/m2/s) from downward solar radiation x FPAR. Also extracts
    # temperature (tc, degC) and surface pressure (patm, Pa) arrays resampled
    # to the chosen time step. These arrays are direct model inputs.
    # =============================================================================
    vpd = calc_vpd(ds_instant.t2m, ds_instant.d2m).resample(f'{agg}').mean()
    daily_vpd = calc_vpd(ds_instant.t2m.resample('D').mean(), 
                         ds_instant.d2m.resample('D').mean()).reset_index()
    daily_vpd.rename(columns={"valid_time": "date"}, inplace=True)
    daily_vpd = daily_vpd.rename(columns={0: 'vpd'})

    # --- Build continuous FPAR/LAI using MODIS before 2017 and RF gap-filled GPR after 2017 ---
    target_lat = LAT
    target_lon = LON
    start_year_int = pd.to_datetime(START_DATE, format='%Y-%m-%d').year

    modis_file = os.path.join(WRK_DIR, 'eo', f'{SITE_NAME}_LC_LAI_FPAR_GPP/MCD15A2H.061_500m_aid0001.nc')
    
    def _load_modis_daily_series(var_name: str):
        with xr.open_dataset(modis_file) as eo_data:
            time_str = eo_data.time.dt.strftime('%Y-%m-%d')
            mask = (time_str >= start_date_iso) & (time_str <= '2016-12-31')
            modis_da = eo_data[var_name].sel(time=eo_data.time[mask])

            modis_df = modis_da.sel(lon=target_lon, lat=target_lat, method='nearest').to_dataframe()
            date_range = pd.date_range(start=start_date_iso, end='2016-12-31', freq='D')
            modis_df.index = pd.to_datetime(modis_da.time.dt.strftime('%Y-%m-%d').values)
            modis_df = modis_df[[var_name]].reindex(date_range)
            modis_df[var_name] = modis_df[var_name].interpolate(method='linear').bfill().ffill()

            if agg == '1D':
                return np.array(modis_df[var_name].values)
            
            return np.array(modis_df[var_name].resample(f'{agg}').mean().values)

    fapar_before_2017 = np.array([])
    lai_before_2017 = np.array([])
    if start_year_int < 2017:
        if not os.path.exists(modis_file):
            raise FileNotFoundError(f'Missing required MODIS file for pre-2017 period: {modis_file}')
    fapar_before_2017 = _load_modis_daily_series('Fpar_500m')
    lai_before_2017 = _load_modis_daily_series('Lai_500m')

    fapar_lai_gpr_ds = xr.open_dataset(os.path.join(WRK_DIR, 'eo', f'{SITE_NAME}_fapar_lai_gpr_openeo.nc'))
    s1bs_ds = xr.open_dataset(os.path.join(WRK_DIR, 'eo', f'{SITE_NAME}_openeo_s1bs.nc'))

    coarse_fact = 5
    fapar_lai_gpr_ds_coarse = fapar_lai_gpr_ds.coarsen(y=coarse_fact, x=coarse_fact, boundary='trim').mean()
    s1bs_ds_coarse = s1bs_ds.coarsen(y=coarse_fact, x=coarse_fact, boundary='trim').mean()

    gpr_df = fapar_lai_gpr_ds_coarse[['lai_gpr', 'fapar_gpr']].to_dataframe()
    s1_df = s1bs_ds_coarse[['VH', 'VV']].to_dataframe()
    train_df = gpr_df.join(s1_df, how='inner').reset_index()
    train_df['date'] = pd.to_datetime(train_df['t'])
    train_df = pd.merge(train_df, daily_vpd[['date', 'vpd']], on='date', how='left')
    train_df['doy'] = train_df['date'].dt.dayofyear

    features = ['VH', 'VV', 'vpd', 'doy']
    targets = ['lai_gpr', 'fapar_gpr']
    df_rf = train_df[features + targets].dropna()

    if df_rf.empty:
        raise RuntimeError('RF training dataset is empty; cannot build continuous LAI/FPAR time series.')

    X = df_rf[features].values
    y_lai = df_rf['lai_gpr'].values
    y_fapar = df_rf['fapar_gpr'].values

    X_train_lai, X_test_lai, y_train_lai, y_test_lai = train_test_split(X, y_lai, test_size=0.3, random_state=0)
    rf_lai = RandomForestRegressor(n_estimators=100, random_state=0)
    rf_lai.fit(X_train_lai, y_train_lai)
    lai_pred = rf_lai.predict(X_test_lai)
    lai_rf_r2 = float(r2_score(y_test_lai, lai_pred))
    lai_rf_rmse = float(np.sqrt(mean_squared_error(y_test_lai, lai_pred)))
    print('LAI RF R2:', round(lai_rf_r2, 3))
    print('LAI RF RMSE:', round(lai_rf_rmse, 3))

    X_train_fapar, X_test_fapar, y_train_fapar, y_test_fapar = train_test_split(X, y_fapar, test_size=0.3, random_state=0)
    rf_fapar = RandomForestRegressor(n_estimators=100, random_state=0)
    rf_fapar.fit(X_train_fapar, y_train_fapar)
    fapar_pred = rf_fapar.predict(X_test_fapar)
    fapar_rf_r2 = float(r2_score(y_test_fapar, fapar_pred))
    fapar_rf_rmse = float(np.sqrt(mean_squared_error(y_test_fapar, fapar_pred)))
    print('FAPAR RF R2:', round(fapar_rf_r2, 3))
    print('FAPAR RF RMSE:', round(fapar_rf_rmse, 3))

    def _extract_3x3_series(ds: xr.Dataset, var_name: str) -> pd.DataFrame:
        lat_idx = np.abs(ds.y.values - target_lat).argmin()
        lon_idx = np.abs(ds.x.values - target_lon).argmin()

        lat_indices = np.arange(max(0, lat_idx - 1), min(lat_idx + 2, len(ds.y)))
        lon_indices = np.arange(max(0, lon_idx - 1), min(lon_idx + 2, len(ds.x)))

        out = ds[var_name][:, lat_indices, lon_indices].mean(dim=['y', 'x']).to_dataframe().reset_index()
        out['t'] = pd.to_datetime(out['t'])
        return out[['t', var_name]]

    s1_mean_df = s1bs_ds[['VV', 'VH']].mean(dim=['y', 'x']).to_dataframe().reset_index()
    s1_mean_df['t'] = pd.to_datetime(s1_mean_df['t'])
    daily_vpd_t = daily_vpd.rename(columns={'date': 't'})[['t', 'vpd']].copy()
    daily_vpd_t['t'] = pd.to_datetime(daily_vpd_t['t'])

    def _gapfill_and_resample(var_name: str, model: RandomForestRegressor) -> np.ndarray:
        gpr_series = _extract_3x3_series(fapar_lai_gpr_ds, var_name)
        gpr_series = gpr_series[(gpr_series['t'] >= pd.to_datetime('2017-01-01')) & (gpr_series['t'] <= pd.to_datetime(end_date_iso))]

        merged = pd.merge(daily_vpd_t, gpr_series, on='t', how='left')
        merged = pd.merge(merged, s1_mean_df[['t', 'VV', 'VH']], on='t', how='left')
        merged['doy'] = merged['t'].dt.dayofyear

        missing_mask = merged[var_name].isna() & merged[['VV', 'VH', 'vpd']].notna().all(axis=1)
        to_predict = merged.loc[missing_mask, features]
        if not to_predict.empty:
            merged.loc[missing_mask, var_name] = model.predict(to_predict.values)

        merged = merged[(merged['t'] >= pd.to_datetime('2017-01-01')) & (merged['t'] <= pd.to_datetime(end_date_iso))]
        merged = merged.set_index('t')
        merged[var_name] = merged[var_name].interpolate(method='linear').bfill().ffill()
        return np.array(merged[var_name].resample(f'{agg}').mean().values)

    fpar = _gapfill_and_resample('fapar_gpr', rf_fapar)
    lai = _gapfill_and_resample('lai_gpr', rf_lai)
                
    if start_year_int < 2017:
        fpar = np.concatenate((fapar_before_2017, fpar))
        lai = np.concatenate((lai_before_2017, lai))

    fapar_lai_gpr_ds.close()
    s1bs_ds.close()

    # --- Calculate sunshine fraction from daylength and total cloud cover ---
    daily_dates = pd.date_range(start=start_date_iso, end=end_date_iso, freq='D')
    dl = daylength(np.array(daily_dates.day_of_year), float(target_lat))

    # Extract total cloud cover (tcc) at 12 and 16 hours from DataFrame
    tcc_subset = tcc_df[tcc_df['valid_time'].dt.hour.isin([8, 12, 16])]

    # Resample to daily mean
    tcc_daily = tcc_subset.set_index('valid_time')['tcc'].resample('1D').mean().reindex(daily_dates)
    sf_daily = dl * (1 - tcc_daily.values)

    sf_min = np.nanmin(sf_daily)
    sf_max = np.nanmax(sf_daily)
    if np.isclose(sf_max, sf_min):
        sf_daily = np.zeros_like(sf_daily)
    else:
        sf_daily = (sf_daily - sf_min) / (sf_max - sf_min)

    sf_df = pd.DataFrame({'sf': sf_daily}, index=daily_dates)
    if agg == '1D':
        sf = np.array(sf_df['sf'].values)
    else:
        sf = np.array(sf_df['sf'].resample(f'{agg}').mean().values)
    log_step_success(7, f"Continuous {agg_label} LAI/FPAR and sunshine fraction prepared ({len(lai)} LAI points, {len(fpar)} FPAR points, {len(sf)} sf points).")
    
    
    # --- Calculate Photosynthetic Photon Flux Density (PPFD) from FPAR and SSRD --- 
    
    # Convert SSRD from J/m2 to mol photons/m2:
    # 1 J ≈ 4.6 μmol photons (PAR) ---> 1 J/m2 = 4.6e-6 mol photons/m2
    # Calculate PPFD (μmol photons m⁻² s⁻¹)
    # SSRD is accumulated over time intervals, so divide by the time interval in seconds
    
    ssrd = ds_accum.ssrd.resample(f'{agg}').mean() # J/m2 per time step
    interval_seconds = 3600*4  # 1 hour
    # Calculate incident PAR (J/m2 per time step)
    if len(ssrd) != len(fpar):
        fpar = fpar[:-1]
    par = ssrd * fpar
    # Convert PAR to PPFD (μmol/m2/s)
    ppfd = (par / interval_seconds) * 4.6  # μmol/m2/s
    
    # --- Temperature and pressure data  -- 
    tc = ds_instant.t2m.resample(f'{agg}').mean() - 273.15
    patm = ds_instant.sp.resample(f'{agg}').mean()

    # --- Save final processed variables to NetCDF ---
    precip_mm = ds_accum.tp.resample(f'{agg}').sum() * 1000
    # Keep all exported variables aligned to a shared length.
    n = len(vpd)
    time_coord = pd.date_range(start=start_date_iso, end=end_date_iso, freq=agg)[:n]

    output_vars = {
        'lai': np.asarray(lai).reshape(-1)[:n],
        'temperature_celcius': np.asarray(tc).reshape(-1)[:n],
        'vpd_pa': np.asarray(vpd).reshape(-1)[:n],
        'co2_ppm': np.asarray(co2).reshape(-1)[:n],
        'pressure_pa': np.asarray(patm).reshape(-1)[:n],
        'fapar': np.asarray(fpar).reshape(-1)[:n],
        'ppfd_umol_m2_s1': np.asarray(ppfd).reshape(-1)[:n],
        'sunshine_fraction': np.asarray(sf).reshape(-1)[:n],
        'precipitation_mm': np.asarray(precip_mm).reshape(-1)[:n],
    }
    
    # --- MODIS FAPAR and LAI for full period ---
    # Load MODIS daily series for full period
    def _load_modis_full_period(var_name: str):
        with xr.open_dataset(modis_file) as eo_data:
            time_str = eo_data.time.dt.strftime('%Y-%m-%d')
            mask = (time_str >= start_date_iso) & (time_str <= end_date_iso)
            modis_da = eo_data[var_name].sel(time=eo_data.time[mask])
            modis_df = modis_da.sel(lon=target_lon, lat=target_lat, method='nearest').to_dataframe()
            date_range = pd.date_range(start=start_date_iso, end=end_date_iso, freq='D')
            modis_df.index = pd.to_datetime(modis_da.time.dt.strftime('%Y-%m-%d').values)
            modis_df = modis_df[[var_name]].reindex(date_range)
            modis_df[var_name] = modis_df[var_name].interpolate(method='linear').bfill().ffill()
            return np.array(modis_df[var_name].resample(f'{agg}').mean().values)

    modis_fapar_full = _load_modis_full_period('Fpar_500m')
    modis_lai_full = _load_modis_full_period('Lai_500m')

    # Ensure MODIS arrays match n
    modis_fapar_full = np.asarray(modis_fapar_full).reshape(-1)[:n]
    modis_lai_full = np.asarray(modis_lai_full).reshape(-1)[:n]

    output_vars['modis_fapar'] = modis_fapar_full
    output_vars['modis_lai'] = modis_lai_full

    agg_norm = str(agg).strip().lower()
    if agg_norm in {'1', '1d'}:
        agg_label = 'daily'
    elif agg_norm in {'7', '7d'}:
        agg_label = 'weekly'
    elif agg_norm in {'30', '30d'}:
        agg_label = 'monthly'
    else:
        agg_label = f'agg_{agg_norm}'

    # Load soil raster datasets and extract values at site location

    soil_dir = os.path.join(WRK_DIR, 'soilData')
    ocs_path = os.path.join(soil_dir, 'ocs_sum_0_100cm.tif')
    clay_path = os.path.join(soil_dir, 'clay_0_100cm.tif')

    def extract_raster_value(raster_path, lat, lon):
        with rasterio.open(raster_path) as src:
            # Convert lat/lon to raster row/col
            row, col = src.index(lon, lat)
            band_data = src.read(1)
            value = float(band_data[row, col])
            return value

    organic_carbon = extract_raster_value(ocs_path, LAT, LON)
    clay_mean = extract_raster_value(clay_path, LAT, LON)    

    # Prepare y and x as 1D arrays (single point)
    y = np.array([LAT])
    x = np.array([LON])
    # Expand all variables to shape (time, y, x)
    def expand_to_3d(arr, n_time, n_y=1, n_x=1):
        arr = np.asarray(arr).reshape(n_time, 1, 1)
        return arr

    n_time = len(time_coord)
    data_vars = {k: (('time', 'y', 'x'), expand_to_3d(v, n_time)) for k, v in output_vars.items()}

    final_ds = xr.Dataset(
        data_vars,
        coords={'time': time_coord, 'y': y, 'x': x},
        attrs={
            'site_name': SITE_NAME,
            'center_lat': LAT,
            'center_lon': LON,
            'SOC': organic_carbon,
            'clay_fraction': clay_mean,
            'date_start': start_date_iso,
            'date_end': end_date_iso,
            'time_step': agg_label,
            'aggregation': str(agg),
            'aggregation_label': agg_label,
            'site_type': site_type,
            'lai_rf_r2': round(lai_rf_r2, 2),
            'lai_rf_rmse': round(lai_rf_rmse, 2),
            'fapar_rf_r2': round(fapar_rf_r2, 2),
            'fapar_rf_rmse': round(fapar_rf_rmse, 2),
            'Conventions': 'CF-1.8',
            'history': f'Created by data_production.py on {pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")}',
        }
    )
    
    # Add CRS info using rioxarray
    final_ds = final_ds.rio.write_crs("EPSG:4326", inplace=True)

    final_ds['lai'].attrs['units'] = 'm2 m-2'
    final_ds['temperature_celcius'].attrs['units'] = 'degC'
    final_ds['vpd_pa'].attrs['units'] = 'Pa'
    final_ds['co2_ppm'].attrs['units'] = 'ppm'
    final_ds['pressure_pa'].attrs['units'] = 'Pa'
    final_ds['fapar'].attrs['units'] = 'fraction'
    final_ds['ppfd_umol_m2_s1'].attrs['units'] = 'umol m-2 s-1'
    final_ds['sunshine_fraction'].attrs['units'] = '1'
    final_ds['precipitation_mm'].attrs['units'] = 'mm'
    final_ds['modis_fapar'].attrs['units'] = 'fraction'
    final_ds['modis_lai'].attrs['units'] = 'm2 m-2'

    final_out = os.path.join(WRK_DIR, 'inputData', f'{SITE_NAME}_{agg_label}_final_variables.nc')
    os.makedirs(os.path.dirname(final_out), exist_ok=True)
    final_ds.to_netcdf(final_out)
    log_step_success(8, f"Final derived weekly meteorological variables saved to {final_out}.")


    # === SAVE ADDITIONAL DAILY NETCDF WITH ATTRIBUTES ===
    # Compute elevation if not already available
    try:
        elevation = get_elevation(LAT, LON)
    except Exception:
        elevation = None

    # Prepare daily arrays
    daily_dates = pd.date_range(start=start_date_iso, end=end_date_iso, freq='D')
    # Temperature in degC
    daily_temp = ds_instant.t2m - 273.15
    if isinstance(daily_temp, pd.DataFrame):
        daily_temp = daily_temp['t2m']
    temp_vals = daily_temp.values[:len(daily_dates)]
    # Precipitation in mm
    daily_precip = ds_accum.tp * 1000
    if isinstance(daily_precip, pd.DataFrame):
        daily_precip = daily_precip['tp']
    precip_vals = daily_precip.values[:len(daily_dates)]
    # Sunshine fraction
    sf_vals = sf_df['sf'].values[:len(daily_dates)]

    # Prepare y and x as 1D arrays (single point)
    y = np.array([LAT])
    x = np.array([LON])

    # Expand all variables to shape (time, y, x)
    def expand_to_3d(arr, n_time, n_y=1, n_x=1):
        arr = np.asarray(arr).reshape(n_time, 1, 1)
        return arr

    n_time = len(daily_dates)
    daily_ds2 = xr.Dataset({
        'gpp': (('time', 'y', 'x'), expand_to_3d(temp_vals, n_time), {'units': 'this is a copy of lai'}),
        'lai': (('time', 'y', 'x'), expand_to_3d(temp_vals, n_time), {'units': 'this is a copy of temp'}),
        'temperature_celcius': (('time', 'y', 'x'), expand_to_3d(temp_vals, n_time), {'units': 'degC'}),
        'precipitation_mm': (('time', 'y', 'x'), expand_to_3d(precip_vals, n_time), {'units': 'mm'}),
        'sunshine_fraction': (('time', 'y', 'x'), expand_to_3d(sf_vals, n_time), {'units': 'fraction'}),
    }, coords={'time': daily_dates, 'y': y, 'x': x}, attrs={
        'latitude': float(LAT),
        'longitude': float(LON),
        'elevation_m': float(elevation) if elevation is not None else np.nan,
        'description': 'Daily meteorological variables: temperature (degC), precipitation (mm), sunshine fraction',
        'history': f'Created by data_production.py on {pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")}',
    })
    # Add CRS info using rioxarray
    daily_ds2 = daily_ds2.rio.write_crs("EPSG:4326", inplace=True)

    daily_out2 = os.path.join(WRK_DIR, 'inputData', f'{SITE_NAME}_daily_final_variables.nc')
    os.makedirs(os.path.dirname(daily_out2), exist_ok=True)
    daily_ds2.to_netcdf(daily_out2)
    log_step_success(8, f"Final derived daily meteorological variables saved to {daily_out2}.")
    
    # === SAVE ADDITIONAL MONTHLY NETCDF WITH ATTRIBUTES ===
    # Aggregate all weekly variables to monthly (mean for most, sum for precip)
    monthly_time_coord = pd.date_range(start=start_date_iso, end=end_date_iso, freq='MS')
    # Use pandas DataFrame for aggregation
    df = pd.DataFrame({k: np.asarray(v).reshape(-1)[:n] for k, v in output_vars.items()}, index=time_coord)
    # For precipitation, use sum; for others, use mean
    monthly_vars = {}
    for k in output_vars:
        if 'precip' in k:
            monthly_vars[k] = df[k].resample('MS').sum().values
        else:
            monthly_vars[k] = df[k].resample('MS').mean().values
    # MODIS variables
    monthly_vars['modis_fapar'] = df['modis_fapar'].resample('MS').mean().values
    monthly_vars['modis_lai'] = df['modis_lai'].resample('MS').mean().values
    # Prepare y and x as 1D arrays (single point)
    y = np.array([LAT])
    x = np.array([LON])
    n_months = len(monthly_time_coord)
    data_vars_monthly = {k: (('time', 'y', 'x'), expand_to_3d(v, n_months)) for k, v in monthly_vars.items()}
    monthly_ds = xr.Dataset(
        data_vars_monthly,
        coords={'time': monthly_time_coord, 'y': y, 'x': x},
        attrs={
            'site_name': SITE_NAME,
            'center_lat': LAT,
            'center_lon': LON,
            'SOC': organic_carbon,
            'clay_fraction': clay_mean,
            'date_start': start_date_iso,
            'date_end': end_date_iso,
            'time_step': 'monthly',
            'aggregation': '30D',
            'aggregation_label': 'monthly',
            'site_type': site_type,
            'lai_rf_r2': round(lai_rf_r2, 2),
            'lai_rf_rmse': round(lai_rf_rmse, 2),
            'fapar_rf_r2': round(fapar_rf_r2, 2),
            'fapar_rf_rmse': round(fapar_rf_rmse, 2),
            'Conventions': 'CF-1.8',
            'history': f'Created by data_production.py on {pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")}',
        }
    )
    # Add CRS info using rioxarray
    monthly_ds = monthly_ds.rio.write_crs("EPSG:4326", inplace=True)
    # Copy units from weekly dataset
    for var in monthly_ds.data_vars:
        if var in final_ds.data_vars and 'units' in final_ds[var].attrs:
            monthly_ds[var].attrs['units'] = final_ds[var].attrs['units']
    monthly_out = os.path.join(WRK_DIR, 'inputData', f'{SITE_NAME}_monthly_final_variables.nc')
    os.makedirs(os.path.dirname(monthly_out), exist_ok=True)
    monthly_ds.to_netcdf(monthly_out)
    log_step_success(9, f"Final derived monthly meteorological variables saved to {monthly_out}.")
    
    # === STATIC SITE NETCDF ===
    # Prepare static variables for the site grid (single point)
    static_y = np.array([LAT])
    static_x = np.array([LON])

    # Fetch elevation if not already available
    if 'elevation' not in locals() or elevation is None:
        try:
            elevation = get_elevation(LAT, LON)
        except Exception:
            elevation = np.nan

    # Example plant_type assignment (user may want to customize this)
    # Map site_type to plant_type integer codes
    plant_type_mapping = {
        'grasslands': 1,
        'savannas': 1,
        'croplands': 3,
        'forest': 2,
        'evergreen needleleaf forests': 2,
        'evergreen broadleaf forests': 2,
        'deciduous needleleaf forests': 2,
        'deciduous broadleaf forests': 2,
        'mixed forests': 2,
        'open shrublands': 4,
        'closed shrublands': 4,
    }
    plant_type = plant_type_mapping.get(site_type.lower() if site_type else None, 0)

    # Example values for static variables (customize as needed)
    max_soil_moisture = 0.45  # m3/m3, typical for loam
    clay_content = clay_mean  # from raster extraction above
    soil_depth = 1.0  # meters, typical for many soils
    organic_carbon_stock = organic_carbon  # from raster extraction above

    # Initial pool values (user may want to adjust)
    root_pool_init = 0.1  # kgC/m2
    leaf_pool_init = 0.1  # kgC/m2
    stem_pool_init = 0.1  # kgC/m2

    static_ds = xr.Dataset(
        {
            "elevation": (("y", "x"), np.full((1, 1), elevation), {"units": "m"}),
            "plant_type": (("y", "x"), np.array([[plant_type]]), {"description": "Plant functional type"}),
            "max_soil_moisture": (("y", "x"), np.full((1, 1), max_soil_moisture), {"units": "m3 m-3"}),
            "clay_content": (("y", "x"), np.full((1, 1), clay_content), {"units": "percent"}),
            "soil_depth": (("y", "x"), np.full((1, 1), soil_depth), {"units": "m"}),
            "organic_carbon_stocks": (("y", "x"), np.full((1, 1), organic_carbon_stock), {"units": "kg m-2"}),
            "root_pool_init": (("y", "x"), np.full((1, 1), root_pool_init), {"units": "kgC m-2"}),
            "leaf_pool_init": (("y", "x"), np.full((1, 1), leaf_pool_init), {"units": "kgC m-2"}),
            "stem_pool_init": (("y", "x"), np.full((1, 1), stem_pool_init), {"units": "kgC m-2"}),
        },
        coords={"y": static_y, "x": static_x},
        attrs={
            "site_name": SITE_NAME,
            "center_lat": LAT,
            "center_lon": LON,
            "description": "Static site variables for model initialization",
            "history": f"Created by data_production.py on {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}",
        }
    )
    # Add CRS info using rioxarray
    static_ds = static_ds.rio.write_crs("EPSG:4326", inplace=True)

    static_out = os.path.join(WRK_DIR, 'inputData', f'{SITE_NAME}_static_site_variables.nc')
    os.makedirs(os.path.dirname(static_out), exist_ok=True)
    static_ds.to_netcdf(static_out)
    log_step_success(10, f"Static site variables saved to {static_out}.")


def main():
    """CLI entrypoint wrapper.

    Parses command-line arguments and forwards them to `run_pipeline`.
    """
    args = parser.parse_args()
    run_pipeline(
        args.wrk_dir,
        args.site,
        args.lat,
        args.lon,
        args.start_date,
        args.end_date,
    )

if __name__ == "__main__":
    main()
