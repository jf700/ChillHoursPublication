import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from scipy.ndimage import gaussian_filter
import matplotlib.colors as mcolors 



target_years = [2023, 2024, 2025, 2026, 2027]
Tc_celsius = 7.2

model_files = {
    "MIROC6": {
        "tasmin": "tasmin.MIROC6.ssp245.r1i1p1f1.2015-2044.LOCA_16thdeg_v20220413.west.nc",
        "tasmax": "tasmax.MIROC6.ssp245.r1i1p1f1.2015-2044.LOCA_16thdeg_v20220413.west.nc"
    },
    "CanESM5": {
        "tasmin": "tasmin.CanESM5.ssp245.r1i1p1f1.2015-2044.LOCA_16thdeg_v20220413.west.nc",
        "tasmax": "tasmax.CanESM5.ssp245.r1i1p1f1.2015-2044.LOCA_16thdeg_v20220413.west.nc"
    },
    "HadGEM3-GC31-LL": {
        "tasmin": "tasmin.HadGEM3-GC31-LL.ssp245.r1i1p1f3.2015-2044.LOCA_16thdeg_v20220413.west.nc",
        "tasmax": "tasmax.HadGEM3-GC31-LL.ssp245.r1i1p1f3.2015-2044.LOCA_16thdeg_v20220413.west.nc"
    },
    "CNRM-CM6-1": {
        "tasmin": "tasmin.CNRM-CM6-1.ssp245.r1i1p1f2.2015-2044.LOCA_16thdeg_v20220413.west.nc",
        "tasmax": "tasmax.CNRM-CM6-1.ssp245.r1i1p1f2.2015-2044.LOCA_16thdeg_v20220413.west.nc"
    }
}

# === Chill hour calculation function ===
def compute_corrected_chill_hours(tasmin, tasmax, latitudes, times, smoothing_sigma=5):
    season_c = tasmin - 273.15 # Convert to Celsius
    temp_range = tasmax - tasmin
    temp_range = xr.where(temp_range < 0.5, np.nan, temp_range) # Handle very small temp range
    ratio = (Tc_celsius + 273.15 - tasmin) / temp_range
    ratio = ratio.clip(min=-1, max=1)

    CD = times.dayofyear
    CD_expanded = CD.broadcast_like(tasmin)
    lat_rad = np.radians(latitudes % 360)

    # Calculate daylight hours (DL) based on latitude and day of year
    DL = xr.where(
        latitudes <= 40,
        12.14 + 3.34 * np.tan(lat_rad) * np.cos(0.0172 * CD_expanded - 1.95),
        12.25 + ((1.6164 + 1.7643 * (np.tan(lat_rad))**2) * np.cos(0.0172 * CD_expanded - 1.95))
    )
    DL = xr.where(DL < 0, 0, DL) # Ensure daylight hours are not negative

    # Calculate raw chill hours (CH)
    CH = xr.where(season_c < Tc_celsius, 24 - DL, 0)
    chill_hours = CH.sum(dim='time')

    # === Apply regional correction with smoothing ===
    lat = chill_hours['lat'].values
    lon = chill_hours['lon'].values
    lat_grid, lon_grid = np.meshgrid(lat, lon, indexing='ij')

    # Initialize correction array with ones
    # Use float dtype to ensure correct calculations after smoothing
    correction = np.ones_like(chill_hours, dtype=float)

    correction[(lat_grid >= 35) & (lat_grid <= 39) & (lon_grid >= -122) & (lon_grid <= -119)] = 0.68
    correction[lat_grid < 35] = 0.29
    correction[lat_grid > 39] = 1.31

    # Apply a Gaussian filter to smooth the transitions in the correction array
    # sigma controls the amount of smoothing. Adjust this value to fine-tune.
    smoothed_correction_array = gaussian_filter(correction, sigma=smoothing_sigma)

    # Convert the smoothed NumPy array back to an xarray DataArray
    # This ensures it aligns correctly with chill_hours for multiplication.
    smoothed_correction_da = xr.DataArray(
        smoothed_correction_array,
        coords=chill_hours.coords,
        dims=chill_hours.dims
    )

    return chill_hours * smoothed_correction_da # Apply the smoothed correction

# === Loop over models, compute and sum chill hours ===
ensemble_sum = None
model_count = 0

for model_name, paths in model_files.items():
    print(f"Processing: {model_name}")
    try:
        # Open full datasets and then select specific seasons in the loop for clarity and efficiency
        ds_min = xr.open_dataset(paths["tasmin"])["tasmin"]
        ds_max = xr.open_dataset(paths["tasmax"])["tasmax"]
        latitudes = ds_min["lat"]

        chill_stack = []
        for year in target_years:
            # Define the chill season: Oct 1st of (year-1) to Jun 1st of year
            start_date = f"{year - 1}-10-01"
            end_date = f"{year}-06-01"

            tasmin_season = ds_min.sel(time=slice(start_date, end_date))
            tasmax_season = ds_max.sel(time=slice(start_date, end_date))

            # Check if selection yielded any data (important if target_years extend beyond data availability)
            if tasmin_season.time.size == 0 or tasmax_season.time.size == 0:
                print(f"Warning: No data found for {year-1}/{year} season in {model_name}. Skipping.")
                continue

            # Compute chill hours with the smoothed correction
            # You can adjust 'smoothing_sigma' here for different levels of smoothness
            chill = compute_corrected_chill_hours(tasmin_season, tasmax_season, latitudes, tasmin_season['time'].dt, smoothing_sigma=5)
            chill_stack.append(chill)

        if not chill_stack: # If no chill data was computed for this model
            print(f"No valid chill hours computed for {model_name}. Skipping model average.")
            continue

        avg_chill = sum(chill_stack) / len(chill_stack)

        if ensemble_sum is None:
            ensemble_sum = avg_chill
        else:
            ensemble_sum += avg_chill
        model_count += 1
    except FileNotFoundError:
        print(f"Error: One or more files for {model_name} not found. Please check file paths.")
        continue # Skip to the next model

# === Final ensemble average ===
if model_count == 0:
    print("No models were processed successfully. Cannot compute ensemble average or generate plot.")
    exit() # Exit if no data to process further

ensemble_avg = ensemble_sum / model_count
ensemble_avg.name = "Ensemble Chill Hours"
ensemble_avg.to_netcdf("ensemble_2025_avg_ssp245.nc")

# === Area Calculation for Almond Chill Hours (200-500) ===
print("\nCalculating area for almond chill hours (200-500)...")

# Filter the ensemble average for values between 200 and 500
almond_chill_mask = (ensemble_avg >= 200) & (ensemble_avg <= 500)
almond_chill_data = ensemble_avg.where(almond_chill_mask)

# Get latitude and longitude arrays
lats = almond_chill_data['lat'].values
lons = almond_chill_data['lon'].values


# LOCA data is 1/16th degree, which is approx 0.0625 degrees.
R_earth_km = 6371

# Calculate cell area in km^2
# Area = (delta_lon * R * cos(lat)) * (delta_lat * R)
# Where delta_lon and delta_lat are in radians

# Determine the grid resolution
lat_res = np.abs(np.diff(lats)[0]) if len(lats) > 1 else 0.0625 # Fallback if only one lat
lon_res = np.abs(np.diff(lons)[0]) if len(lons) > 1 else 0.0625 # Fallback if only one lon

# Convert resolutions to radians
lat_res_rad = np.radians(lat_res)
lon_res_rad = np.radians(lon_res)

# Create a 2D array of cell areas
cell_areas_km2 = np.zeros_like(almond_chill_data.values, dtype=float)
for i, lat in enumerate(lats):
    # Width of cell (longitude extent) varies with latitude
    width_km = R_earth_km * np.cos(np.radians(lat)) * lon_res_rad
    # Height of cell (latitude extent) is constant
    height_km = R_earth_km * lat_res_rad
    cell_areas_km2[i, :] = width_km * height_km

# Sum the areas where the almond_chill_mask is True
total_almond_chill_area_km2 = np.nansum(cell_areas_km2 * almond_chill_mask.values)

print(f"Total area suitable for almond chill hours (200-500) within the plotted region: {total_almond_chill_area_km2:.2f} km²")

# === Plot ===
fig, ax = plt.subplots(figsize=(12, 8), subplot_kw={'projection': ccrs.PlateCarree()})

# Get the 'Blues' colormap
cmap_blues = plt.colormaps.get_cmap('Blues')

# Create a copy to avoid modifying the original colormap if it's used elsewhere
modified_cmap = cmap_blues.copy()

# Set the 'under' color (for values < vmin) to white
modified_cmap.set_under('white')
# Set the 'over' color (for values > vmax) to white
modified_cmap.set_over('white')

# Plot only the chill hours for almonds (200-500)
almond_chill_data.plot(
    ax=ax,
    transform=ccrs.PlateCarree(),
    cmap=modified_cmap,
    vmin=200,            # Only show colors for values >= 200
    vmax=500,            # Only show colors for values <= 500
    extend='both',       # This ensures the colorbar shows triangles for under/over
    cbar_kwargs={'label': 'Avg Corrected Chill Hours (200-500 for Almonds)'}
)
ax.add_feature(cfeature.STATES, linewidth=1.0)
ax.add_feature(cfeature.BORDERS, linestyle=":")
ax.add_feature(cfeature.COASTLINE)
ax.set_title("Ensemble Chill Hours for Almonds (200-500) — CMIP6 SSP2-4.5 (Smoothed Correction)", fontsize=14)

plt.tight_layout()
plt.show()