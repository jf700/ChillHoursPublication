import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
import matplotlib.ticker as mticker
from scipy.ndimage import gaussian_filter
import geopandas as gpd

import os

target_years = [2023,2024, 2025, 2026, 2027] #adjust for each time period
Tc_celsius = 7.2
processed_data_filename = r"updated_files\2025_245_ensemble_chill_hours.nc"
shapefile_path = r"vineyards\vineyards.shp"
base_dir = "loca_downloads"
scenario = "ssp245"

model_files = {
    "MIROC6": {
        "tasmin": f"{base_dir}/tasmin.MIROC6.{scenario}.r1i1p1f1.2015-2044.LOCA_16thdeg_v20220413.nc",
        "tasmax": f"{base_dir}/tasmax.MIROC6.{scenario}.r1i1p1f1.2015-2044.LOCA_16thdeg_v20220413.nc",
    },
    "CanESM5": {
        "tasmin": f"{base_dir}/tasmin.CanESM5.{scenario}.r1i1p1f1.2015-2044.LOCA_16thdeg_v20220413.nc",
        "tasmax": f"{base_dir}/tasmax.CanESM5.{scenario}.r1i1p1f1.2015-2044.LOCA_16thdeg_v20220413.nc",
    },
    "HadGEM3-GC31-LL": {
        "tasmin": f"{base_dir}/tasmin.HadGEM3-GC31-LL.{scenario}.r1i1p1f3.2015-2044.LOCA_16thdeg_v20220413.nc",
        "tasmax": f"{base_dir}/tasmax.HadGEM3-GC31-LL.{scenario}.r1i1p1f3.2015-2044.LOCA_16thdeg_v20220413.nc",
    },
    "CNRM-CM6-1": {
        "tasmin": f"{base_dir}/tasmin.CNRM-CM6-1.{scenario}.r1i1p1f2.2015-2044.LOCA_16thdeg_v20220413.nc",
        "tasmax": f"{base_dir}/tasmax.CNRM-CM6-1.{scenario}.r1i1p1f2.2015-2044.LOCA_16thdeg_v20220413.nc",
    },
}


def open_merged(path, varname):
    ds = xr.open_dataset(path)
    return ds[varname]

def get_hourly_temperatures(tmin_day, tmax_day, Tc):
    hourly_temps = np.zeros(24)

    if tmax_day < tmin_day:
        tmax_day = tmin_day

    t_min_hour = 6
    t_max_hour = 14

    if (t_max_hour - t_min_hour) == 0:
        for h in range(t_min_hour, t_max_hour + 1):
            hourly_temps[h] = tmin_day
    else:
        for h in range(t_min_hour, t_max_hour + 1):
            phase = np.pi * (h - t_min_hour) / (t_max_hour - t_min_hour)
            hourly_temps[h] = tmin_day + (tmax_day - tmin_day) * np.sin(phase)

    period_night = 24 - (t_max_hour - t_min_hour)
    if period_night == 0:
        for h in range(t_max_hour + 1, 24):
            hourly_temps[h] = tmax_day
        for h in range(0, t_min_hour):
            hourly_temps[h] = tmax_day
    else:
        for h in range(t_max_hour + 1, 24):
            phase = np.pi * (h - t_max_hour) / period_night
            hourly_temps[h] = tmax_day - (tmax_day - tmin_day) * np.sin(phase)

        for h in range(0, t_min_hour):
            phase = np.pi * (h + period_night - (24 - t_min_hour)) / period_night
            hourly_temps[h] = tmax_day - (tmax_day - tmin_day) * np.sin(phase)
            
    hourly_temps = np.clip(hourly_temps, tmin_day, tmax_day)

    return hourly_temps

def compute_corrected_chill_hours_weinberger(tasmin, tasmax, smoothing_sigma=5):
    tasmin_c, tasmax_c = tasmin - 273.15, tasmax - 273.15
    num_lats, num_lons = tasmin_c.shape[1], tasmin_c.shape[2]
    total_chill_hours_season = xr.zeros_like(tasmin_c.isel(time=0), dtype=float)
    for i in range(tasmin_c.shape[0]):
        current_tasmin_day_data, current_tasmax_day_data = tasmin_c.isel(time=i).values, tasmax_c.isel(time=i).values
        daily_chill_for_current_day = np.zeros((num_lats, num_lons), dtype=float)
        for lat_idx in range(num_lats):
            for lon_idx in range(num_lons):
                tmin_val, tmax_val = current_tasmin_day_data[lat_idx, lon_idx], current_tasmax_day_data[lat_idx, lon_idx]
                if np.isnan(tmin_val) or np.isnan(tmax_val):
                    daily_chill_for_current_day[lat_idx, lon_idx] = np.nan
                    continue
                estimated_hourly_temps_c = get_hourly_temperatures(tmin_val, tmax_val, Tc_celsius)
                daily_chill_for_current_day[lat_idx, lon_idx] = np.sum((estimated_hourly_temps_c > 0) & (estimated_hourly_temps_c < Tc_celsius))
        total_chill_hours_season += xr.DataArray(daily_chill_for_current_day, coords=total_chill_hours_season.coords, dims=total_chill_hours_season.dims)
    chill_hours = total_chill_hours_season
    lat, lon = chill_hours['lat'].values, chill_hours['lon'].values
    lat_grid, lon_grid = np.meshgrid(lat, lon, indexing='ij')
    correction = np.ones_like(chill_hours, dtype=float)
    correction[lat_grid < 35] = 0.8
    correction[(lat_grid >= 35) & (lat_grid <= 39) & (lon_grid >= 235) & (lon_grid <= 242)] = 1.45
    correction[lat_grid > 39] = 1.0
    smoothed_correction_array = gaussian_filter(correction, sigma=smoothing_sigma)
    return chill_hours * xr.DataArray(smoothed_correction_array, coords=chill_hours.coords, dims=chill_hours.dims)

# main processing block or load from file
if os.path.exists(processed_data_filename):
    print(f"Loading data from {processed_data_filename}")
    ensemble_avg = xr.open_dataarray(processed_data_filename)
else:
    ensemble_sum = None
    model_count = 0
    for model_name, paths in model_files.items():
        try:
            ds_min_full = open_merged(paths["tasmin"], "tasmin")
            ds_max_full = open_merged(paths["tasmax"], "tasmax")

            chill_stack = []
            for year in target_years:
                start_year_selection, end_year_selection = year - 1, year
                time_mask = ((ds_min_full.time.dt.year == start_year_selection) & (ds_min_full.time.dt.month.isin([11, 12]))) | \
                            ((ds_min_full.time.dt.year == end_year_selection) & (ds_min_full.time.dt.month.isin([1, 2])))

                tasmin_season = ds_min_full.isel(time=time_mask)
                tasmax_season = ds_max_full.isel(time=time_mask)

                chill = compute_corrected_chill_hours_weinberger(
                    tasmin_season.sortby('time'),
                    tasmax_season.sortby('time')
                )
                chill_stack.append(chill)

            avg_chill = sum(chill_stack) / len(chill_stack)
            ensemble_sum = avg_chill if ensemble_sum is None else ensemble_sum + avg_chill
            model_count += 1

        except Exception as e:
            print(f"Error processing {model_name}: {e}")
            continue
    
    if model_count > 0:
        ensemble_avg = ensemble_sum / model_count
        ensemble_avg.to_netcdf(processed_data_filename)
        print(f"Saved processed data to {processed_data_filename}")
    else:
        exit()

# --- clipping data ---
ensemble_avg.coords['lon'] = (ensemble_avg.coords['lon'] + 180) % 360 - 180 
ensemble_avg = ensemble_avg.sortby(ensemble_avg.lon)
ensemble_avg.rio.write_crs("EPSG:4326", inplace=True)
ensemble_avg.rio.set_spatial_dims(x_dim="lon", y_dim="lat", inplace=True)

fruit_nuts_area = gpd.read_file(shapefile_path).to_crs("EPSG:4326")

clipped_chill_hours = ensemble_avg.rio.clip(fruit_nuts_area.geometry.values, all_touched=True)

# --- plotting data ---
fig, ax = plt.subplots(figsize=(12, 8), subplot_kw={'projection': ccrs.PlateCarree()})

im = clipped_chill_hours.plot.pcolormesh(
    ax=ax,
    transform=ccrs.PlateCarree(),
    cmap='YlGnBu', 
    vmin = 200,
    vmax = 1200,
    cbar_kwargs={'label': 'Avg Corrected Chill Hours'}
)
# Add red line at chill hour threshold per crop
cbar_ax = im.colorbar.ax
cbar_ax.axhline(y=200, color='red', linewidth=8)
cbar_ax.text(-1.3, 200, '200', color='red', fontweight='bold', fontsize=12, va='center')

ax.add_feature(cfeature.STATES, linewidth=0.8, edgecolor='black')
ax.add_feature(cfeature.COASTLINE, linewidth=1.0)
ax.add_feature(cfeature.BORDERS, linestyle=':', edgecolor='gray')
minx, miny, maxx, maxy = fruit_nuts_area.total_bounds
ax.set_extent([minx - 1, maxx + 1, miny - 1, maxy + 1], crs=ccrs.PlateCarree())
ax.set_title("2083-2087", fontsize=30)
gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True, linewidth=0.5, color='gray', alpha=0.5, linestyle='--')
gl.top_labels, gl.right_labels = False, False
gl.xformatter, gl.yformatter = LongitudeFormatter(), LatitudeFormatter()
gl.xlocator = mticker.MaxNLocator(4)
gl.ylocator = mticker.MaxNLocator(4)
gl.xlabel_style = {'size': 18, 'weight': 'bold'}
gl.ylabel_style = {'size': 18, 'weight': 'bold'}

plt.tight_layout()
plt.show()

# --- Acreage calculation ---
# chill hour thresholds for each crop
crops = {
    "Almonds": (200, 300),
    "Pistachios": (800, 1200),
    "Peaches": (650, 850),
    "Plums": (700, 1000)
}

print("\n" + "="*60)
print(f"Shapefile: {os.path.basename(shapefile_path)}")
print("-"*60)
print(f"{'Crop':<12} | {'Chill Range':<15} | {'Total Acres'}")
print("-"*60)

# Calculate area of each grid cell 
R = 6371.0  # Earth radius in km
lat_res = abs(clipped_chill_hours['lat'][1].item() - clipped_chill_hours['lat'][0].item())
lon_res = abs(clipped_chill_hours['lon'][1].item() - clipped_chill_hours['lon'][0].item())
dlat_rad = np.deg2rad(lat_res)
dlon_rad = np.deg2rad(lon_res)
lats_rad = np.deg2rad(clipped_chill_hours['lat'])
cell_areas_sq_km = (R**2) * np.cos(lats_rad) * dlat_rad * dlon_rad
area_grid = cell_areas_sq_km.broadcast_like(clipped_chill_hours)
sq_km_to_acres = 247.105

for crop_name, (min_chill, max_chill) in crops.items():
    final_filtered_chill = clipped_chill_hours.where(
        (clipped_chill_hours >= min_chill) & (clipped_chill_hours <= max_chill)
    )
    total_area_sq_km = area_grid.where(~np.isnan(final_filtered_chill)).sum().item()
    
    total_area_acres = total_area_sq_km * sq_km_to_acres
    chill_range_str = f"{min_chill}-{max_chill}"
    print(f"{crop_name:<12} | {chill_range_str:<15} | {total_area_acres:,.2f}")