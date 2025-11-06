from turtle import pd
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import geopandas as gpd
import os
import glob
import rioxarray 
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
from scipy.ndimage import gaussian_filter

# === Settings ===
target_years = [1983, 1984, 1985, 1986, 1987]
Tc_celsius = 7.2 
PRISM_DATA_DIR = r"C:\Users\Josh\Documents\ChillResearch\prism_data"
PROCESSED_DATA_DIR = r"C:\Users\Josh\Documents\ChillResearch\processed_data"
shapefile_path = r"C:\Users\Josh\Documents\ChillResearch\Fruit_Nuts\Fruits_Nuts.shp"

# Ensure the output directory exists
os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)

# === Function to estimate hourly temperatures for a single day/grid (for Weinberger) ===
def get_hourly_temperatures(tmin_day, tmax_day, Tc):
    hours = np.arange(0, 24)
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

# === Chill hour calculation function (no changes needed) ===
def compute_chill_hours_weinberger_prism(tasmin, tasmax):
    tasmin_c = tasmin
    tasmax_c = tasmax

    if tasmin_c.ndim != 3 or tasmax_c.ndim != 3:
        raise ValueError(f"Expected 3 dimensions (time, lat, lon) for temperature data, got {tasmin_c.ndim} for tasmin and {tasmax_c.ndim} for tasmax.")

    num_time, num_lats, num_lons = tasmin_c.shape[0], tasmin_c.shape[1], tasmin_c.shape[2]
    
    total_chill_hours_season = xr.DataArray(
        np.zeros((num_lats, num_lons), dtype=float),
        coords={'lat': tasmin_c['lat'], 'lon': tasmin_c['lon']},
        dims=['lat', 'lon']
    )
    
    for i in range(num_time): # Loop through each day
        current_tasmin_day_data = tasmin_c.isel(time=i).values
        current_tasmax_day_data = tasmax_c.isel(time=i).values

        daily_chill_for_current_day = np.zeros((num_lats, num_lons), dtype=float)

        for lat_idx in range(num_lats): # Loop through each latitude grid cell
            for lon_idx in range(num_lons): # Loop through each longitude grid cell
                tmin_val = current_tasmin_day_data[lat_idx, lon_idx]
                tmax_val = current_tasmax_day_data[lat_idx, lon_idx]

                if np.isnan(tmin_val) or np.isnan(tmax_val):
                    daily_chill_for_current_day[lat_idx, lon_idx] = np.nan
                    continue

                estimated_hourly_temps_c = get_hourly_temperatures(tmin_val, tmax_val, Tc_celsius)

                current_daily_ch_count = np.sum((estimated_hourly_temps_c > 0) & (estimated_hourly_temps_c < Tc_celsius))
                
                daily_chill_for_current_day[lat_idx, lon_idx] = current_daily_ch_count
        
        total_chill_hours_season += xr.DataArray(
            daily_chill_for_current_day,
            coords=total_chill_hours_season.coords,
            dims=total_chill_hours_season.dims
        )
    
    return total_chill_hours_season

# === Function to load PRISM data for a given year and variable ===
def load_prism_data(variable, year, prism_data_dir):
    files_to_load = []
    
    for month in [11, 12]:
        path_pattern = os.path.join(prism_data_dir, f"PRISM_{variable}_*_4kmD2_{year-1}{month:02d}*_bil.bil")
        files_to_load.extend(glob.glob(path_pattern))
    
    for month in [1, 2]:
        path_pattern = os.path.join(prism_data_dir, f"PRISM_{variable}_*_4kmD2_{year}{month:02d}*_bil.bil")
        files_to_load.extend(glob.glob(path_pattern))
    
    if not files_to_load:
        print(f"Warning: No PRISM {variable} files found for chill season ending in {year}.")
        return None

    try:
        files_to_load.sort()
        datasets = []
        for fpath in files_to_load:
            try:
                ds_single = rioxarray.open_rasterio(fpath, chunks='auto').squeeze('band', drop=True)
                filename = os.path.basename(fpath)
                date_str = filename.split('_')[4][:8]
                file_date = pd.to_datetime(date_str)
                ds_single = ds_single.rename(variable).rename({'x': 'lon', 'y': 'lat'})
                ds_single_with_time = ds_single.expand_dims(time=[file_date])
                datasets.append(ds_single_with_time)
            except Exception as e_single:
                print(f"Error opening single PRISM file {fpath}: {e_single}. Skipping file.")
                continue

        if not datasets: return None
        ds_combined = xr.concat(datasets, dim='time').sortby('time')

        if ds_combined.rio.crs and ds_combined.rio.crs != "EPSG:4326":
            ds_combined = ds_combined.rio.reproject("EPSG:4326")
            if 'x' in ds_combined.dims and 'y' in ds_combined.dims:
                ds_combined = ds_combined.rename({'x': 'lon', 'y': 'lat'})
        
        return ds_combined.transpose('time', 'lat', 'lon')
    
    except Exception as e:
        print(f"An unexpected error occurred while loading PRISM {variable} data for year {year}: {e}")
        return None

# === Process PRISM data, compute and sum chill hours (WITH CACHING) ===
prism_chill_years = []

for year in target_years:
    output_filepath = os.path.join(PROCESSED_DATA_DIR, f"prism_chill_hours_{year}.nc")
    
    if os.path.exists(output_filepath):
        print(f"Loading pre-computed chill hours for {year} from: {output_filepath}")
        chill = xr.open_dataarray(output_filepath)
        prism_chill_years.append(chill)
        continue

    print(f"No pre-computed file found. Processing PRISM data for chill season ending {year}...")
    tasmin_season = load_prism_data('tmin', year, PRISM_DATA_DIR)
    tasmax_season = load_prism_data('tmax', year, PRISM_DATA_DIR)

    if tasmin_season is None or tasmax_season is None:
        print(f"Skipping year {year} due to missing PRISM data.")
        continue

    try:
        tasmin_season, tasmax_season = xr.align(tasmin_season, tasmax_season, join="inner")
    except ValueError as e:
        print(f"Error aligning tasmin and tasmax for year {year}: {e}. Skipping year.")
        continue

    if tasmin_season.sizes['time'] == 0:
        print(f"No common time steps for year {year} after alignment. Skipping.")
        continue

    chill = compute_chill_hours_weinberger_prism(tasmin_season, tasmax_season)
    chill = chill.expand_dims(year=[year])

    try:
        print(f"Saving computed chill hours for {year} to: {output_filepath}")
        chill.to_netcdf(output_filepath)
        prism_chill_years.append(chill)
    except Exception as e:
        print(f"Could not save NetCDF file for year {year}. Error: {e}")

# === Final PRISM Average, Clipping, Plotting, and Acreage Calculation ===
if prism_chill_years:
    all_chill_years = xr.concat(prism_chill_years, dim='year')
    prism_avg_chill = all_chill_years.mean(dim='year')
    prism_avg_chill.name = "PRISM Chill Hours"

    # Set spatial information for rioxarray
    prism_avg_chill.rio.write_crs("EPSG:4326", inplace=True)
    prism_avg_chill.rio.set_spatial_dims(x_dim="lon", y_dim="lat", inplace=True)
    
    # Load the shapefile and ensure it has the correct CRS
    fruit_nuts_area = gpd.read_file(shapefile_path).to_crs("EPSG:4326")

    # Clip the chill hour grid using the shapefile.
    clipped_chill_hours = prism_avg_chill.rio.clip(fruit_nuts_area.geometry.values, all_touched=True)

    # --- Plotting Section ---
    fig, ax = plt.subplots(figsize=(12, 8), subplot_kw={'projection': ccrs.PlateCarree()})

    clipped_chill_hours.plot.pcolormesh(
        ax=ax,
        transform=ccrs.PlateCarree(),
        cmap='YlGnBu',
        vmin = 200,
        vmax = 1200,
        cbar_kwargs={'label': 'Avg Corrected Chill Hours'}
    )
    ax.add_feature(cfeature.STATES, linewidth=0.8, edgecolor='black')
    ax.add_feature(cfeature.COASTLINE, linewidth=1.0)
    ax.add_feature(cfeature.BORDERS, linestyle=':', edgecolor='gray')
    minx, miny, maxx, maxy = fruit_nuts_area.total_bounds
    ax.set_extent([minx - 1, maxx + 1, miny - 1, maxy + 1], crs=ccrs.PlateCarree())
    ax.set_title(f"Average PRISM Chill Hours (Nov-Feb, {target_years[0]-1}-{target_years[-1]}) for Fruit and Nuts Area", fontsize=12)
    gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True, linewidth=0.5, color='gray', alpha=0.5, linestyle='--')
    gl.top_labels, gl.right_labels = False, False
    gl.xformatter, gl.yformatter = LongitudeFormatter(), LatitudeFormatter()

    plt.tight_layout()
    plt.show()

    # === NEW: Acreage Calculation Section for Multiple Crops ===

    # Define the chill hour requirements for each crop
    crops = {
        "Almonds": (200, 300),
        "Pistachios": (800, 1200),
        "Peaches": (650, 850),
        "Plums": (700, 1000)
    }

    print("\n" + "="*60)
    print("Calculating Acreage for Specific Crop Chill Hour Requirements")
    print(f"PRISM DATA")
    print(f"Shapefile: {os.path.basename(shapefile_path)}")
    print("-"*60)
    print(f"{'Crop':<12} | {'Chill Range':<15} | {'Total Acres'}")
    print("-"*60)

    # Calculate area of each grid cell (reusable for all crops)
    R = 6371.0  # Earth radius in km
    lat_res = abs(clipped_chill_hours['lat'][1].item() - clipped_chill_hours['lat'][0].item())
    lon_res = abs(clipped_chill_hours['lon'][1].item() - clipped_chill_hours['lon'][0].item())
    dlat_rad = np.deg2rad(lat_res)
    dlon_rad = np.deg2rad(lon_res)
    lats_rad = np.deg2rad(clipped_chill_hours['lat'])
    cell_areas_sq_km = (R**2) * np.cos(lats_rad) * dlat_rad * dlon_rad
    area_grid = cell_areas_sq_km.broadcast_like(clipped_chill_hours)
    sq_km_to_acres = 247.105

    # Loop through each crop to calculate its specific acreage
    for crop_name, (min_chill, max_chill) in crops.items():
        # Filter the clipped data for the crop's specific chill hour range
        final_filtered_chill = clipped_chill_hours.where(
            (clipped_chill_hours >= min_chill) & (clipped_chill_hours <= max_chill)
        )
        
        # Sum the areas of valid pixels for the current crop
        total_area_sq_km = area_grid.where(~np.isnan(final_filtered_chill)).sum().item()
        
        # Convert to acres
        total_area_acres = total_area_sq_km * sq_km_to_acres
        
        # Print the result for the current crop
        chill_range_str = f"{min_chill}-{max_chill}"
        print(f"{crop_name:<12} | {chill_range_str:<15} | {total_area_acres:,.2f}")

    print("="*60)

else:
    print("No PRISM data processed. Check data paths, target years, and file naming conventions.")