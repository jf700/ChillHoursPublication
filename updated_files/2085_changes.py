import xarray as xr
import numpy as np


ds_245 = xr.open_dataarray("ensemble_2045_avg_ssp245.nc")
ds_585 = xr.open_dataarray("ensemble_2045_avg_ssp585.nc")


# === Define California's geographical bounds ===
# Using 0-360 convention for longitudes
california_min_lon = 235
california_max_lon = 246
california_min_lat = 32.0
california_max_lat = 43.0


# === Slice both DataArrays to California ===

ds_245_california = ds_245.sel(
    lon=slice(california_min_lon, california_max_lon),
    lat=slice(california_min_lat, california_max_lat)
)

ds_585_california = ds_585.sel(
    lon=slice(california_min_lon, california_max_lon),
    lat=slice(california_min_lat, california_max_lat)
)

# === Compute changes for overall California ===

delta_california = ds_585_california - ds_245_california
percent_change_california = (delta_california / ds_245_california) * 100

# These masks will select data points from the already California-sliced DataArrays.
regions_data = {}

# Northern California: Latitude > 39 N
northern_mask = (ds_245_california.lat > 39)
regions_data['Northern California'] = {
    '245': ds_245_california.where(northern_mask),
    '585': ds_585_california.where(northern_mask)
}

# Central California: Latitude 35-39 N AND Longitude 238-241 E (which is -122 to -119 W)
central_mask = (ds_245_california.lat >= 35) & \
               (ds_245_california.lat <= 39) & \
               (ds_245_california.lon >= 238) & \
               (ds_245_california.lon <= 241)
regions_data['Central California'] = {
    '245': ds_245_california.where(central_mask),
    '585': ds_585_california.where(central_mask)
}

# Southern California: Latitude < 35 N
southern_mask = (ds_245_california.lat < 35)
regions_data['Southern California'] = {
    '245': ds_245_california.where(southern_mask),
    '585': ds_585_california.where(southern_mask)
}


# === Print information for California and its sub-regions ===
print("\nChill Hour Comparison for 2045 (SSP2-4.5 vs SSP5-8.5) - California Only\n")

# Overall California
print(f"California Region (Overall):")
print(f"- Avg (SSP2-4.5): {ds_245_california.mean().item():.2f} hours")
print(f"- Avg (SSP5-8.5): {ds_585_california.mean().item():.2f} hours")
print(f"- Change in Absolute  : {delta_california.mean().item():.2f} hours")
print(f"- Change in Percent   : {percent_change_california.mean().item():.2f}%\n")

# Per-region breakdown
for region_name, data in regions_data.items():
    sub_245 = data['245']
    sub_585 = data['585']

    sub_delta = sub_585 - sub_245
    sub_percent = (sub_delta / sub_245) * 100

    print(f"{region_name}:")
    print(f"- Avg (SSP2-4.5): {sub_245.mean().item():.2f} hours")
    print(f"- Avg (SSP5-8.5): {sub_585.mean().item():.2f} hours")
    print(f"- Change in Absolute  : {sub_delta.mean().item():.2f} hours")
    print(f"- Change in Percent   : {sub_percent.mean().item():.2f}%\n")