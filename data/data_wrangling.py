import numpy as np
import xarray as xr

###################
# MCLEAN
###################
# McLean county coordinates
lat, lon = 40.61752192312278, -88.68366477345201

# Open the NetCDF dataset containing validation data
ds = xr.open_dataset("./data/VIC_validation.nc")

# Get the longitude and latitude arrays from the dataset
lons = ds.lon
lats = ds.lat

# Find the index of the nearest longitude to the specified lon
ix = (np.abs(lons - lon)).argmin().values
print(ix)  # Print the index for debugging

# Find the index of the nearest latitude to the specified lat
iy = (np.abs(lats - lat)).argmin().values
print(iy)  # Print the index for debugging

# Save the soil moisture data for the nearest location to a CSV file
np.savetxt(
    "./data/mclean_county_daily_soilM_VIC_mm_2016_2022.csv",
    ds.sel(lat=lat, lon=lon, method="nearest")["soilMoist"].to_numpy(),
)

# Load the input data from a NumPy .npz file
npz = np.load("./data/inputs.npz")

# Extract temperature, precipitation, and leaf area index data for the nearest location
tas = npz["tas"][ix, iy, :]
prcp = npz["prcp"][ix, iy, :]
lai = npz["lai"][ix, iy, :]

# Save the extracted data to CSV files
np.savetxt("./data/mclean_county_daily_tas_degC_2016-2022.csv", tas)
np.savetxt("./data/mclean_county_daily_prcp_mm_2016-2022.csv", prcp)
np.savetxt("./data/mclean_county_daily_lai_climatology.csv", lai)


###############
# GMFD
###############
# Read all climate data
tas = xr.open_mfdataset("../data/gmfd/tas_*.nc", combine="by_coords")
prcp = xr.open_mfdataset("../data/gmfd/prcp_*.nc", combine="by_coords")


# Subset to US
def subset_conus(ds):
    import regionmask

    countries = regionmask.defined_regions.natural_earth_v5_0_0.countries_50
    mask = countries.mask(tas.longitude, tas.latitude)
    mask_conus = mask == countries.map_keys("United States of America")

    ds_conus = ds.where(mask_conus, drop=True)
    ds_conus = ds_conus.where(
        (ds_conus.latitude <= 50) & (ds_conus.longitude > 230) & (ds_conus.latitude > 20),
        drop=True,
    )  # drop AL, HI

    return ds_conus


# Do it
tas_conus = subset_conus(tas)
prcp_conus = subset_conus(prcp)

# Store
ds = xr.merge(
    [
        tas_conus.convert_calendar(calendar="noleap", dim="time").isel(z=0).drop_vars("z"),
        prcp_conus.convert_calendar(calendar="noleap", dim="time").isel(z=0).drop_vars("z"),
    ]
)

ds.to_netcdf("../data/gmfd/conus.nc")
