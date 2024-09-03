import numpy as np
import xarray as xr

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
