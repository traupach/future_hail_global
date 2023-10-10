"""
regrid_ERA5 - regrid ERA5 data to coarser output resolution and save as nc.
"""

import os
import sys
import dask
import warnings
import xarray
import xesmf as xe
import numpy as np
import fut_hail as fh
from glob import glob
import modules.parcel_functions as parcel
from dask.distributed import Client, Scheduler, LocalCluster

# Settings.
warnings.filterwarnings("ignore", category=FutureWarning)    # Ignore FutureWarnings (in Dask). 
outdir = '/g/data/up6/tr2908/future_hail_global/era5_1deg/'  # Processing output directory.
out_res = 1                                                  # Output resolution (degrees).
era5_dir = '/g/data/rt52/era5/pressure-levels/reanalysis/'   # Directory with ERA5 pressure level data.
variables = '[t,q,u,v,z]'                                    # ERA5 variables to read.
num_vars = 5                                                 # No. variables to expect.       
rename_map = {'t': 'temperature', 'q': 'specific_humidity'}  # Rename map for variables.
years = np.arange(1980, 2000)

# Start a dask cluster.
print('Starting cluster...')
client = Client(scheduler_file='sched_' + os.environ['PBS_JOBID'] + '.json')
print(client)

# Set up output grid.
out_lon = np.arange(-180.0, 180.0 + out_res, out_res)
out_lat = np.arange(-90, 90 + out_res, out_res)
out_grid = xarray.Dataset(coords={'lat': (out_lat[:-1] + out_lat[1:]) / 2,
                                  'lon': (out_lon[:-1] + out_lon[1:]) / 2, 
                                  'lat_b': out_lat,
                                  'lon_b': out_lon})

# Loop through each month of the year.
regridder = None
for year in years:
    for month in np.arange(1,13):
        print(f'{year} - {month}')
        outfile = f'{outdir}/era5_1deg_{year}{month:02}.nc'
        if os.path.exists(outfile):
            print(f'Skipping existing {outfile}.')
            continue
    
        month_files = glob(f'{era5_dir}/{variables}/{year}/*{year}{month:02}*.nc')
        if len(month_files) != num_vars:
            print(f'Variable missing for {month}, skipping this month.')
    
        era5 = []
        for month_file in month_files:
            dat = xarray.open_dataset(month_file, chunks={'time': 1, 'latitude': -1, 'longitude': -1, 'level': 1})
            times = dat.time.where(dat.time.dt.hour % 6 == 0, drop=True).values
            dat = dat.sel(time=times).chunk({'time': 30, 'latitude': -1, 'longitude': -1, 'level': 1})
            era5.append(dat)
        era5 = xarray.merge(era5)
        era5 = era5.rename(rename_map)
    
        # Regrid to coarser resolution.
        if regridder is None:
            regridder = xe.Regridder(era5, out_grid, 'bilinear', periodic=True)
            
        attrs = {'history': f'ERA5 regridded to {out_res} x {out_res} degree grid using xESMF.'}
        era5 = regridder(era5, keep_attrs=True).load()
        era5 = era5.rename({'lat': 'latitude', 'lon': 'longitude'})
        fh.write_output(dat=era5, file=outfile, attrs=attrs)
