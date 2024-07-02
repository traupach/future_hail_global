import os
import sys
import xarray
import numpy as np
import pandas as pd
from glob import glob
from dask.distributed import Client

## NB requires use of conda/analysis3-24.04 (earlier versions do not have a recent-enough NetCDF installation).
from cmethods import adjust

base_dir = '/g/data/up6/tr2908/future_hail_global/CMIP_conv/'     # Base directory under which to find CMIP6 files.
out_dir = '/g/data/up6/tr2908/future_hail_global/CMIP_detrended/ingredients/' # Output directory for cached outputs.
runs_file = '~/git/future_hail_global/results/runs_list.csv'      # Range of dates for each run. 
epoch = '3C'                                                      # The epoch to detrend.
exclude_vars = ['shear_u', 'shear_v', 'positive_shear']           # Variables not to detrend.

# Get model number to process from the command line.
assert len(sys.argv) == 2, 'Usage: calc_proxies.py <model_number>'
model_num = int(sys.argv[1])

# Start a dask cluster.
print('Starting cluster...')
client = Client(scheduler_file='sched_' + os.environ['PBS_JOBID'] + '.json')
print(client)

# Get model directory.
model_dirs = sorted(glob(f'{base_dir}/*'))
model_dir = model_dirs[model_num]
model = os.path.basename(model_dir)
print('Processing ' + model + '...')

# Get run information.
runs = pd.read_csv(runs_file)
runs = runs[runs.model == model]
runs = runs[runs.epoch_name == epoch]
assert len(runs) == 1, 'Run not properly defined.'

# Load all historical and future data for this model. Note use of h5netcdf because of 
# instabilities in netCDF4 in analysis3-24.04.
hist = xarray.open_mfdataset(f'{model_dir}/historical/*.nc', parallel=True, engine='h5netcdf')
fut = xarray.open_mfdataset(f'{model_dir}/ssp585/*.nc', parallel=True, engine='h5netcdf')

# Chunk for efficiency.
hist = hist.chunk({'time': -1, 'lat': 20, 'lon': 20})
fut = fut.chunk({'time': -1, 'lat': 20, 'lon': 20})

# Subset to 20 year epoch for future run and check length matches historical run.
fut = fut.sel(time=slice(f'{runs.start_year.values[0]}', f'{runs.end_year.values[0]}'))
assert fut.sizes == hist.sizes

# Control is the future run with present-day times, used to calculate the bias in the future run.
ctrl = fut.copy(deep=True)
assert len(ctrl.time) == len(hist.time), 'Times mismatch'
ctrl['time'] = hist.time

# Deal with each variable in turn.
for variable in [x for x in hist.data_vars if x not in exclude_vars]:
    print(variable)

    detrended_file = f'{out_dir}/{model}.{variable}.detrended_{epoch}.nc'
    if not os.path.exists(detrended_file):
        unb = adjust(method='quantile_mapping',    
                     obs=hist[variable],
                     simh=ctrl[variable],
                     simp=fut[variable],
                     n_quantiles=100,
                     kind='+').load()
        comp = dict(zlib=True, shuffle=True, complevel=4)
        encoding = {var: comp for var in unb.data_vars}
        unb.to_netcdf(detrended_file, encoding=encoding)

