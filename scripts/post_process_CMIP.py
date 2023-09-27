import os
import sys
import dask
import warnings
import xarray
import numpy as np
import fut_hail as fh
from dask.distributed import Client, Scheduler, LocalCluster

# Settings.
warnings.filterwarnings("ignore", category=FutureWarning)               # Ignore FutureWarnings (in Dask). 
outdir = '/g/data/up6/tr2908/future_hail_global/CMIP_conv_annual_stats' # Processing output directory.
out_res = 1                                                             # Output resolution (degrees).

# [epoch_name, dir_postfix, model_name, epoch]
epochs = [
    ['historical', '-historical', 'MRI-ESM2-0',                    (1980, 2000)],
    ['historical', '-historical', 'MPI-ESM1-2-HR',                 (1980, 2000)],
    ['historical', '-historical', 'CMCC-ESM2',                     (1980, 2000)],
    ['historical', '-historical', 'EC-Earth_Consortium.EC-Earth3', (1980, 2000)],
    ['2C',         '-SSP585',     'MRI-ESM2-0',                    (2025, 2045)],
    ['2C',         '-SSP585',     'MPI-ESM1-2-HR',                 (2029, 2049)],
    ['2C',         '-SSP585',     'CMCC-ESM2',                     (2022, 2042)],
    ['2C',         '-SSP585',     'EC-Earth_Consortium.EC-Earth3', (2018, 2038)],
    ['3C',         '-SSP585',     'MRI-ESM2-0',                    (2051, 2071)],
    ['3C',         '-SSP585',     'MPI-ESM1-2-HR',                 (2056, 2076)],
    ['3C',         '-SSP585',     'CMCC-ESM2',                     (2040, 2060)],
    ['3C',         '-SSP585',     'EC-Earth_Consortium.EC-Earth3', (2039, 2059)],
]

# Command line arguments.
assert len(sys.argv) == 2, 'Usage: post_process_CMIP.py <epoch_number>.'
epoch_number = int(sys.argv[1])

# Skip if outfile exists.
epoch = epochs[epoch_number]
out_file = f'{outdir}/{epoch[2]}_{epoch[0]}_{epoch[3][0]}-{epoch[3][1]}_native_grid.nc'
if os.path.exists(out_file):
    print(f'Skipping existing {out_file}.')
else:
    # Start a dask cluster.
    print('Starting cluster...')
    client = Client(scheduler_file='sched_' + os.environ['PBS_JOBID'] + '.json')
    print(client)

    # Allow large dask chunks.
    _ = dask.config.set(**{'array.slicing.split_large_chunks': False})

    # Process epoch.
    stats = fh.process_epoch(epoch_name=epoch[0],
                             dir_postfix=epoch[1],
                             model_name=epoch[2],
                             epoch_dates=epoch[3])
    
    stats = stats.sortby('year')
    stats['year_num'] = ('year', np.arange(1, stats.year.size+1))
    stats = stats.swap_dims({'year': 'year_num'}).reset_coords()

    comp = dict(zlib=True, shuffle=True, complevel=4)
    encoding = {var: comp for var in stats.data_vars}
    stats.to_netcdf(out_file, encoding=encoding)            