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

# Command line arguments.
assert len(sys.argv) == 5, 'Usage: post_process_CMIP.py <epoch_name> <model_name> <start_year> <end_year>'
epoch_name = sys.argv[1]
model_name = sys.argv[2]
epoch_start = int(sys.argv[3])
epoch_end = int(sys.argv[4])

# Skip if outfile exists.
out_file = f'{outdir}/{model_name}_{epoch_name}_{epoch_start}-{epoch_end}_native_grid.nc'
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
    stats = fh.process_epoch(epoch_name=epoch_name,
                             model_name=model_name,
                             epoch_dates=(epoch_start, epoch_end))
    
    stats = stats.sortby('year')
    stats['year_num'] = ('year', np.arange(1, stats.year.size+1))
    stats = stats.swap_dims({'year': 'year_num'}).reset_coords()

    comp = dict(zlib=True, shuffle=True, complevel=4)
    encoding = {var: comp for var in stats.data_vars}
    stats.to_netcdf(out_file, encoding=encoding)            