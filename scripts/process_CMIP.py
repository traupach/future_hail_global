import os
import sys
import dask
import warnings
import numpy as np
import fut_hail as fh
import modules.parcel_functions as parcel
from dask.distributed import Client, Scheduler, LocalCluster

# Settings.
warnings.filterwarnings("ignore", category=FutureWarning)                    # Ignore FutureWarnings (in Dask). 
outdir = '/g/data/up6/tr2908/future_hail_global/CMIP_conv/'                  # Processing output directory.
proxy_results_file = '../../../aus400_hail/results/results_era5.json'        # Trained proxy definition file. 
proxy_conds_file = '../../../aus400_hail/results/era5_proxy_extra_conds.csv' # Extra proxy conditions file.
days_per_outfile = 20                                                        # Days per output file.

# Command line arguments.
assert len(sys.argv) == 3, 'Usage: process_CMIP.py <model> <year>.'
model = sys.argv[1]
year = int(sys.argv[2])

# Start a dask cluster.
print('Starting cluster...')
client = Client(scheduler_file='sched_' + os.environ['PBS_JOBID'] + '.json')
print(client)

# Allow large dask chunks.
_ = dask.config.set(**{'array.slicing.split_large_chunks': False})

# Read in all CMIP6 data.
dat = fh.read_all_CMIP_data(model=model)

# Process for the given year.
fh.conv_CMIP(dat=dat, year=year, proxy_results_file=proxy_results_file, 
             proxy_conds_file=proxy_conds_file, outdir=f'{outdir}', 
             days_per_outfile=30)
