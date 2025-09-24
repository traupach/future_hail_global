"""Do convective calculations for a CMIP6 dataset."""

import os
import sys
import warnings

import dask
import fut_hail as fh
from dask.distributed import Client

# Settings.
warnings.filterwarnings("ignore", category=FutureWarning)                    # Ignore FutureWarnings (in Dask).
outdir = '/g/data/up6/tr2908/future_hail_global/CMIP_conv/'                  # Processing output directory.
proxy_results_file = '../../../aus400_hail/results/results_era5.json'        # Trained proxy definition file.
proxy_conds_file = '../../../aus400_hail/results/era5_proxy_extra_conds.csv' # Extra proxy conditions file.
days_per_outfile = 20                                                        # Days per output file.
CMIP6_dir = '/g/data/oi10/replicas'                                          # CMIP6 data directory.

# Command line arguments.
assert len(sys.argv) == 3, 'Usage: process_CMIP.py <model> <year>.'  # noqa: PLR2004
model = sys.argv[1]
year = int(sys.argv[2])

# Start a dask cluster.
print('Starting cluster...')
client = Client(scheduler_file='sched_' + os.environ['PBS_JOBID'] + '.json')
print(client, flush=True)

# Allow large dask chunks.
_ = dask.config.set(**{'array.slicing.split_large_chunks': False})

# Read in all CMIP6 data.
print('Reading data', flush=True)
dat = fh.read_all_CMIP_data(model=model, CMIP6_dir=CMIP6_dir)
print('Done reading data', flush=True)

# Process for the given year.
fh.conv_CMIP(dat=dat, year=year, outdir=f'{outdir}', days_per_outfile=30)
