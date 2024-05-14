import os
import sys
import json
import xarray
import numpy as np
import pandas as pd
from glob import glob
import modules.fut_hail as fh
import modules.updated_proxy as up

base_dir = '/g/data/up6/tr2908/future_hail_global/era5_conv/' # Base directory under which to find ERA5 conv files.
outfile = '/g/data/up6/tr2908/future_hail_global/era5_conv/era5_proxies.nc' # Output file.
proxy_results_file = '../../aus400_hail/results/results_era5.json' # Trained proxy definition file. 
proxy_conds_file = '../../aus400_hail/results/era5_proxy_extra_conds.csv' # Extra conditions proxy file.

if not os.path.exists(outfile):
    dat = xarray.open_mfdataset(f'{base_dir}/*conv*.nc', parallel=True)
    dat = dat.chunk({'time': 3000, 'latitude': -1, 'longitude': -1})

    # Calculate literature cape-shear proxies.
    print('Literature proxies...')
    prox = fh.storm_proxies(dat=dat)

    # Add Raupach hail proxies.
    print('Raupach proxies...')
    prox['proxy_Raupach2023_original'] = up.apply_Raupach_proxy(dat=dat, results_file=proxy_results_file, 
                                                                extra_conds_file=proxy_conds_file, load=False, 
                                                                band_limits=None)
    prox['proxy_Raupach2023_original_noconds'] = up.apply_Raupach_proxy(dat=dat, results_file=proxy_results_file, 
                                                                        extra_conds_file=None, load=False,
                                                                        band_limits=None)
    prox['proxy_Raupach2023_updated'] = up.apply_Raupach_proxy(dat=dat, results_file=proxy_results_file, 
                                                               extra_conds_file=proxy_conds_file,
                                                               band_limits=[2000, None], load=False)
    prox['proxy_Raupach2023_updated_noconds'] = up.apply_Raupach_proxy(dat=dat, results_file=proxy_results_file, 
                                                                       extra_conds_file=None,
                                                                       band_limits=[2000, None], load=False)

    fh.write_output(dat=prox, file=outfile)
    del prox, dat
else:
    print(f'Skipping because output file {outfile} exists.')
