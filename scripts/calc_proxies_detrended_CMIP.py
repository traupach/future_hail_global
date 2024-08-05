import os
import sys
import xarray
from glob import glob
import modules.fut_hail as fh
import modules.updated_proxy as up
from dask.distributed import Client

base_dir = '/g/data/up6/tr2908/future_hail_global/CMIP_conv/' # Base directory under which to find CMIP6 files.
out_dir = '/g/data/up6/tr2908/future_hail_global/CMIP_detrended/proxy_results' # Output directory for cached outputs.
detrended_dir = '/g/data/up6/tr2908/future_hail_global/CMIP_detrended/ingredients' # Directory for detrended ingredients.
proxy_results_file = '../../aus400_hail/results/results_era5.json' # Trained proxy definition file. 
proxy_conds_file = '../../aus400_hail/results/era5_proxy_extra_conds.csv' # Extra conditions proxy file.
detrended_ing = ['freezing_level', 'lapse_rate_700_500', 'melting_level', 'mixed_100_cape', 
                 'mixed_100_cin', 'mixed_100_dci', 'mixed_100_lifted_index', 'mu_cape', 
                 'mu_cin', 'mu_mixing_ratio', 'shear_magnitude', 'temp_500']

# Get model number to process from the command line.
assert len(sys.argv) == 2, 'Usage: calc_proxies.py <model_number>'
model_num = int(sys.argv[1])

# Start a dask cluster.
print('Starting cluster...')
client = Client(scheduler_file='sched_' + os.environ['PBS_JOBID'] + '.json')
print(client, flush=True)

# Get model directory.
model_dirs = sorted(glob(f'{base_dir}/*/'))
model_dir = model_dirs[model_num]
subdir = glob(f'{model_dir}ssp585')[0] # Future epoch to process.

dirs = subdir.split('/')
epoch = dirs.pop()
model = dirs.pop()
outfile = f'{out_dir}/{model}_{epoch}_proxies_detrended_'
print(outfile)

# Open input data.
dat = xarray.open_mfdataset(f'{subdir}/*.nc', parallel=True)
dat = dat.chunk({'time': 3000, 'lat': -1, 'lon': -1})

for ing in detrended_ing:
    outfile_ing = f'{outfile}{ing}.nc'
    
    if not os.path.exists(outfile_ing):
        print(f'Calculating proxies with detrended {ing}.')
        detrended_file = glob(f'{detrended_dir}/{model}.{ing}.detrended*.nc')
        assert len(detrended_file) == 1, 'Error finding detrended ingredient.'
        detrended_file = detrended_file[0]
    
        # Open detrended data.
        detrended_ingredient = xarray.open_dataset(detrended_file)[ing]
        detrended_ingredient = detrended_ingredient.load().chunk({'time': 3000, 'lat': -1, 'lon': -1})
    
        # Align times.
        dat = dat.sel(time=detrended_ingredient.time)
        assert dat.time.equals(detrended_ingredient.time), 'Time mismatch.'
        
        # Replace detrended ingredient.
        detrended = dat.copy()
        detrended[ing] = detrended_ingredient
    
        # Calculate literature cape-shear proxies.
        print('Literature proxies...')
        prox = fh.storm_proxies(dat=detrended, proxies={'proxy_Eccel2012': 'Eccel 2012',
                                                        'proxy_SHIP_0.1': 'SHIP > 0.1'})

        # Add Raupach hail proxies.
        print('Raupach proxies...')
        prox['proxy_Raupach2023_updated'] = up.apply_Raupach_proxy(dat=detrended,
                                                                   results_file=proxy_results_file, 
                                                                   extra_conds_file=proxy_conds_file,
                                                                   band_limits=[2000, None], load=False)
        prox['proxy_Raupach2023_updated_noconds'] = up.apply_Raupach_proxy(dat=detrended,
                                                                           results_file=proxy_results_file, 
                                                                           extra_conds_file=None,
                                                                           band_limits=[2000, None], load=False)
    
        prox.attrs['note'] = f'Proxies calculated for 3C epoch with ingredient {ing} detrended to match historical period.'
        fh.write_output(dat=prox, file=outfile_ing)
    
        del detrended, detrended_ingredient, prox