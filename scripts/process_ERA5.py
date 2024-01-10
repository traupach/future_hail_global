import os
import sys
import dask
import warnings
import xarray
import numpy as np
import fut_hail as fh
from glob import glob
import modules.parcel_functions as parcel
import modules.hail_sounding_functions as hs
from dask.distributed import Client, Scheduler, LocalCluster

# Settings.
warnings.filterwarnings("ignore", category=FutureWarning)                    # Ignore FutureWarnings (in Dask). 
outdir = '/g/data/up6/tr2908/future_hail_global/era5_conv/'                  # Processing output directory.
proxy_results_file = '../../aus400_hail/results/results_era5.json'           # Trained proxy definition file. 
proxy_conds_file = '../../aus400_hail/results/era5_proxy_extra_conds.csv'    # Extra proxy conditions file.
ERA5_dir = '/g/data/up6/tr2908/future_hail_global/era5_1deg/'                # ERA5 data directory.
height_file = '/g/data/up6/tr2908/future_hail_global/era5_1deg/era5_surface_height.nc' # File with geopotential heights of surface.
attrs = {'history': 'ERA5 data interpolated onto 1 degree grid, then convective indices and hail proxy calculated.'}

# Command line arguments.
assert len(sys.argv) == 3, 'Usage: process_ERA5.py <process_num> <total_processes>.'
num = int(sys.argv[1])
total = int(sys.argv[2])

# Start a dask cluster.
print('Starting cluster...')
client = Client(scheduler_file='sched_' + os.environ['PBS_JOBID'] + '.json')
print(client, flush=True)

# Allow large dask chunks.
_ = dask.config.set(**{'array.slicing.split_large_chunks': False})

# Open surface heights. To generate the surface height file:
# fh.regrid_global(path='/g/data/rt52/era5/single-levels/reanalysis/z/1979/z_era5_oper_sfc_19790101-19790131.nc', 
#                  out_res=1, rename={'lon': 'longitude', 'lat': 'latitude'}, isel=dict(time=0),
#                  output_file=lambda x: '/g/data/up6/tr2908/future_hail_global/era5_1deg/era5_surface_height.nc')
surface_height = xarray.open_dataset(height_file).z/9.81
surface_height = surface_height.load()

# Determine which files to process.
files = sorted(glob(ERA5_dir + '/era5_1deg*.nc'))
files_per_process = int(np.ceil(len(files) / total))
files_from = (num-1) * files_per_process
files_to = num * files_per_process
if files_from >= len(files):
    print('No files left to process.')
    exit()
if files_to > len(files) - files_per_process:
    files_to = len(files)
print(f'Processing files from index {files_from} to index {files_to-1}.')
files = files[files_from:files_to]

# Process ERA5 files.
for file in files:
    out_file = f'{outdir}/{os.path.basename(file).replace(".nc", "_conv.nc")}'
    if os.path.exists(out_file):
            print(f'Skipping existing output {out_file}.')
            continue
    
    print(f'Reading {file}...')
    era5 = xarray.open_dataset(file)
    
    _, era5['pressure'] = xarray.broadcast(era5.temperature, era5.level)
    era5 = era5.assign_coords({'level': np.arange(len(era5.level), 0, step=-1)})
    era5 = era5.sortby(['level', 'latitude', 'longitude'])
    era5.pressure.attrs['units'] = 'hPa'
    
    # Convert to a noleap calendar.
    era5 = era5.convert_calendar('noleap')

    # Process each day in turn and write output for each day.
    from_date = era5.time.min().dt.floor('1D').item()
    to_date = era5.time.max().dt.floor('1D').item()
    days = xarray.cftime_range(from_date, to_date, freq='D', calendar='noleap')

    conv = []
    for day in days:
        print(day)
        day_string = f'{day.year}-{day.month:02}-{day.day:02}'
        day_dat = era5.sel(time=day_string).load()

        print('Calculating heights and subsetting to above surface...')
        day_dat['height_asl'] = day_dat.z/9.81
        day_dat['height_above_surface'] = day_dat.height_asl - surface_height
        day_dat = day_dat.where(day_dat.height_above_surface >= 0)
        day_dat['wind_height_above_surface'] = day_dat.height_above_surface
        day_dat['surface_wind_u'] = era5.u.isel(level=0)
        day_dat['surface_wind_v'] = era5.v.isel(level=0)
        day_dat = day_dat.rename({'u': 'wind_u', 'v': 'wind_v'})
        
        # Shift out NaNs.
        day_dat = parcel.shift_out_nans(x=day_dat, name='pressure', dim='level')

        # Chunk for efficiency.
        day_dat = day_dat.chunk({'time': 1, 'level': -1, 'latitude': -1, 'longitude': -1})
        
        # Calculate hail-proxy indices.
        print('Calculating convective properties...')
        res = parcel.min_conv_properties(dat=day_dat, vert_dim='level')
    
        print('Finding ERA5 physical mask...')
        mask = day_dat.specific_humidity.min('level') > 0
        mask.name = 'physical_mask'
        mask.attrs['long_name'] = 'Physical mask for ERA5 data.'
        mask.attrs['description'] = ('Indicates 0 where data is invalid because there' + 
                                     ' were negative values of q in the column.')
        res = xarray.merge([res, mask])
    
        # Add hail proxies.
        res['hail_proxy'] = hs.apply_trained_proxy(dat=res, results_file=proxy_results_file, 
                                                   extra_conds_file=proxy_conds_file)
        res['hail_proxy_noconds'] = hs.apply_trained_proxy(dat=res, results_file=proxy_results_file, 
                                                           extra_conds_file=None)
        
        conv.append(res.load())
        del res, day_dat

    fh.write_output(xarray.merge(conv), attrs=attrs, file=out_file)
    era5.close()
    del era5, conv