import os
import re
import sys
import dask
import xarray
import intake
import cftime
import itertools
import geopandas
import scipy as sp
import xesmf as xe
import numpy as np
import pandas as pd
from glob import glob
from matplotlib import cm
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
from cartopy.io import shapereader
import matplotlib.ticker as mticker
import modules.warming_levels as wl
import modules.parcel_functions as parcel
import modules.hail_sounding_functions as hs
from matplotlib.colors import ListedColormap, LinearSegmentedColormap

# Settings for xarray_parcel: set up parcel adiabat calculations.
lookup_dir = '/g/data/w42/tr2908/aus400_hail/'
parcel.load_moist_adiabat_lookups(base_dir=lookup_dir, chunks=-1)

# Pretty colors for hail maps.
cmap_colours = [
    [0.9, 0.9, 0.9, 1], 
    [0.92941176, 0.97254902, 1.        , 1.        ],
    [0.85882353, 0.90980392, 0.98823529, 1.        ],
    [0.85490196, 0.90980392, 0.99215686, 1.        ],
    [0.57254902, 0.75686275, 0.98039216, 1.        ],
    [0.51372549, 0.66666667, 0.97254902, 1.        ],
    [0.4745098 , 0.60784314, 0.96862745, 1.        ],
    [0.39607843, 0.54117647, 0.96862745, 1.        ],
    [0.29411765, 0.66666667, 0.21960784, 1.        ],
    [0.35686275, 0.76470588, 0.34509804, 1.        ],
    [0.74117647, 0.83921569, 0.5254902 , 1.        ],
    [0.7372549 , 0.96470588, 0.53333333, 1.        ],
    [0.99607843, 0.96862745, 0.32156863, 1.        ],
    [0.95294118, 0.69019608, 0.24313725, 1.        ],
    [0.91372549, 0.2       , 0.14117647, 1.        ],
    [0.86666667, 0.18823529, 0.13333333, 1.        ],
    [0.63529412, 0.12941176, 0.08235294, 1.        ]]
hail_cmap = LinearSegmentedColormap.from_list("", cmap_colours)

@dask.delayed
def open_CMIP_file_delayed(filename, chunks={'time': 20}):
    """
    Open a CMIP file using dask delayed so it is only accessed when requested.
    
    Arguments:
        filename: The filename to open.
        chunks: Chunks to use.
    """

    return xarray.open_dataset(filename, use_cftime=True).chunk(chunks)

def open_CMIP_file(filename, res, ex, calendar, chunks={'time': 20}, **kwargs):
    """
    Open a CMIP file using dask delayed so it is only accessed when requested.
    
    Arguments:
        filename: The filename to read.
        res: Resolution to use.
        ex: An example version of the variable to get coordinate info from.
        calendar: Datetime calendar to use.
        chunks: Data chunks to use.
        **kgwargs: Extra arguments to open_CMIP_file_delayed.
    """
    
    dat = open_CMIP_file_delayed(filename=filename, chunks=chunks, **kwargs)[ex.name].data
    times = times_in_CMIP_file(filename=filename, res=res, calendar=calendar)
    shape = (times.size,) + ex.shape[1:]
    return dask.array.from_delayed(value=dat, shape=shape, dtype=ex.dtype)
    
def open_CMIP(path, res, variable, chunks={'time': 25}, max_year=2100):
    """
    Open all CMIP files in a given path, using dask delayed for speed.
    
    Arguments:
        path: The path containing nc files.
        res: Temporal resolution.
        variable: The variable to open.
        chunks: Chunking for result.
        max_year: Open files with years up to this year.
    """
    
    # List files to open.
    all_files = sorted(glob(f'{path}/*.nc'))
    
    # Use the variable from the first file to get coordinate information.
    first = xarray.open_dataset(all_files[0], use_cftime=True).chunk(chunks)
    ex = first[variable]

    # Keep files up to max year.
    files = []
    for file in all_files:
        times = times_in_CMIP_file(file, res=res, calendar=first.time.dt.calendar)
        if times.min().year < max_year:
            files.append(file)
    
    # Get the time range using filename information.
    times_first = times_in_CMIP_file(files[0], res=res, calendar=first.time.dt.calendar)
    times_last = times_in_CMIP_file(files[-1], res=res, calendar=first.time.dt.calendar)
    times = xarray.cftime_range(times_first[0], times_last[-1], freq=res, inclusive='both',
                                calendar=first.time.dt.calendar)
    
    # Check that times line up.
    assert times[0] == first.time[0], 'First time does not match.'
    assert times[1] == first.time[1], 'Temporal resolution is not correct.'
    
    # Concatenate all the delayed-opening objects for all files.
    dat = dask.array.concatenate(
        [open_CMIP_file(filename=f, res=res, ex=ex, calendar=first.time.dt.calendar,
                       chunks=chunks) for f in files], axis=0)
    
    # Convert to an xarray object.
    coords = {x: ex[x] for x in ex.dims}
    coords['time'] = times
    ret = xarray.DataArray(dat, name=ex.name, attrs=ex.attrs, dims=ex.dims, coords=coords)
    ret = ret.chunk(chunks)
    
    # Clean up.
    ex.close()
    return ret

def times_in_CMIP_file(filename, res, calendar):
    """
    Use a CMIP filename to return all the timesteps the file contains.
    
    Arguments:
        file: The CMIP filename, must end in YYYYMMDDHHMM-YYYYMMDDHHMM.nc.
        res: Temporal resolution for the file.
        calendar: The calendar to use.
        
    Returns: An array of cftimes for all timesteps in the file.
    """
    
    basename = os.path.basename(filename)
    time_range = [re.search(r'.*_([0-9]{12})-[0-9]{12}\.nc', basename).group(1),
                  re.search(r'.*_[0-9]{12}-([0-9]{12})\.nc', basename).group(1)]
    time_range = [f'{x[0:4]}-{x[4:6]}-{x[6:8]} {x[8:10]}:{x[10:12]}' for x in time_range]
    times = xarray.cftime_range(time_range[0], time_range[1], freq=res, inclusive='both',
                                calendar=calendar)
    return times

def geopotential_height(dat, vert_dim='lev', Rd=287.04749, g=9.80665): 
    """
    Calculate geopotential height for all levels in the data.
    
    Arguments:
        dat: Data containing orog, surface_temperature, surface_pressure, surface_specific_humidity,
             temperature, pressure, specific_humidity, levels.
        vert_dim: The name of the vertical dimension.
        chunks: Chunks to apply to intermediate objects.
        
    Returns: A DataArray of geopotential heights (m), and a DataArray of heights above the surface.
    """
    
    assert np.all(dat[vert_dim].diff(vert_dim) == 1), 'Vert dim should be an integer increasing index.'
    
    # Calculate virtual temperature.
    virt_temp = (dat.temperature * (1 + 0.608 * dat.specific_humidity))
    vt_surf = dat.surface_temperature * (1 + 0.608 * dat.surface_specific_humidity)

    # Virtual temperature at bottom/top of each layer.
    vt_from = xarray.concat([vt_surf.expand_dims({vert_dim: [0]}), 
                             virt_temp.sel({vert_dim:slice(1, dat[vert_dim].max()-1)})], dim=vert_dim)
    vt_from = vt_from.assign_coords({vert_dim: vt_from[vert_dim] + 1})
    vt_to = virt_temp
    vt_mean = (vt_from + vt_to) / 2

    # Pressure at the bottom/top of each layer.
    p_from = xarray.concat([dat.surface_pressure.expand_dims({vert_dim: [0]}), 
                            dat.pressure.sel({vert_dim:slice(1, dat[vert_dim].max()-1)})], dim=vert_dim)
    p_from = p_from.assign_coords({vert_dim: p_from[vert_dim] + 1})
    p_to = dat.pressure
    p_div = np.log(p_from/p_to)

    # Calculate thickness of each layer.
    thickness = Rd/g * vt_mean * p_div
    level_heights = xarray.concat([dat.orog.expand_dims({vert_dim: [0]}), thickness], dim=vert_dim)
    
    # Do a cumsum to get the height of each layer.
    level_heights = level_heights.cumsum(vert_dim).sel({vert_dim: slice(1,None)})
    level_heights.attrs['long_name'] = 'Geopotential height'
    level_heights.attrs['units'] = 'm'
    
    # Calculate above-surface height.
    above_surface = level_heights - dat.orog
    above_surface.attrs['long_name'] = 'Height above surface'
    above_surface.attrs['units'] = 'm'
    
    return level_heights, above_surface

def read_all_CMIP_data(model, CMIP6_dir='/g/data/oi10/replicas', orog_path='fx/orog/',
                       backup_orog_path='/g/data/up6/tr2908/future_hail_global/interpolated_orography/',
                       max_year=2100, pressure_var='ta', out_res='6H', chunks={'time': 20}, grid=None,
                       tables=[['6hrLev', '6H'], ['3hr', '3H']],
                       variables=['va', 'ua', 'ta', 'hus', 'ps', 'vas', 'uas', 'huss', 'tas']):
    """
    Read all CMIP6 data for a given model, at 6H resolution. 
    
    Arguments:
        model: Model spec (e.g. CMIP6.ScenarioMIP.MRI.MRI-ESM2-0.ssp585.r1i1p1f1)
        CMIP6_dir: Directory for CMIP6 data.
        orog_path: Path to orography.
        backup_orog_path: Path to directory containing interpolated orography files for models 
                          missing orography.
        max_year: Read in data to this year only.
        pressure_var: The variable to use for pressure definition.
        out_res: Output resolution.
        chunks: Chunks to use for opening datasets.
        grid: Grid to use.
        tables: [path, res] groups to search for variables in, in order of preference.
        variables: variables to read.
        
    Returns: An xarray object with all CMIP6 data for the given model at the correct resolution.
    """
    
    model_dir = model.replace('.', '/')
    base_path = f'{CMIP6_dir}/{model_dir}'
    dat = xarray.Dataset()
    
    # Parse model string.
    project, mip, source, model_name, exp, ensemble = model.split('.')

    for v in variables:
        for [p, res] in tables:
            path = f'{base_path}/{p}/{v}'
            if os.path.exists(path):
                break
        assert os.path.exists(path), f'Could not find path/res combinations for {v}.'
        
        if grid is None:
            grids = [os.path.basename(x) for x in sorted(glob(f'{path}/*'))]
            assert len(grids) == 1, f'Multiple grids to choose from for {path}/{v}.'
            grid = grids[0]
            
        path = f'{path}/{grid}'
        version = [os.path.basename(x) for x in sorted(glob(f'{path}/v*'))][-1] # Use latest version.
        path = f'{path}/{version}'

        print(path, flush=True)
        d = open_CMIP(path=path, res=res, variable=v, chunks=chunks)
        d = d.sel(time=slice(None, f'{max_year-1}-12-31'))

        # Keep data only every 6H.
        if res != out_res:
            assert 'time' in dat, f'Must include data natively at {out_res} before data that requires resampling.'
            keep_times = xarray.cftime_range(dat.time.values[0], dat.time.values[-1], freq=out_res, 
                                             inclusive='both', calendar=d.time.dt.calendar)
            keep_times = keep_times[keep_times >= d.time.values[0]]
            keep_times = keep_times[keep_times <= d.time.values[-1]]
            d = d.sel(time=keep_times)

        # Add metadata.
        d.attrs['CMIP_model_spec'] = model
        d.attrs['CMIP_model_name'] = model_name
        d.attrs['CMIP_model_version'] = version
        d.attrs['CMIP_grid_spec'] = grid
        d.attrs['processing_note'] = f'Data subset to {out_res} temporal resolution and maximum year {max_year}.'
            
        # Chunk.
        dat[v] = d.chunk(chunks)
        
        # Check all grid point match.
        if 'lat' in dat.coords:
            assert np.all(dat.lat == d.lat), 'Latitudes for {v} do not match other variables.'
            assert np.all(dat.lon == d.lon), 'Longitudes for {v} do not match other variables.'
            if 'lev' in d.coords:
                assert np.all(dat.lev == d.lev), 'Levels for {v} do not match other variables.'
        
        # Process pressure as required.
        if v == pressure_var:
            ex = xarray.open_dataset(glob(f'{path}/*.nc')[0]).chunk(chunks)
            
            # Pressure is defined as p = a*p0 + b*ps.
            if 'p0' in ex:
                # Some modules use p = a*p0 + b*ps.
                dat['a'] = ex.a
                dat['p0'] = ex.p0
                assert dat.lev.attrs['formula'] == 'p = a*p0 + b*ps', 'Pressure formula problem.'
                
            else:
                # Some modules use ap, in which a is already multiplied by p.
                dat['a'] = ex.ap
                dat['p0'] = xarray.ones_like(dat.a)
                assert dat.lev.attrs['formula'] == 'p = ap + b*ps', 'Pressure formula problem.'
            
            dat['b'] = ex.b
            del ex

    # Rename variables.
    dat = dat.rename({'vas': 'surface_wind_v',
                      'uas': 'surface_wind_u',
                      'va': 'wind_v',
                      'ua': 'wind_u',
                      'ta': 'temperature',
                      'tas': 'surface_temperature',
                      'hus': 'specific_humidity',
                      'huss': 'surface_specific_humidity',
                      'ps': 'surface_pressure'})

    # Read orography.
    orog_p = f'{base_path}/{orog_path}/{grids[0]}/'
    if os.path.exists(orog_p):
        # Use model orography if it is provided.
        orog_version = [os.path.basename(x) for x in sorted(glob(f'{orog_p}/*'))][0]
        orog = xarray.open_mfdataset(f'{orog_p}/{orog_version}/*.nc', use_cftime=True)['orog']
        
        # If orography coordinates differ slightly from other-data coordinates, use main data coordinates.
        if not np.all(dat.lat.values == orog.lat.values):
            assert np.max(np.abs(dat.lat.values - orog.lat.values)) < 1e-5, 'Mismatch in orography latitudes.'
            orog = orog.assign_coords({'lat': dat.lat.values})
        if not np.all(dat.lon.values == orog.lon.values):
            assert np.max(np.abs(dat.lon.values - orog.lon.values)) < 1e-5, 'Mismatch in orography longitudes.'
            orog = orog.assign_coords({'lon': dat.lon.values})

        dat['orog'] = orog
        dat.orog.attrs['CMIP_model_spec'] = model
        dat.orog.attrs['CMIP_model_name'] = model_name
        dat.orog.attrs['CMIP_model_version'] = orog_version
        dat.orog.attrs['CMIP_grid_spec'] = grids[0]
    else:
        # Otherwise use the backup orography.
        print('Using backup orography...', flush=True)
        orog_file = f'{backup_orog_path}/orog_{model_name}.{exp}.{ensemble}.nc'
        assert os.path.exists(orog_file), 'Interpolated backup orography is missing.'
        orog = xarray.open_mfdataset(orog_file).orog
        assert np.all(orog.lat == dat.lat), 'Coordinate error with backup orog.'
        assert np.all(orog.lon == dat.lon), 'Coordinate error with backup orog.'
        dat['orog'] = orog

    # Some datasets use NaN in the orography for ocean; set NaNs to zero.
    dat['orog'] = dat.orog.where(~np.isnan(dat.orog), other=0)
    
    # Add pressure at each level.
    dat['pressure'] = (dat.a * dat.p0 + dat.b * dat.surface_pressure) / 100
    dat.pressure.attrs['long_name'] = 'Atmospheric pressure'
    dat.pressure.attrs['units'] = 'hPa'

    # Convert surface pressure from Pa to hPa.
    with xarray.set_options(keep_attrs=True):
        dat['surface_pressure'] = dat.surface_pressure / 100
        dat.surface_pressure.attrs['units'] = 'hPa'

    # Replace sigma coordinates with integer level numbers.
    dat = dat.assign_coords({'lev': np.arange(1, dat.lev.size+1)})

    return dat

def write_output(dat, file, attrs):
    """
    Write data in 'dat', with attributes 'attrs', to 'file' as NetCDF with compression.
    """
    
    comp = dict(zlib=True, shuffle=True, complevel=4)
    encoding = {var: comp for var in dat.data_vars}
    dat.attrs = attrs
    dat.to_netcdf(file, encoding=encoding)
    
def regrid_global(path, out_res=1, rename=None, output_file=lambda x: x.replace('native_grid', 'common_grid'), isel=None):
    """
    Regrid all (global) files in a directory to a common grid.
    
    Arguments:
        path: Path to match files to regrid.
        out_res: Output resolution in degrees.
        rename: Rename variables before saving?
        output_file: Function to convert input filename to output filename.
    """
    
    files = glob(path)

    out_lon = np.arange(-180.0, 180.0 + out_res, out_res)
    out_lat = np.arange(-90, 90 + out_res, out_res)
    out_grid = xarray.Dataset(coords={'lat': (out_lat[:-1] + out_lat[1:]) / 2,
                                      'lon': (out_lon[:-1] + out_lon[1:]) / 2, 
                                      'lat_b': out_lat,
                                      'lon_b': out_lon})

    for file in files:
        outfile = output_file(file)
        if not os.path.exists(outfile):
            print(f'Regridding {file}.')
            d = xarray.open_dataset(file)
            regridder = xe.Regridder(d, out_grid, 'bilinear', periodic=True)
            d = regridder(d, keep_attrs=True)
            attrs = {'history': f'Regridded to {out_res} x {out_res} degree grid using xESMF.'}

            if not rename is None:
                d = d.rename(rename)
            if not isel is None:
                d = d.isel(isel)
            
            write_output(dat=d, file=outfile, attrs=attrs)

def conv_CMIP(dat, year, proxy_results_file, proxy_conds_file,
              outdir='/g/data/up6/tr2908/future_hail_global/CMIP_conv/',
              days_per_outfile=30):
    """
    Calculate convective parameters for a given year and output to NetCDF.
    
    Arguments:
        dat: The CMIP6 data to process.
        year: The year to process.
        proxy_results_file: Proxy results file to use with apply_trained_proxy.
        proxy_conds_file: Proxy extra  conditions file.
        outdir: Output directory minus model name.
        days_per_outfile:
    """
    
    _, _, _, model_name, exp, _ = dat.temperature.CMIP_model_spec.split('.')
    outdir = f'{outdir}/{model_name}/{exp}'
    if not os.path.exists(outdir):
        print(f'Output path {outdir} does not exist...')
        return
    
    # Subset to the required year. For purposes of comparing models, some of which 
    # have 29 Febs included and some of which don't, process days using a no-leap-year 
    # calendar.
    from_date = cftime.DatetimeNoLeap(year, 1, 1)
    to_date =  cftime.DatetimeNoLeap(year, 12, 31)
    days_in_year = xarray.cftime_range(from_date, to_date, freq='D', calendar='noleap')

    # Process attributes.
    data_vars = [x for x in dat.data_vars if 'CMIP_model_version' in dat[x].attrs]
    attrs = {'CMIP_version_' + x: dat[x].attrs['CMIP_model_version'] for x in data_vars}
    attrs = {'CMIP_grid_' + x: dat[x].attrs['CMIP_grid_spec'] for x in data_vars}
    attrs['CMIP_model_spec'] = dat.temperature.CMIP_model_spec
    attrs['CMIP_model_name'] = dat.temperature.CMIP_model_name
    attrs['CMIP_note'] = 'Pressure derived from version used for temperature (ta).'
    
    # Process each day in turn.
    i = 1
    conv = []
    
    for j, day in enumerate(days_in_year):
        f_from = (j // days_per_outfile) * days_per_outfile + 1
        f_to = f_from + days_per_outfile - 1
        if f_to > len(days_in_year):
            f_to = len(days_in_year)
        outfile = f'{outdir}/{model_name}_{year}_{f_from:03}-{f_to:03}.nc'
        
        assert day.year == year, 'Year mismatch.'
        
        if os.path.exists(outfile):
            print(f'Skipping output for {outfile}...')
        else:
            # Get this day's data.
            print(f'Processing {day.year}-{day.month:02}-{day.day:02}...')
            day_dat = dat.sel(time=f'{day.year}-{day.month:02}-{day.day:02}')

            # Calculate geopotential heights.
            day_dat['height_asl'], day_dat['wind_height_above_surface'] = geopotential_height(day_dat)

            # Load for speed.
            day_dat = day_dat.load()
            day_dat = day_dat.chunk({'time': 1})

            # Calculate convective parameters.
            c = parcel.min_conv_properties(dat=day_dat, vert_dim='lev').load()

            # Add hail proxies.
            c['hail_proxy'] = hs.apply_trained_proxy(dat=c, results_file=proxy_results_file, 
                                                     extra_conds_file=proxy_conds_file)
            c['hail_proxy_noconds'] = hs.apply_trained_proxy(dat=c, results_file=proxy_results_file, 
                                                             extra_conds_file=None)
            conv.append(c)
        
            print('Cleaning up...')
            del day_dat, c

            if len(conv) >= days_per_outfile:
                write_output(dat=xarray.merge(conv), attrs=attrs, file=outfile)
                del conv
                conv = []
    
    if len(conv) != 0:
        write_output(dat=xarray.merge(conv), attrs=attrs, file=outfile)
        
def plot_map_to_ax(dat, ax, coastlines=True, grid=True, dat_proj=ccrs.PlateCarree(),
                   disp_proj=ccrs.PlateCarree(), title=None, colour_scale=None, cmap=hail_cmap, norm=None,
                   cbar_ticks=None, tick_labels=None, contour=False, stippling=None, stipple_size=3,
                   colourbar=True, ticks_left=True, ticks_bottom=True, cbar_aspect=25, cbar_fraction=0.07,
                   cbar_shrink=0.4, cbar_pad=0.015, cbar_label=None, cbar_orientation='vertical', 
                   coastlines_colour='black', xlims=None, ylims=None, num_ticks=None, divergent=False, 
                   cbar_inset=False, title_inset=False, discrete=False, log_scale=False, nan_colour='#eeeeee',
                   axis_off=False, country=None, annotations=None, num_contours=len(cmap_colours)+1):
    """
    Plot data on a map to a specified plot axis object.
    
    Arguments:
    
        - dat: DataSet to plot or list of datasets to plot.
        - ax: GeoAxes object to plot to.
        - dat_proj, dist_proj: Data and display projections.
        - figsize: Figure size width x height.
        - coastlines: Show coastlines?
        - grid: Show grid?
        - ncol/nrows: Number of columns/rows to plot.
        - title: Title for the plot.
        - colour_scale: None for default, or a tuple of min/max values for the scale.
        - cmap: The matplotlib colour map to use.
        - norm: A norm object for colours (e.g. colors.BoundaryNorm).
        - cbar_ticks: Colour bar ticks.
        - tick_labels: Colour bar tick labels.
        - contour: Plot using xarray's contourf function?
        - stippling: True where trippling should appear.
        - stipple_size: Size for stippling points.
        - colourbar: Include a colourbar?
        - ticks_left: Include ticks on left of plot?
        - ticks_bottom: Include ticks on bottom of plot?
        - cbar_aspect: colorbar aspect ratio?
        - cbar_fraction: fraction argument for colorbar().
        - cbar_shrink: shrink argument for colorbar().
        - cbar_pad: pad argument for colorbar().
        - cbar_label: Overwrite label?
        - cbar_orientation: orientation argument for colorbar().
        - coastlines_colour: Colour for coastlines.
        - xlims, ylims: x and y plot limits.
        - num_ticks: Number of ticks for x and y axes (None for auto).
        - divergent: Is the colour scale divergent? If so make zero central.
        - cbar_inset: Inset the colorbar in lower left?
        - title_inset: Inset the title in the upper left?
        - discrete: Make the colour bar discrete?
        - log_scale: Make the colour scale log-scaled?
        - nan_colour: Colour for missing values.
        - axis_off: Turn off all axes.
        - country: Plot coastlines only for a specific country.
        - annotations: Add annotations to the map - dictionary of {'Text': [x, y, xadj, yadj, ha]} where 
                       x, y are position of text, xadj, yadj give offsets to the label, ha is 'left' or 
                       'right' for horizontal anchor. 
        - num_contours: The number of contours to use in a contour plot.
        
    """
    
    col_min = None
    col_max = None
     
    # Rasterize elements with zorder below 0.
    ax.set_rasterization_zorder(0)
        
    if colour_scale is not None:
        col_min = colour_scale[0]
        col_max = colour_scale[1]
        
    if divergent:
        if col_min is None or col_max is None:
            col_min = dat.min()
            col_max = dat.max()
        
        col_min = -1*np.max(np.abs([col_min, col_max]))
        col_max = np.max(np.abs([col_min, col_max]))
     
    cbar_spacing = 'proportional'
    
    if discrete:
        assert cbar_ticks is not None, 'Discrete colorbar requires cbar_ticks'
        assert cbar_ticks == sorted(cbar_ticks), 'cbar_ticks must be sorted for discrete plot.'
        cbar_ticks = np.array(cbar_ticks + [np.max(cbar_ticks)+1])
        norm = colors.BoundaryNorm(cbar_ticks, ncolors=len(cbar_ticks)-1)
        cbar_ticks = (cbar_ticks[0:-1] + cbar_ticks[1:]) / 2
        cbar_spacing = 'uniform'
        
    if log_scale:
        norm = colors.LogNorm(vmin=col_min, vmax=col_max)
        
    fmt = mticker.ScalarFormatter(useOffset=False, useMathText=True)
    fmt.set_powerlimits((-4, 6))
    cbar_args = {'spacing': cbar_spacing, 
                 'fraction': cbar_fraction,
                 'ticks': cbar_ticks,
                 'aspect': cbar_aspect,
                 'shrink': cbar_shrink,
                 'pad': cbar_pad,
                 'orientation': cbar_orientation,
                 'format': '%g'}
    
    if cbar_inset:
        cax = inset_axes(ax, width='50%', height='3%', loc='lower left',
                         bbox_to_anchor=(0.05, 0.15, 1, 1),
                         bbox_transform=ax.transAxes) 
        cbar_args['cax'] = cax
        
    if cbar_label is not None:
        cbar_args['label'] = cbar_label
                    
    if colourbar == False:
        cbar_args = None
        
    cmap = plt.get_cmap(cmap).copy()
    cmap.set_bad(nan_colour)
        
    if not contour:
        res = dat.plot(ax=ax, transform=dat_proj, vmin=col_min, vmax=col_max, 
                       cmap=cmap, norm=norm, cbar_kwargs=cbar_args,
                       add_colorbar=colourbar, zorder=-10)
    else:
        if col_min is not None:
            col_min = np.floor(col_min)
            col_max = np.ceil(col_max)
            l = np.linspace(col_min, col_max, num_contours, dtype=np.int64)
        else:
            l = num_contours
        
        res = dat.plot.contourf(ax=ax, transform=dat_proj, vmin=col_min, vmax=col_max, 
                                cmap=cmap, norm=norm, cbar_kwargs=cbar_args,
                                add_colorbar=colourbar, levels=l)
    
    if not stippling is None:
        ax.autoscale(False)
        pts = stippling.where(stippling).to_dataframe().dropna().reset_index()
        stippling.plot.contourf(transform=dat_proj, ax=ax, add_colorbar=False,
                    levels=3, colors='none', hatches=[None, '\\\\\\'])
        
    if xlims is not None:
        ax.set_xlim(xlims)
    if ylims is not None:
        ax.set_ylim(ylims)
    if not tick_labels is None:
        assert len(tick_labels) == len(cbar_ticks), 'Labels and ticks must have same length'
        res.colorbar.ax.set_yticklabels(tick_labels)
    if not title is None:
        if title_inset:
            ax.annotate(text=title, xy=(0.05, 0.9), xycoords='axes fraction',
                        fontweight='bold', fontsize=plt.rcParams['font.size'])
        else:
            ax.set_title(title, fontsize=plt.rcParams['font.size'])
    if coastlines:
        if country is not None:
            shpfilename = shapereader.natural_earth(resolution='10m', category='cultural', name='admin_0_countries')
            df = geopandas.read_file(shpfilename)
            poly = df.loc[df['ADMIN'] == country]['geometry'].values[0]
            ax.add_geometries(poly, crs=ccrs.PlateCarree(), facecolor='none', edgecolor=coastlines_colour, linewidth=0.75)
        else:
            ax.coastlines(color=coastlines_colour)
    if grid:
        locator = None
        if num_ticks is not None:
            locator = mticker.MaxNLocator(nbins=num_ticks+1)
        gl = ax.gridlines(crs=disp_proj, draw_labels=True, alpha=0.5, 
                          xlocs=locator, ylocs=locator)
        gl.top_labels = gl.right_labels = False
        gl.left_labels = ticks_left
        gl.bottom_labels = ticks_bottom
    if axis_off:
        ax.axis('off')
        
    if annotations is not None:
        for text, [x, y, xadj, yadj, ha] in annotations.items():
            if np.abs(xadj) >= 1 or np.abs(yadj) >= 1:
                if ha == 'right' or ha == 'left':
                    ax.plot([x, x+xadj-(0.2*np.sign(xadj))], [y, y+yadj+0.2], color='black')
                elif ha == 'center':
                    ax.plot([x, x+xadj-(0.2*np.sign(xadj))], [y, y+yadj-0.2], color='black')
                    if yadj < 0:
                        print('Warning: ha=center and negative y adjustment are not supported.')
                else:
                    assert 1 == 0, 'Invalid value of ha.'
            ax.annotate(xy=(x+xadj, y+yadj), text=text, ha=ha)
        
    return res

def plot_map(dat, dat_proj=ccrs.PlateCarree(), disp_proj=ccrs.PlateCarree(), figsize=(12,8), 
             grid=True, ncols=1, nrows=1, title=None, share_scale=False, 
             colour_scale=None, cbar_ticks=None, tick_labels=None,
             file=None, scale_label='', share_axes=False, ticks_left=True,
             ticks_bottom=True, wspace=0.05, hspace=0.05, stippling=None,
             cbar_adjust=0.862, cbar_pad=0.015, col_labels=None, row_labels=None, 
             xlims=None, ylims=None, show=True, shared_scale_quantiles=(0,1), 
             row_label_rotation=90, row_label_scale=1.33, row_label_offset=0.03,
             row_label_adjust=0.02, **kwargs):
    """
    Plot data on a map.
    
    Arguments:
    
        - dat: DataSet to plot or list of datasets to plot.
        - dat_proj, dist_proj: Data and display projections.
        - figsize: Figure size width x height.
        - grid: Show grid?
        - ncols/nrows: Number of columns/rows to plot.
        - title: Title(s) for the plot(s).
        - share_scale: Make the range of values in each plot the same?
        - colour_scale: Tuple with min/max values to use on scale. Overwritten by share_scale.
        - cbar_ticks: Ticks for the colourbar.
        - tick_labels: Colour bar tick labels.
        - file: If specified save to 'file' instead of showing onscreen.
        - scale_label: The label for a shared scale.
        - share_axes: Share left/bottom axes?
        - ticks_left, ticks_bottom: Display ticks on the left/bottom of plots?
        - wspace, hspace: gridspec wspace and hspace arguments.
        - stippling: Stippling per axis.
        - cbar_adjust: Amount to shrink plots by to add cbar for shared scale.
        - cbar_pad: Padding between figure and colour bar.
        - col_labels/row_labels: Labels for each column/row; overwrites individial plot titles.
        - xlims, ylims: x and y limits.
        - show: Show the map?
        - row_label_rotation: Rotation for row labels.
        - row_label_scale: Scale factor for gap between row labels.
        - row_label_offset: Offset for first row label.
        - row_label_adjust: Adjust setting for row label space on left of plot.
        - kwargs: Extra arguments to plot_map_to_ax.
        
    Return: 
        - The axis plotted to.
        
    """
    
    fig, ax = plt.subplots(figsize=figsize, ncols=ncols, nrows=nrows, 
                           subplot_kw={'projection': disp_proj},
                           gridspec_kw={'wspace': wspace,
                                        'hspace': hspace})
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    
    if not isinstance(dat, list):
        im = plot_map_to_ax(dat=dat, ax=ax, grid=grid, dat_proj=dat_proj, 
                            disp_proj=disp_proj, title=title, stippling=stippling,
                            colour_scale=colour_scale, cbar_ticks=cbar_ticks,
                            cbar_pad=cbar_pad, tick_labels=tick_labels, ticks_left=ticks_left,
                            ticks_bottom=ticks_bottom, xlims=xlims, ylims=ylims, **kwargs)
    else:
        assert ncols*nrows >= len(dat), 'Not enough cols/rows to fit all plots.'
      
        if share_scale and colour_scale is None:
            all_vals = np.array([])
            
            for d in dat:
                all_vals = np.concatenate([all_vals, np.array(d.values.flat)])
            
            colour_scale = (np.nanquantile(all_vals, shared_scale_quantiles[0]), 
                            np.nanquantile(all_vals, shared_scale_quantiles[1]))
            assert not (np.isnan(colour_scale[0]) or np.isnan(colour_scale[1])), 'share_scale cannot be used with subplots missing data.'
                
        for i, d in enumerate(dat):
            ax_title = None
            if not title is None:
                ax_title = title[i]
            
            tb = ticks_bottom
            tl = ticks_left
            if share_axes:
                if i < (ncols*nrows)-ncols:
                    tb = False
                if i % ncols != 0:
                    tl = False
                
            stipple = None if stippling is None else stippling[i]
            proj = dat_proj if not isinstance(dat_proj, list) else dat_proj[i]
            xlim = xlims if not isinstance(xlims, list) else xlims[i]
            ylim = ylims if not isinstance(ylims, list) else ylims[i]
            
            im = plot_map_to_ax(dat=d, ax=ax.flat[i], grid=grid, 
                                dat_proj=proj, disp_proj=disp_proj, title=ax_title,
                                colour_scale=colour_scale, cbar_pad=cbar_pad,
                                cbar_ticks=cbar_ticks, tick_labels=tick_labels,
                                colourbar=(not share_scale),
                                stippling=stipple, xlims=xlim, ylims=ylim,
                                ticks_left=tl, ticks_bottom=tb, **kwargs)
            
        while i+1 < len(ax.flat):
            fig.delaxes(ax.flat[i+1])
            i = i + 1
        
        if share_scale:
            fig.subplots_adjust(right=cbar_adjust)
            cbar_ax = fig.add_axes([cbar_adjust+cbar_pad, 0.23, 0.02, 0.55])
            fmt = mticker.ScalarFormatter(useOffset=False, useMathText=True)
            fmt.set_powerlimits((-4, 6))
            cb = fig.colorbar(im, ax=ax, cax=cbar_ax, ticks=cbar_ticks, label=scale_label, format=fmt)
            
        if col_labels is not None or row_labels is not None:
            for a in ax.flat:
                a.set_title('')

            if col_labels is not None:
                axes = ax if ax.ndim == 1 else ax[0,:] 
                for a, lab in zip(axes, col_labels):
                    a.set_title(lab, fontsize=plt.rcParams['font.size'])

            if row_labels is not None:
                fig.subplots_adjust(left=row_label_adjust)
                lab_ax = fig.add_axes([0, 0.11, row_label_adjust, 0.78], autoscale_on=True)
                lab_ax.axis('off')
                
                for i, lab in enumerate(row_labels):
                    p = row_label_offset + (i/len(row_labels))*row_label_scale
                    lab_ax.annotate(lab, xy=(0.5, 1-p), rotation=row_label_rotation,
                                    xycoords='axes fraction', ha='center')

    if not file is None:
        plt.savefig(fname=file, bbox_inches='tight')
        
        if show:
            plt.show()
        plt.close()
    else:
        if show:
            plt.show()
        
    return ax

def annual_stats(d, factor, day_vars = ['hail_proxy', 'hail_proxy_noconds'],
                 mean_vars = ['mixed_100_cape', 'mixed_100_cin', 'mixed_100_lifted_index', 'lapse_rate_700_500', 
                              'temp_500', 'melting_level', 'shear_magnitude'],
                 quantile_vars = {0.01: ['mixed_100_cin', 'mixed_100_lifted_index', 'lapse_rate_700_500'],
                                  0.99: ['mixed_100_cape', 'temp_500', 'melting_level', 'shear_magnitude']},
                 chunks={'time': -1, 'lat': 30, 'lon': 30},
                 time_chunk=500):
    """
    Calculate annual statistics for a dataset.
    
    Arguments:
        d: Data on which to calculate statistics.
        factor: The factor by which to multiply the mean number of days (for 'day_vars' variables) 
                to get the number of days per period. Eg. 365 for annual, 91 for seasonal. 
        day_vars: Variables for which the statistic is mean number of days per year. Expected to be boolean or binary.
        mean_vars: Variables for which to calculate the mean per year.
        quantile_vars: {percentile: variables} dictionary showing which percentiles to use for extreme values of each variable.
        chunks: Chunks to use for quantile calculation.
        time_chunk: Chunk for times in daily timeseries.
        
    Returns: statistics per year.
    """
    
    # Annual hail-prone days.
    daily = d[day_vars].resample(time='D').max(keep_attrs=True).chunk({'time': time_chunk})
    with xarray.set_options(keep_attrs=True):
        days = daily.groupby('time.year').mean(keep_attrs=True) * factor
        days = days.chunk(-1)
    
    # Annual ingredient means.
    means = d[mean_vars].groupby('time.year').mean(keep_attrs=True)
    for v in mean_vars:
        means = means.rename({v: f'mean_{v}'})
    
    # Annual ingredient extremes.
    extremes = xarray.Dataset()
    for q in quantile_vars:
        quants = d[quantile_vars[q]].chunk(chunks).groupby('time.year').quantile(q, keep_attrs=True).drop('quantile')
        extremes = xarray.merge([extremes, quants])
        for v in quantile_vars[q]:
            extremes[v].attrs['description'] = f'Percentile {q}'
            extremes = extremes.rename({v: f'extreme_{v}'})
            
    ret = xarray.merge([days, means, extremes])
    return ret

def epoch_stats(d, factors = {'DJF': 90, 'MAM': 92, 'JJA': 92, 'SON': 91}):
    """
    Calculate annual and seasonal statistics.
    
    Arguments:
        d: Data to work on.
        factors: Factor argument (length in days) for each season.
        
    Returns: A single xarray object with annual and seasonal stats.
    """
    
    print('Annual...')
    annual = annual_stats(d=d, factor=365)
    annual = annual.rename({n: f'annual_{n}' for n in annual.data_vars})

    seasonal = []
    for s in ['DJF', 'MAM', 'JJA', 'SON']:
        print(f'{s}...')
        seasonal.append(annual_stats(d.where(d.time.dt.season == s), factor=factors[s]).expand_dims({'season': [s]}))

    seasonal = xarray.combine_nested(seasonal, concat_dim='season', combine_attrs='no_conflicts')
    seasonal = seasonal.rename({n: f'seasonal_{n}' for n in seasonal.data_vars})
    dat = xarray.merge([seasonal, annual])
    return dat
    
def process_epoch(epoch_name, model_name, exp, epoch_dates, expected_times=365*20*4,
                  data_dir='/g/data/up6/tr2908/future_hail_global/CMIP_conv/',
                  non_na_var='hail_proxy'):
    """
    Process an epoch for a given simulation.
    
    Arguments:
        epoch_name: Name of the epoch.
        model_name: The model name. 
        exp: Name of the experiment (e.g. historical/ssp585).
        epoch_dates: The epoch start/end times as a tuple.
        expected_times: Number of times expected in the epoch.
        data_dir: Base data directory.
        non_na_var: A variable to check there are no NAs in.     
    """

    sim_dir = f'{model_name}/{exp}'
    files = glob(f'{data_dir}/{sim_dir}/*.nc')
    if len(files) == 0:
        print(f'No files available for {sim_dir}.')
        return None

    dat = xarray.open_mfdataset(sorted(files), parallel=True, chunks={'time': 1000}, 
                                combine='nested', concat_dim='time').sortby('time')
    dat = dat.sel(time=slice(f'{epoch_dates[0]}-01-01', f'{epoch_dates[1]}-12-31'))

    # Convert from longitude 0-360 to -180-180.
    lon_attrs = dat.lon.attrs
    dat['lon'] = dat.lon.where(dat.lon < 180, other=dat.lon-360)
    dat = dat.sortby('lon')
    dat.lon.attrs = lon_attrs

    if dat.time.size != expected_times:
        msg = (f'Warning: {sim_dir} epoch {epoch_name}: number of times returned ' + 
               f'({dat.time.size}) does not match expected size ({expected_times}).')
        print(msg)
        return None

    # Check no NAs in non_na_var.
    assert not np.any(np.isnan(dat))[non_na_var], f'NaN found in {non_na_var}'
    
    stats = epoch_stats(d=dat)
    stats = stats.chunk(-1)
    stats = stats.expand_dims({'model': [model_name],
                               'epoch': [epoch_name]})
    stats.attrs['epoch_dates'] = f'{epoch_dates[0]}-{epoch_dates[1]}'
        
    return stats

def make_landsea_mask(lsm_file='/g/data/oi10/replicas/CMIP6/CMIP/MRI/MRI-ESM2-0/historical/r1i1p1f1/fx/sftlf/gn/v20190603/sftlf_fx_MRI-ESM2-0_historical_r1i1p1f1_gn.nc',
                      out_res = 1, percent_land_required=25):
    """
    Calculate a land-sea mask based on a CMIP dataset's land-sea mask.
    
    Arguments:
        lsm_file: The CMIP land-sea mask file to open and regrid.
        out_res: Output resolution to grid to (in degrees).
        percent_land_required: How much of a grid cell needs to be considered land to be a "land" pixel in the returned map?
        
    Returns: A land-sea mask.
    """
    
    lsm = xarray.open_mfdataset(lsm_file)
    
    # Regrid to output resolution.
    out_lon = np.arange(-180.0, 180.0 + out_res, out_res)
    out_lat = np.arange(-90, 90 + out_res, out_res)
    out_grid = xarray.Dataset(coords={'lat': (out_lat[:-1] + out_lat[1:]) / 2,
                                      'lon': (out_lon[:-1] + out_lon[1:]) / 2, 
                                      'lat_b': out_lat,
                                      'lon_b': out_lon})
    
    regridder = xe.Regridder(lsm, out_grid, 'bilinear', periodic=True)
    lsm = regridder(lsm)
    
    lsm['land'] = lsm.sftlf > percent_land_required
    lsm = lsm.drop('sftlf')
    lsm.land.attrs['long_name'] = 'Land mask'
    lsm.land.attrs['description'] = f'Defined as sftlf > 50 in {lsm_file}.'
    
    return lsm

def plot_seasonal_maps(dat, variable, lat_range=slice(None, None), lon_range=slice(None, None), figsize=(12,8), cmap=hail_cmap, **kwargs):
    """
    Plot maps per model and per season - rows per model and columns per season.
    
    Argumnets:
        d: The data to plot.
        variable: Variable to plot.
        lat/lon_range: Ranges of latitude/longitude.
        figsize: Figure width x height.
        kwargs: Extra arguments to plot_map.
    """
        
    d = [dat[variable].mean('year_num').sel(model=m, season=s, lat=lat_range, lon=lon_range) for 
         m, s in itertools.product(dat.model.values, dat.season.values)]
    titles = [f'{m} ({s})' for m, s in itertools.product(dat.model.values, dat.season.values)]

    _ = plot_map(d, title=titles, figsize=figsize, nan_colour='white', cmap=cmap,
                 ncols=dat.season.size, nrows=dat.model.size, share_scale=True, share_axes=True, 
                 col_labels=dat.season.values, row_labels=dat.model.values, **kwargs)
    
def ttest(dat, variables, fut_epoch, hist_epoch='historical', sig_level=0.05):
    """
    Apply Welch's t-test for across a given axis between epochs for each model.
    
    Arguments:
        dat: The data to work on.
        variables: The variables to apply the t test to. 
        fut_epoch: The future epoch to compare historical data to.
        hist_epoch: Name of the historical epoch.
        sig_level: The p value to require for significance.
        
    Returns: the t test statistic and the significance result.
    """
    
    res = []
    for variable in variables:
        for model in dat.model.values:
            statres, pval = sp.stats.ttest_ind(a=dat.sel(epoch=fut_epoch, model=model)[variable].values, 
                                               b=dat.sel(epoch=hist_epoch, model=model)[variable].values,
                                               equal_var=False)
    
            res_dims = list(dat.sel(epoch=fut_epoch, model=model)[variable].dims)[1:]
            r = xarray.Dataset({variable+'_ttest_stat': (res_dims, statres),
                                variable+'_sig': (res_dims, pval < sig_level)},
                               coords={x: dat[x].values for x in res_dims})
            r = r.expand_dims({'model': [model]})
            res.append(r)

    res = xarray.merge(res)
    res.attrs['p-value for significance'] = sig_level
    return res

def select_all_models(dstore_files=[('/g/data/dk92/catalog/v2/esm/cmip6-oi10/catalog.json', '/g/data/oi10/replicas'),
                                    ('/g/data/dk92/catalog/v2/esm/cmip6-fs38/catalog.json', '/g/data/fs38/publications')], **kwargs):
    """
    Return modules that are suitable for this work, from a list of catalogue files.

    Arguments:
        datastore_files: Tuples of (catalog_file, data_dir) to use.
        **kwargs: Extra arguments to select_models().

    Returns: A list of suitable model details.
    """

    mods = []
    for [f, d] in dstore_files:
        mods.append(select_models(datastore_file=f, **kwargs))
        mods[-1]['CMIP6_dir'] = d

    return pd.concat(mods).reset_index(drop=True)

def select_models(datastore_file='/g/data/dk92/catalog/v2/esm/cmip6-oi10/catalog.json',
                  variables=['tas', 'ta' , 'uas', 'ua', 'vas', 'va', 'huss', 'hus', 'ps'],
                  experiments=['historical', 'ssp585'],
                  tables=['3hr', '6hrLev']):
    """
    Return models that are suitable for use in this work.
    
    Arguments:
        datastore_file: The datastore file to use.
        variables: Required variables.
        experiments: Required experiments.
        tables: Required tables.
    """
    
    cmip6 = intake.open_esm_datastore(datastore_file)

    subset = cmip6.search(experiment_id=experiments,
        variable_id=variables,
        table_id=tables)

    subset = subset.df.groupby(['source_id', 'experiment_id', 'institution_id', 
                                'member_id', 'project_id']).variable_id.unique()
    
    # Subset to models that have all variables required.
    subset = subset[subset.apply(len).values == len(variables)]

    # Subset to models that have both historical and ssp585 experiments.
    counts = subset.groupby(['source_id', 'member_id']).count()
    assert np.all(counts <= 2), 'More than two options for some model(s).'
    counts = counts[counts == 2]
    chosen_models = counts.index.values
    subset = subset.reset_index().set_index(['source_id', 'member_id']).loc[chosen_models]

    # Make the description string.
    subset = subset.reset_index()
    subset['desc'] = ('CMIP6.' + subset['project_id'] + '.' + 
                      subset['institution_id'] + '.' + 
                      subset['source_id'] + '.' +
                      subset['experiment_id'] + '.' + 
                      subset['member_id'])

    # Tidy up.
    subset = subset.drop(columns=['variable_id', 'project_id'])
    subset = subset.rename(columns={'source_id': 'model',
                                    'member_id': 'ensemble',
                                    'experiment_id': 'exp'})
    
    return subset

def make_run_scripts(runs, scripts_dir='scripts/', template='scripts/process_CMIP_template.sh', yrs=9):
    """
    Make scripts to produce convective indices for each model run.
    
    Arguments:
        runs: As returned by define_runs().
        scripts_dir: The directory in which to write scripts.
        template: The script template to copy and modify.
        yrs: The years to use around the central year of each run.
    """
    
    # Make scripts for each run.
    for index, row in runs.iterrows():
        
        # Make scripts directory per model.
        script_dir = f'{scripts_dir}/{row.model}/'
        if not os.path.exists(script_dir):
            os.mkdir(script_dir)
    
        exp = row.exp.replace(' ', '_').replace('(', '').replace(')', '')
        
        for start in [row.start_year, row.end_year-yrs]:
            script = f'{script_dir}/process_{row.model}_{exp}'
            script = f'{script}_{start}-{start+yrs}.sh'
    
            # Copy the template.
            os.system(f'cp {template} {script}')
        
            # Adapt the template for the model.
            os.system(f'sed -i s/DESC/{row.model}_{start}-{start+yrs}/g {script}')
            os.system(f'sed -i s/MODEL/{row.desc}/g {script}')
            os.system(f'sed -i s/START/{start}/g {script}')
            os.system(f'sed -i s/END/{start+yrs}/g {script}')

def make_postprocessing_scripts(runs, scripts_dir='scripts/', template='scripts/post_process_CMIP_template.sh'):
    """
    Make scripts to produce convective indices for each model run.
    
    Arguments:
        runs: As returned by define_runs().
        scripts_dir: The directory in which to write scripts.
        template: The script template to copy and modify.
    """
    
    # Make scripts for each run.
    for index, row in runs.iterrows():
        
        # Make scripts directory per model.
        script_dir = f'{scripts_dir}/{row.model}/'
        if not os.path.exists(script_dir):
            os.mkdir(script_dir)
    
        # Copy the template.
        script = f'{script_dir}/post_process_{row.model}_{row.epoch_name}.sh'
        os.system(f'cp {template} {script}')
        
        # Adapt the template for the model.
        os.system(f'sed -i s/MODEL_NAME/{row.model}/g {script}')
        os.system(f'sed -i s/EPOCH_NAME/{row.epoch_name}/g {script}')
        os.system(f'sed -i s/EXP/{row.exp_name}/g {script}')
        os.system(f'sed -i s/START_YEAR/{row.start_year}/g {script}')
        os.system(f'sed -i s/END_YEAR/{row.end_year}/g {script}')

def plot_run_years(runs, figsize=(10,2.8), legend_y=9.5, file=None, show=True):
    """
    Make a plot to show the (overlapping) run years per model.

    Arguments:
        runs: As from define_runs().
        figsize: Figure size to use.
        legend_y: Y position for legend.
        file: Plot output file.
        show: Show the plot?
    """

    rename_exps = {'historical': 'Historical',
                   'ssp585 (2C)': 'SSP5-8.5 (2C)',
                   'ssp585 (3C)': 'SSP5-8.5 (3C)'}

    runs = runs.copy()
    for key, val in rename_exps.items():
        runs.loc[runs.exp == key, 'exp'] = val
        
    fig, axs = plt.subplots(figsize=figsize, ncols=1, nrows=len(np.unique(runs.model)))
    
    for i, model in enumerate(np.unique(runs.model)):
        mod_runs = runs.loc[runs.model == model]
        for _, run in mod_runs.iterrows():
            axs[i].fill_betweenx([0, 1], run.start_year, run.end_year+1, label=run.exp, alpha=0.5, zorder=1)
    
        if i < len(np.unique(runs.model)) - 1:
            axs[i].set_xticks([])
        axs[i].set_yticks([])
        axs[i].set_xlim(1980, 2100)
        axs[i].set_ylabel(model, rotation=0, ha='right', va='center')
        
    plt.legend(loc='upper right', bbox_to_anchor=(1.1, legend_y), framealpha=1)

    if not file is None:
        plt.savefig(fname=file, bbox_inches='tight')
        
        if show:
            plt.show()
        plt.close()
    else:
        if show:
            plt.show()

def warming_years(future_models, warming_degrees=2, baseline_range=[1980,1999]):
    """
    For each model, determine the 20 year period around when that model
    reaches a certain warming level over pre-industrial temperatures.

    Arguments:
        future_models: A list of full model descriptions to find years for.
        warming_degrees: The degrees warming to find the year range for.
        baseline_range: Baseline year range.

    Returns: pd.DataFrame with model description, start_year, end_year.
    """

    # Retrieve warming data.
    warming_levels = wl.warming_window(warming_amount=warming_degrees, project='CMIP6', 
                                       baseline_range=baseline_range,
                                       baseline_experiment='historical', 
                                       future_experiment='ssp585').reset_index()

    # New dataframe for results.
    all_levels = pd.DataFrame()
    
    for mod in future_models:
        _, _, _, model, exp, ensemble = mod.split('.')
    
        levels = warming_levels.loc[warming_levels.model == f'{model}_{ensemble}'].copy()
        levels['desc'] = mod
        levels = levels.drop(columns='model')
        all_levels = pd.concat([all_levels, levels])
    
    all_levels = all_levels.reset_index(drop=True)

    return all_levels

def define_runs(models, hist_start=1980, hist_end=1999, warming_degrees=[2, 3],
                tables=[['6hrLev', '6H'], ['3hr', '3H']], 
                variables=['va', 'ta', 'hus', 'ps', 'vas', 'huss', 'tas']):
    # Define model runs with start/end years for the period in which they reached
    # certain warming levels, and a historical period. Exclude models that don't cover
    # the required times.
    
    # Arguments:
    #     models: The models to use.
    #     hist_start, hist_end: The historical period to use.
    #     warming_degrees: Number of degrees warming.
    #     tables, variables: CMIP6 tables and variables to check for temporal coverage.
    
    # Returns: A DataFrame with one row per run.
    # """

    # Remove duplicate models and keep only the first ensemble_id of duplicate models.
    unique_models = models[['model', 'ensemble']].drop_duplicates()
    unique_models = unique_models.groupby('model').first().reset_index()
    unique_models = unique_models.set_index(['model', 'ensemble'])
    m = models.set_index(['model', 'ensemble'])
    models = m.join(unique_models, how='right').reset_index()
    
    # Historical models have the same start and end years.
    hist = models.loc[models.exp == 'historical'].copy()
    hist['start_year'] = hist_start
    hist['end_year'] = hist_end
    hist['exp_name'] = hist.exp
    hist['epoch_name'] = hist.exp
    
    # Find start/end dates for future models for various warming levels.
    all = hist
    for deg in warming_degrees:
        future = models.loc[models.exp != 'historical'].copy()
        warming_period = warming_years(future_models=future.desc.values, warming_degrees=deg, 
                                       baseline_range=[hist_start, hist_end])
        warming_period = warming_period.set_index('desc')[['start_year', 'end_year']]
        future = future.set_index('desc')
        future = future.join(warming_period).reset_index()
        future['epoch_name'] = f'{deg}C'
        future['exp_name'] = future.exp
        future['exp'] = future.exp + f' ({deg}C)'
        all = pd.concat([all, future])
    all = all.reset_index(drop=True)
    
    # Exclude models that are not on pressure levels or do not have the temperal coverage required.
    exclude = []
    for i, run in all.iterrows():
        if np.isin(run.model, exclude):
            continue
        
        base_path = f'{run.CMIP6_dir}/{run.desc.replace(".", "/")}'

        for v in variables:
            for [p, res] in tables:
                path = f'{base_path}/{p}/{v}'
                if os.path.exists(path):
                    break
            assert os.path.exists(path), f'Could not find path/res combinations for {v} under {base_path}.'

            grids = [os.path.basename(x) for x in sorted(glob(f'{path}/*'))]
            assert len(grids) == 1, f'Multiple grids to choose from for {path}/{v}.'
            grid = grids[0]
            
            path = f'{path}/{grid}'
            version = [os.path.basename(x) for x in sorted(glob(f'{path}/v*'))][-1] # Use latest version.
            path = f'{path}/{version}'
            
            files = sorted(glob(f'{path}/*.nc'))
            min_time = times_in_CMIP_file(filename=files[0], res=res, calendar='standard').min()
            max_time = times_in_CMIP_file(filename=files[-1], res=res, calendar='standard').max()
            
            if (min_time > pd.to_datetime(f'{run.start_year}-1-1') or 
                max_time < pd.to_datetime(f'{run.end_year}-12-31')):
                print(f'{run.model} {run.exp} requires {run.start_year}-{run.end_year}, but data for {v} covers only {min_time.year}-{max_time.year}. Excluding {run.model}.')
                exclude.append(run.model)
                break

            # Check model is on pressure levels.
            if v == 'ta':
                ta = xarray.open_dataset(files[0])

                # While we're here, collect some metadata.
                all.loc[i, 'nominal_resolution'] = ta.attrs['nominal_resolution']
                all.loc[i, 'vertical_levels'] = str(int(len(ta.lev.values)))
                
                sn = ta.lev.attrs['standard_name']
                ta.close()
                if sn != 'atmosphere_hybrid_sigma_pressure_coordinate':
                    print(f'{run.model} {run.exp} is not on pressure levels. Excluding {run.model}.')
                    exclude.append(run.model)
                    break
    
    runs = all.iloc[~np.isin(all.model, exclude)]
    runs = runs.reset_index(drop=True)
    
    return runs.reset_index(drop=True)

def make_backup_orography(runs, CMIP6_dir='/g/data/oi10/replicas',
                          backup_orog=('CMIP6/CMIP/CNRM-CERFACS/CNRM-CM6-1/historical/' + 
                                       'r1i1p1f2/fx/orog/gr/v20180917/' + 
                                       'orog_fx_CNRM-CM6-1_historical_r1i1p1f2_gr.nc'), 
                          backup_orog_dir = '/g/data/up6/tr2908/future_hail_global/interpolated_orography',
                          tables = [['6hrLev', '6H'], ['3hr', '3H']],
                          grid_var = 'tas'):
    """
    Make files for backup orography, interpolated from another CMIP6 model, for those models 
    that have not provided orography.

    Arguments:
        runs: Runs as from define_runs().
        CMIP6_dir: Base CMIP6 data directory.
        backup_orog: The orography file to interpolate from.
        backup_orog_dir: The directory for resulting orography files.
        tables: [table, res] pairs to use, in order of preference, for grid_var.
        grid_var: The variable from which to get the grid to interpolate to.

    Returns: The run list with a column 'backup_orography' added.
    """

    orog = None
    runs['backup_orography'] = False
    for i, run in runs.iterrows():
        base_path = CMIP6_dir + '/' + run.desc.replace('.', '/') 
        orog_path = base_path + '/fx/orog/'
        
        if not os.path.exists(orog_path):
            runs.loc[i, 'backup_orography'] = True
            
            _, _, _, mod, exp, ens = run.desc.split('.')
            out_file = f'{backup_orog_dir}/orog_{mod}.{exp}.{ens}.nc'            
            if(os.path.exists(out_file)):
                continue

            print('Producing backup orography for ' + run.desc)
            
            # Open a variable to get the grid to interpolate to.
            for [p, res] in tables:
                path = f'{base_path}/{p}/{grid_var}'
                if os.path.exists(path):
                    break
            assert os.path.exists(path), f'Could not find path/res combinations for {grid_var}.'
            grids = [os.path.basename(x) for x in sorted(glob(f'{path}/*'))]
            assert len(grids) == 1, f'Multiple grids to choose from for {path}/{v}.'
            grid = grids[0]
            path = f'{path}/{grid}'
            version = [os.path.basename(x) for x in sorted(glob(f'{path}/v*'))][-1] # Use latest version.
            path = f'{path}/{version}'
            print(path)
    
            # Open the existing variable.
            dat = xarray.open_dataset(sorted(glob(path + '/*.nc'))[0])
            
            # Open backup orography if required.
            if orog is None:
                orog = xarray.open_dataset(CMIP6_dir + '/' + backup_orog).orog
    
            # Interpolate backup orography to the model grid. 
            regridder = xe.Regridder(orog, dat, 'bilinear', periodic=True)
            orog = regridder(orog, keep_attrs=True)
    
            orog.attrs['note'] = ('Model orography unavailable, so orography provided by ' + 
                                   backup_orog + ' and regridded to model grid using xESMF.')
    
            orog.to_netcdf(out_file)
            
def read_processed_data(data_dir='/g/data/up6/tr2908/future_hail_global/CMIP_conv_annual_stats/', 
                        data_exp='*common_grid.nc', 
                        rename_models={'EC-Earth_Consortium.EC-Earth3': 'EC-Earth3'}):
    """
    Read all processed data, regridding to a common grid on the way. 

    Arguments:
        data_dir: The data directory to read from.
        data_exp: Expression to match for data.
        rename_models: Models to shorten the names of.

    Returns: Dataset.
    """
    
    # Regrid native grids to common grids.
    regrid_global(path=f'{data_dir}/*native_grid.nc')
    
    # Open all data.
    dat = xarray.open_mfdataset(f'{data_dir}/{data_exp}')

    # Shorten model names if required.
    dat = dat.assign_coords({'model': [rename_models[x] if x in rename_models else x for x in dat.model.values]})

    # Open landsea mask.
    lsm = make_landsea_mask()
    
    # Mask to land area only.
    dat_land = dat.where(lsm.land == 1)
    
    # Load for speed.
    dat = dat.persist()
    dat_land = dat_land.persist()
    
    # Transpose - for the ttest the axis over which the t-test should be applied ('year_num') must be first after model/epoch selection.
    dat = dat.transpose('model', 'epoch', 'year_num', 'season', 'lat', 'lon')

    return dat

def era5_climatology(era5_dir = '/g/data/up6/tr2908/future_hail_global/era5_conv/',
                     cache_file = '/g/data/up6/tr2908/future_hail_global/era5_climatology.nc'):
    """
    Calculate the ERA5 climatology of mean annual hail-prone days.

    Arguments:
        era5_dir: Processed convective files directory for ERA5.
        cache_file: A cache file to write/read to/from.

    Returns: Mean annual hail-prone days from ERA5.
    """
    
    if os.path.exists(cache_file):
        return xarray.open_dataset(cache_file)

    era5 = xarray.open_mfdataset(f'{era5_dir}/*.nc', parallel=True)

    assert len(era5.time) == 365*20*4, 'Incorrect number of times in ERA5 historic period.'
    assert not np.any(np.isnan(era5.hail_proxy)), 'NaNs in ERA5 hail proxy.'

    daily = era5.hail_proxy.resample(time='D').max(keep_attrs=True)
    with xarray.set_options(keep_attrs=True):
        days = daily.groupby('time.year').mean(keep_attrs=True) * 365
        days = days.chunk(-1)

    res = days.mean('year')

    write_output(xarray.Dataset({'annual_mean_hail_proxy': res}), attrs={}, file=cache_file)
    return xarray.open_dataset(cache_file)

def calc_epoch_differences(dat, variables, epochs=['2C', '3C']):
    """
    Calculate differences between historical epoch and warming epochs.

    Arguments:
        dat: The data to work on.
        variables: Which variables to select.
        epochs: The future epochs to test.

    Returns: mean differences, relative mean differences (relative to overall mean per model), 
             values added where the historical period had zeros, and difference significances.
    """
    
    reference = dat.sel(epoch='historical')[variables]
    reference_mean = reference.mean(['year_num']).load()
    
    mean_diffs = []
    mean_diffs_rel = []
    new_areas = []
    sigs = []
    
    for e in epochs:
        # Select the epoch.
        epoch = dat.sel(epoch=e)[variables]
        epoch_mean = epoch.mean(['year_num']).load()
    
        # Difference in means between reference and epoch.
        mean_diff = epoch_mean - reference_mean
        mean_diff_rel = mean_diff / reference_mean.mean(['lat', 'lon']) * 100
        new_area = np.logical_and(epoch_mean > 0,
                                  reference_mean == 0)
        new_area = mean_diff.where(new_area == True)
    
        # Significance of differences.
        tt = ttest(dat=dat, fut_epoch=e, variables=variables)
        
        mean_diffs.append(mean_diff.expand_dims({'epoch': [e]}))
        mean_diffs_rel.append(mean_diff_rel.expand_dims({'epoch': [e]}))
        new_areas.append(new_area.expand_dims({'epoch': [e]}))
        sigs.append(tt.expand_dims({'epoch': [e]}))
    
    mean_diffs = xarray.merge(mean_diffs)
    mean_diffs_rel = xarray.merge(mean_diffs_rel)
    new_areas = xarray.merge(new_areas)
    sigs = xarray.merge(sigs)

    return mean_diffs, mean_diffs_rel, new_areas, sigs

def plot_diffs_by_epoch(dat, models, var, scale_label, figsize=(12, 9.5), ncols=2, nrows=4,
                        row_label_scale=1.3, row_label_offset=0.015, row_label_adjust=0.15, file=None):
    """
    Plot a grid with differences by epoch (columns) and model (rows).

    Arguments:
        dat: The data to plot.
        models: The list of models to include.
        var: The variable to plot differences for.
        cbar_label: The label for the scale.
        figsize: Figure size width x height.
        ncols, nrows: Grid definition.
        row*: Arguments to plot_map() for row label positions.
        file: Output file for plot.
    """

    colour_scale = (dat[var].min(),
                    dat[var].max())
    
    diffs = list(itertools.chain(*zip([dat[var].sel(model=m, epoch='2C') for m in models], 
                                      [dat[var].sel(model=m, epoch='3C') for m in models])))

    _ = plot_map(diffs, ncols=ncols, nrows=nrows, figsize=figsize, disp_proj=ccrs.Robinson(),
                 cmap='RdBu_r', divergent=True, share_scale=True, share_axes=True, grid=False, contour=True,
                 col_labels=['2C', '3C'], row_labels=models, row_label_rotation=0, colour_scale=colour_scale,
                 row_label_scale=row_label_scale, row_label_offset=row_label_offset, 
                 row_label_adjust=row_label_adjust, file=file, scale_label=scale_label)

def plot_mean_diffs_for_epoch(diffs, sigs, variable, scale_label, epoch, figsize=(12,6), file=None, ncols=2, nrows=2, seasons=None, **kwargs):
    """
    Plot differences in ensemble means between two epochs with stippling showing model agreement and significance of differences.

    Arguments:
        diffs: The multi-model mean differences data.
        sigs: Significance information.
        variable: The variable to plot.
        scale_label: Label for the scale.
        epoch: The epoch to plot differences for. 
        figsize: Figure size.
        file: Output plot file.
        ncols/nrows: columns/rows to use.
        seasons: List of seasons to include.
        **kwargs: Extra arguments to plot_map.
    """

    if seasons is None:
        seasons = diffs.season.values
    stippling = [sigs.sel(season=s, epoch=epoch)[variable] for s in seasons]
    differences = [diffs.sel(season=s, epoch=epoch)[variable] for s in seasons]
    
    _ = plot_map(differences, stippling=stippling,
                 title=seasons, share_scale=True, share_axes=True, grid=False,
                 ncols=ncols, nrows=nrows, figsize=figsize, disp_proj=ccrs.Robinson(), 
                 contour=True, cmap='RdBu_r', divergent=True, scale_label=scale_label,
                 file=file, **kwargs)

def plot_mean_diffs_for_season(diffs, sigs, variable, scale_label, season, figsize=(12,6), file=None):
    """
    Plot differences in ensemble means between epochs with stippling showing model agreement and significance of differences.

    Arguments:
        diffs: The multi-model mean differences data.
        sigs: Significance information.
        variable: The variable to plot.
        scale_label: Label for the scale.
        season: The season to plot for.
        figsize: Figure size.
        file: Output plot file.
    """
    
    epochs=diffs.epoch.values
    stippling = [sigs.sel(season=season, epoch=e)[variable] for e in epochs]
    differences = [diffs.sel(season=season, epoch=e)[variable] for e in epochs]
    
    _ = plot_map(differences, stippling=stippling,
                 title=[f'{season}, {e}' for e in epochs], share_scale=True, share_axes=True, grid=False,
                 ncols=1, nrows=2, figsize=figsize, disp_proj=ccrs.Robinson(), 
                 contour=True, cmap='RdBu_r', divergent=True, scale_label=scale_label,
                 file=file)

def multi_model_mean_diffs(diffs, sigs):
    """
    Calculate the multi-model mean difference and indicator of significance. 
    A difference is considered significant if more than 50% of the models have
    both a) significant differences in the mean and b) their mean difference has
    the same sign as the multi-model mean difference.

    Arguments:
        diffs: The differences per model.
        sigs: Significance of differences per model.

    Return the mean differences and significance indicator for each point. 
    """
    
    # Calculate mean difference across models.
    mean_diffs = diffs.mean(['model'])
    mean_sign = np.sign(mean_diffs)
    
    # Where does the sign of the per-model difference agree with the mean sign?
    sign_agrees = np.sign(diffs) == mean_sign
    
    # Where are the differences significant AND the sign matches the mean differences?
    varlist = [x for x in list(sign_agrees.keys()) if np.isin(x + '_sig', list(sigs.keys()))]
    s = sigs.rename({k + '_sig': k for k in varlist})
    sig = np.logical_and(sign_agrees, s)
    
    # How many models have both sig diffs and matching sign? Consider mean difference significant 
    # if more than 75% of models have both these conditions true.
    significance = sig.sum('model') > len(diffs.model)*0.5

    return mean_diffs, significance

def plot_ing_changes(diffs, sigs, epoch, variables, file, seasons=['DJF', 'JJA']):
    """
    Plot changes in ingredients by season.

    Arguments:
        diffs: Differences to plot.
        sigs: Significance information for the differences.
        epoch: The epoch to plot for.
        variables: Which variables to include as a {variable: name} dictionary.
        file: Prefix for the file to save in; will have variable name appended.
        seasons: Seasons to plot for.
    """

    
    for var, var_name in variables.items():
        plot_mean_diffs_for_epoch(diffs=diffs, sigs=sigs, variable=var, seasons=seasons, 
                                  scale_label='', epoch=epoch, ncols=2, nrows=1, figsize=(12,3.5),
                                  row_labels=[var_name], row_label_offset=0.65, row_label_adjust=0.03,
                                  col_labels=seasons, file=file + '_' + var_name.replace(' ', '_').replace('%', 'p') + '.pdf')