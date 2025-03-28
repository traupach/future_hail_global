"""Helper functions for future global hail proxy analysis.

Author: Tim Raupach <t.raupach@unsw.edu.au>
"""

import itertools
import os
import re
from glob import glob

import cartopy.crs as ccrs
import cftime
import dask
import geopandas
import geopandas as gp
import intake
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import metpy
import modules.parcel_functions as parcel
import modules.warming_levels as wl
import numpy as np
import pandas as pd
import scipy as sp
import seaborn as sns
import xarray
import xesmf as xe
from cartopy.io import shapereader
from matplotlib import colors
from matplotlib.colors import BoundaryNorm, LinearSegmentedColormap
from matplotlib.patches import Rectangle
from metpy.units import units
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

# ruff: noqa: E712                                # Don't check rule E712 in Ruff.

# Settings for xarray_parcel: set up parcel adiabat calculations.
lookup_dir = '/g/data/li18/tr2908/'
parcel.load_moist_adiabat_lookups(base_dir=lookup_dir, chunks=-1)

# Proxies to consider.
proxies = [
    'proxy_Raupach2023_original',
    'proxy_Raupach2023_original_noconds',
    'proxy_Raupach2023_updated',
    'proxy_Raupach2023_updated_noconds',
    'proxy_Kunz2007',
    'proxy_Eccel2012',
    'proxy_Mohr2013',
    'proxy_SHIP_0.1',
    'proxy_SHIP_0.5',
]

# Long names for each proxy.
proxy_names = {
    'proxy_Raupach2023_original': 'Raupach 2023',
    'proxy_Raupach2023_original_noconds': 'Raupach 2023\n(no extra conds)',
    'proxy_Raupach2023_updated': 'Raupach',
    'proxy_Raupach2023_updated_noconds': 'Raupach\n(no extra conds)',
    'proxy_Kunz2007': 'Kunz 2007',
    'proxy_Eccel2012': 'Eccel 2012',
    'proxy_Mohr2013': 'Mohr 2013',
    'proxy_SHIP_0.1': 'SHIP > 0.1',
    'proxy_SHIP_0.5': 'SHIP > 0.5',
}

# Names/title of proxies when used as a dimension.
proxy_dims = {p[6:None]: proxy_names[p] for p in proxies}

# Pretty colors for hail maps.
cmap_colours = [
    [0.9, 0.9, 0.9, 1],
    [0.92941176, 0.97254902, 1.0, 1.0],
    [0.85882353, 0.90980392, 0.98823529, 1.0],
    [0.85490196, 0.90980392, 0.99215686, 1.0],
    [0.57254902, 0.75686275, 0.98039216, 1.0],
    [0.51372549, 0.66666667, 0.97254902, 1.0],
    [0.4745098, 0.60784314, 0.96862745, 1.0],
    [0.39607843, 0.54117647, 0.96862745, 1.0],
    [0.29411765, 0.66666667, 0.21960784, 1.0],
    [0.35686275, 0.76470588, 0.34509804, 1.0],
    [0.74117647, 0.83921569, 0.5254902, 1.0],
    [0.7372549, 0.96470588, 0.53333333, 1.0],
    [0.99607843, 0.96862745, 0.32156863, 1.0],
    [0.95294118, 0.69019608, 0.24313725, 1.0],
    [0.91372549, 0.2, 0.14117647, 1.0],
    [0.86666667, 0.18823529, 0.13333333, 1.0],
    [0.63529412, 0.12941176, 0.08235294, 1.0],
]
hail_cmap = LinearSegmentedColormap.from_list('', cmap_colours)


@dask.delayed
def open_CMIP_file_delayed(filename, chunks=None):
    """Open a CMIP file using dask delayed so it is only accessed when requested.

    Arguments:
        filename: The filename to open.
        chunks: Chunks to use.

    """
    d = xarray.open_dataset(filename, use_cftime=True)
    return d.chunk({'time': 20} if chunks is None else chunks)


def open_CMIP_file(filename, res, ex, calendar, chunks=None, **kwargs):
    """Open a CMIP file using dask delayed so it is only accessed when requested.

    Arguments:
        filename: The filename to read.
        res: Resolution to use.
        ex: An example version of the variable to get coordinate info from.
        calendar: Datetime calendar to use.
        chunks: Data chunks to use.
        **kwargs: Extra arguments to open_CMIP_file_delayed.

    """
    dat = open_CMIP_file_delayed(filename=filename, chunks=chunks, **kwargs)[
        ex.name
    ].data
    times = times_in_CMIP_file(filename=filename, res=res, calendar=calendar)
    shape = (times.size,) + ex.shape[1:]
    return dask.array.from_delayed(value=dat, shape=shape, dtype=ex.dtype)


def open_CMIP(path, res, variable, chunks=None, max_year=2100):
    """Open all CMIP files in a given path, using dask delayed for speed.

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
    times = xarray.cftime_range(
        times_first[0],
        times_last[-1],
        freq=res,
        inclusive='both',
        calendar=first.time.dt.calendar,
    )

    # Check that times line up.
    assert times[0] == first.time[0], 'First time does not match.'
    assert times[1] == first.time[1], 'Temporal resolution is not correct.'

    # Concatenate all the delayed-opening objects for all files.
    dat = dask.array.concatenate(
        [
            open_CMIP_file(
                filename=f,
                res=res,
                ex=ex,
                calendar=first.time.dt.calendar,
                chunks=chunks,
            )
            for f in files
        ],
        axis=0,
    )

    # Convert to an xarray object.
    coords = {x: ex[x] for x in ex.dims}
    coords['time'] = times
    ret = xarray.DataArray(
        dat,
        name=ex.name,
        attrs=ex.attrs,
        dims=ex.dims,
        coords=coords,
    )
    ret = ret.chunk(chunks)

    # Clean up.
    ex.close()
    return ret


def times_in_CMIP_file(filename, res, calendar):
    """Use a CMIP filename to return all the timesteps the file contains.

    Arguments:
        filename: The CMIP filename, must end in YYYYMMDDHHMM-YYYYMMDDHHMM.nc.
        res: Temporal resolution for the file.
        calendar: The calendar to use.

    Returns: An array of cftimes for all timesteps in the file.

    """
    basename = os.path.basename(filename)
    time_range = [
        re.search(r'.*_([0-9]{12})-[0-9]{12}\.nc', basename).group(1),
        re.search(r'.*_[0-9]{12}-([0-9]{12})\.nc', basename).group(1),
    ]
    time_range = [
        f'{x[0:4]}-{x[4:6]}-{x[6:8]} {x[8:10]}:{x[10:12]}' for x in time_range
    ]

    res = res.replace('H', 'h')

    return xarray.cftime_range(
        time_range[0],
        time_range[1],
        freq=res,
        inclusive='both',
        calendar=calendar,
    )


def geopotential_height(dat, vert_dim='lev', Rd=287.04749, g=9.80665):
    """Calculate geopotential height for all levels in the data.

    Arguments:
        dat: Data containing orog, surface_temperature, surface_pressure,
             surface_specific_humidity, temperature, pressure, specific_humidity,
             levels.
        vert_dim: The name of the vertical dimension.
        chunks: Chunks to apply to intermediate objects.
        Rd: Gas constant for dry air (J kg-1 K-1).
        g: Gravitation constant (ms-2).

    Returns: A DataArray of geopotential heights (m), and a DataArray of heights above
             the surface.

    """
    assert np.all(dat[vert_dim].diff(vert_dim) == 1), (
        'Vert dim should be an integer increasing index.'
    )

    # Calculate virtual temperature.
    virt_temp = dat.temperature * (1 + 0.608 * dat.specific_humidity)
    vt_surf = dat.surface_temperature * (1 + 0.608 * dat.surface_specific_humidity)

    # Virtual temperature at bottom/top of each layer.
    vt_from = xarray.concat(
        [
            vt_surf.expand_dims({vert_dim: [0]}),
            virt_temp.sel({vert_dim: slice(1, dat[vert_dim].max() - 1)}),
        ],
        dim=vert_dim,
    )
    vt_from = vt_from.assign_coords({vert_dim: vt_from[vert_dim] + 1})
    vt_to = virt_temp
    vt_mean = (vt_from + vt_to) / 2

    # Pressure at the bottom/top of each layer.
    p_from = xarray.concat(
        [
            dat.surface_pressure.expand_dims({vert_dim: [0]}),
            dat.pressure.sel({vert_dim: slice(1, dat[vert_dim].max() - 1)}),
        ],
        dim=vert_dim,
    )
    p_from = p_from.assign_coords({vert_dim: p_from[vert_dim] + 1})
    p_to = dat.pressure
    p_div = np.log(p_from / p_to)

    # Calculate thickness of each layer.
    thickness = Rd / g * vt_mean * p_div
    level_heights = xarray.concat(
        [dat.orog.expand_dims({vert_dim: [0]}), thickness],
        dim=vert_dim,
    )

    # Do a cumsum to get the height of each layer.
    level_heights = level_heights.cumsum(vert_dim).sel({vert_dim: slice(1, None)})
    level_heights.attrs['long_name'] = 'Geopotential height'
    level_heights.attrs['units'] = 'm'

    # Calculate above-surface height.
    above_surface = level_heights - dat.orog
    above_surface.attrs['long_name'] = 'Height above surface'
    above_surface.attrs['units'] = 'm'

    return level_heights, above_surface


def read_all_CMIP_data(
    model,
    CMIP6_dir='/g/data/oi10/replicas',
    orog_path='fx/orog/',
    backup_orog_path='/g/data/up6/tr2908/future_hail_global/interpolated_orography/',
    max_year=2100,
    pressure_var='ta',
    out_res='6H',
    chunks=None,
    grid=None,
    tables=None,
    variables=None,
):
    """Read all CMIP6 data for a given model, at 6H resolution.

    Arguments:
        model: Model spec (e.g. CMIP6.ScenarioMIP.MRI.MRI-ESM2-0.ssp585.r1i1p1f1)
        CMIP6_dir: Directory for CMIP6 data.
        orog_path: Path to orography.
        backup_orog_path: Path to directory containing interpolated orography files for
                          models missing orography.
        max_year: Read in data to this year only.
        pressure_var: The variable to use for pressure definition.
        out_res: Output resolution.
        chunks: Chunks to use for opening datasets.
        grid: Grid to use.
        tables: [path, res] groups to search for variables in, in order of preference.
        variables: variables to read.

    Returns: An xarray object with all CMIP6 data for the given model at the correct
             resolution.

    """
    if tables is None:
        tables = [['6hrLev', '6H'], ['3hr', '3H']]
    if variables is None:
        variables = ['va', 'ua', 'ta', 'hus', 'ps', 'vas', 'uas', 'huss', 'tas']
    if chunks is None:
        chunks = {'time': 20}

    model_dir = model.replace('.', '/')
    base_path = f'{CMIP6_dir}/{model_dir}'
    dat = xarray.Dataset()

    # Parse model string. Return values are project, mip, source, model_name,
    # exp, ensemble.
    _, _, _, model_name, exp, ensemble = model.split('.')

    for v in variables:
        for [p, res] in tables:  # noqa: B007
            path = f'{base_path}/{p}/{v}'
            if os.path.exists(path):
                break
        assert os.path.exists(path), f'Could not find path/res combinations for {v}.'

        if grid is None:
            grids = [os.path.basename(x) for x in sorted(glob(f'{path}/*'))]
            assert len(grids) == 1, f'Multiple grids to choose from for {path}/{v}.'
            grid = grids[0]

        path = f'{path}/{grid}'
        version = [os.path.basename(x) for x in sorted(glob(f'{path}/v*'))][
            -1
        ]  # Use latest version.
        path = f'{path}/{version}'

        print(path, flush=True)
        d = open_CMIP(path=path, res=res, variable=v, chunks=chunks)
        d = d.sel(time=slice(None, f'{max_year - 1}-12-31'))

        # Keep data only every 6H.
        if res != out_res:
            assert 'time' in dat, (
                f'Must include data natively at {out_res} '
                'before data that requires resampling.'
            )
            keep_times = xarray.cftime_range(
                dat.time.values[0],
                dat.time.values[-1],
                freq=out_res,
                inclusive='both',
                calendar=d.time.dt.calendar,
            )
            keep_times = keep_times[keep_times >= d.time.values[0]]
            keep_times = keep_times[keep_times <= d.time.values[-1]]
            d = d.sel(time=keep_times)

        # Add metadata.
        d.attrs['CMIP_model_spec'] = model
        d.attrs['CMIP_model_name'] = model_name
        d.attrs['CMIP_model_version'] = version
        d.attrs['CMIP_grid_spec'] = grid
        d.attrs['processing_note'] = (
            f'Data subset to {out_res} temporal resolution and maximum year {max_year}.'
        )

        # Chunk.
        dat[v] = d.chunk(chunks)

        # Check all grid point match.
        if 'lat' in dat.coords:
            assert np.all(dat.lat == d.lat), (
                'Latitudes for {v} do not match other variables.'
            )
            assert np.all(dat.lon == d.lon), (
                'Longitudes for {v} do not match other variables.'
            )
            if 'lev' in d.coords:
                assert np.all(dat.lev == d.lev), (
                    'Levels for {v} do not match other variables.'
                )

        # Process pressure as required.
        if v == pressure_var:
            ex = xarray.open_dataset(glob(f'{path}/*.nc')[0]).chunk(chunks)

            # Pressure is defined as p = a*p0 + b*ps.
            if 'p0' in ex:
                # Some modules use p = a*p0 + b*ps.
                dat['a'] = ex.a
                dat['p0'] = ex.p0
                assert dat.lev.attrs['formula'] == 'p = a*p0 + b*ps', (
                    'Pressure formula problem.'
                )

            else:
                # Some modules use ap, in which a is already multiplied by p.
                dat['a'] = ex.ap
                dat['p0'] = xarray.ones_like(dat.a)
                assert dat.lev.attrs['formula'] == 'p = ap + b*ps', (
                    'Pressure formula problem.'
                )

            dat['b'] = ex.b
            del ex

    # Rename variables.
    dat = dat.rename(
        {
            'vas': 'surface_wind_v',
            'uas': 'surface_wind_u',
            'va': 'wind_v',
            'ua': 'wind_u',
            'ta': 'temperature',
            'tas': 'surface_temperature',
            'hus': 'specific_humidity',
            'huss': 'surface_specific_humidity',
            'ps': 'surface_pressure',
        },
    )

    # Read orography.
    orog_p = f'{base_path}/{orog_path}/{grids[0]}/'
    if os.path.exists(orog_p):
        # Use model orography if it is provided.
        orog_version = [os.path.basename(x) for x in sorted(glob(f'{orog_p}/*'))][0]  # noqa: RUF015
        orog = xarray.open_mfdataset(f'{orog_p}/{orog_version}/*.nc', use_cftime=True)[
            'orog'
        ]

        # If orography coordinates differ slightly from other-data coordinates, use main
        # data coordinates.
        if not np.all(dat.lat.values == orog.lat.values):
            assert np.max(np.abs(dat.lat.values - orog.lat.values)) < 1e-5, (  # noqa: PLR2004
                'Mismatch in orography latitudes.'
            )
            orog = orog.assign_coords({'lat': dat.lat.values})
        if not np.all(dat.lon.values == orog.lon.values):
            assert np.max(np.abs(dat.lon.values - orog.lon.values)) < 1e-5, (  # noqa: PLR2004
                'Mismatch in orography longitudes.'
            )
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
    return dat.assign_coords({'lev': np.arange(1, dat.lev.size + 1)})


def write_output(dat, file, attrs=None):
    """Write data in 'dat'.

    Use attributes 'attrs', to 'file' as NetCDF with compression.
    """
    comp = {'zlib': True, 'shuffle': True, 'complevel': 4}
    encoding = {var: comp for var in dat.data_vars}
    if attrs is not None:
        dat.attrs = attrs
    dat.to_netcdf(file, encoding=encoding)


def regrid_global(
    path,
    out_res=1,
    rename=None,
    output_file=lambda x: x.replace('native_grid', 'common_grid'),
    isel=None,
):
    """Regrid all (global) files in a directory to a common grid.

    Arguments:
        path: Path to match files to regrid.
        out_res: Output resolution in degrees.
        rename: Rename variables before saving?
        output_file: Function to convert input filename to output filename.
        isel: Elements to select (or None to not subset).

    """
    files = glob(path)

    out_lon = np.arange(-180.0, 180.0 + out_res, out_res)
    out_lat = np.arange(-90, 90 + out_res, out_res)
    out_grid = xarray.Dataset(
        coords={
            'lat': (out_lat[:-1] + out_lat[1:]) / 2,
            'lon': (out_lon[:-1] + out_lon[1:]) / 2,
            'lat_b': out_lat,
            'lon_b': out_lon,
        },
    )

    for file in files:
        outfile = output_file(file)
        if not os.path.exists(outfile):
            print(f'Regridding {file}.')
            d = xarray.open_dataset(file)
            regridder = xe.Regridder(d, out_grid, 'bilinear', periodic=True)
            d = regridder(d, keep_attrs=True)
            attrs = d.attrs
            attrs.update(
                {
                    'history': (
                        f'Regridded to {out_res} x {out_res}degree grid using xESMF.'
                    ),
                },
            )

            if rename is not None:
                d = d.rename(rename)
            if isel is not None:
                d = d.isel(isel)

            write_output(dat=d, file=outfile, attrs=attrs)


def conv_CMIP(
    dat,
    year,
    outdir='/g/data/up6/tr2908/future_hail_global/CMIP_conv/',
    days_per_outfile=30,
):
    """Calculate convective parameters for a given year and output to NetCDF.

    Arguments:
        dat: The CMIP6 data to process.
        year: The year to process.
        outdir: Output directory minus model name.
        days_per_outfile: Number of days to write to each output file.

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
    to_date = cftime.DatetimeNoLeap(year, 12, 31)
    days_in_year = xarray.cftime_range(from_date, to_date, freq='D', calendar='noleap')

    # Process attributes.
    data_vars = [x for x in dat.data_vars if 'CMIP_model_version' in dat[x].attrs]
    attrs = {'CMIP_version_' + x: dat[x].attrs['CMIP_model_version'] for x in data_vars}
    attrs = {'CMIP_grid_' + x: dat[x].attrs['CMIP_grid_spec'] for x in data_vars}
    attrs['CMIP_model_spec'] = dat.temperature.CMIP_model_spec
    attrs['CMIP_model_name'] = dat.temperature.CMIP_model_name
    attrs['CMIP_note'] = 'Pressure derived from version used for temperature (ta).'

    # Process each day in turn.
    conv = []

    for j, day in enumerate(days_in_year):
        f_from = (j // days_per_outfile) * days_per_outfile + 1
        f_to = f_from + days_per_outfile - 1
        f_to = min(f_to, len(days_in_year))
        outfile = f'{outdir}/{model_name}_{year}_{f_from:03}-{f_to:03}.nc'

        assert day.year == year, 'Year mismatch.'

        if os.path.exists(outfile):
            print(f'Skipping output for {outfile}...')
        else:
            # Get this day's data.
            print(f'Processing {day.year}-{day.month:02}-{day.day:02}...')
            day_dat = dat.sel(time=f'{day.year}-{day.month:02}-{day.day:02}')

            # Calculate geopotential heights.
            day_dat['height_asl'], day_dat['wind_height_above_surface'] = (
                geopotential_height(day_dat)
            )

            # Load for speed.
            day_dat = day_dat.load()
            day_dat = day_dat.chunk({'time': 1})

            # Calculate convective parameters.
            c = conv_properties(dat=day_dat, vert_dim='lev').load()

            # Save for writing.
            conv.append(c)

            print('Cleaning up...')
            del (
                day_dat,
                c,
            )  # ,res ,prox

            if len(conv) >= days_per_outfile:
                write_output(dat=xarray.merge(conv), attrs=attrs, file=outfile)
                del conv
                conv = []

    if len(conv) != 0:
        write_output(dat=xarray.merge(conv), attrs=attrs, file=outfile)


def plot_map_to_ax(
    dat,
    ax,
    coastlines=True,
    grid=True,
    dat_proj=ccrs.PlateCarree(),
    disp_proj=ccrs.PlateCarree(),
    title=None,
    colour_scale=None,
    cmap=hail_cmap,
    norm=None,
    cbar_ticks=None,
    tick_labels=None,
    contour=False,
    stippling=None,
    colourbar=True,
    ticks_left=True,
    ticks_bottom=True,
    cbar_aspect=25,
    cbar_fraction=0.07,
    cbar_shrink=0.4,
    cbar_pad=0.015,
    cbar_label=None,
    cbar_orientation='vertical',
    coastlines_colour='black',
    xlims=None,
    ylims=None,
    num_ticks=None,
    divergent=False,
    cbar_inset=False,
    title_inset=False,
    title_inset_pos='upper',
    discrete=False,
    log_scale=False,
    nan_colour='#eeeeee',
    axis_off=False,
    country=None,
    annotations=None,
    num_contours=len(cmap_colours) + 1,
    left_title=None,
    cbar_extend='neither',
):
    """Plot data on a map to a specified plot axis object.

    Arguments:
        dat: DataSet to plot or list of datasets to plot.
        ax: GeoAxes object to plot to.
        dat_proj, dist_proj: Data and display projections.
        figsize: Figure size width x height.
        coastlines: Show coastlines?
        grid: Show grid?
        ncol/nrows: Number of columns/rows to plot.
        title: Title for the plot.
        colour_scale: None for default, or a tuple of min/max values for the scale.
        cmap: The matplotlib colour map to use.
        norm: A norm object for colours (e.g. colors.BoundaryNorm).
        cbar_ticks: Colour bar ticks.
        tick_labels: Colour bar tick labels.
        contour: Plot using xarray's contourf function?
        stippling: True where trippling should appear.
        stipple_size: Size for stippling points.
        colourbar: Include a colourbar?
        ticks_left: Include ticks on left of plot?
        ticks_bottom: Include ticks on bottom of plot?
        cbar_aspect: colorbar aspect ratio?
        cbar_fraction: fraction argument for colorbar().
        cbar_shrink: shrink argument for colorbar().
        cbar_pad: pad argument for colorbar().
        cbar_label: Overwrite label?
        cbar_orientation: orientation argument for colorbar().
        coastlines_colour: Colour for coastlines.
        xlims, ylims: x and y plot limits.
        num_ticks: Number of ticks for x and y axes (None for auto).
        divergent: Is the colour scale divergent? If so make zero central.
        cbar_inset: Inset the colorbar in lower left?
        title_inset: Inset the title?
        title_inset_pos: 'lower' or 'upper'.
        discrete: Make the colour bar discrete?
        log_scale: Make the colour scale log-scaled?
        nan_colour: Colour for missing values.
        axis_off: Turn off all axes.
        country: Plot coastlines only for a specific country.
        annotations: Add annotations to the map - dictionary of
                     {'Text': [x, y, xadj, yadj, ha]} where
                     x, y are position of text, xadj, yadj give offsets
                     to the label, ha is 'left' or 'right' for horizontal anchor.
        num_contours: The number of contours to use in a contour plot.
        left_title: Left-hand title to put on the plot.
        cbar_extend: colorbar extend argument.
        dat_proj: Projection of the data.
        disp_proj: Projection for map display.
        xlims: Limits on x axis.
        ylims: iImits on y axis.

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

        col_min = -1 * np.max(np.abs([col_min, col_max]))
        col_max = np.max(np.abs([col_min, col_max]))

    cbar_spacing = 'proportional'

    if discrete:
        assert cbar_ticks is not None, 'Discrete colorbar requires cbar_ticks'
        assert cbar_ticks == sorted(cbar_ticks), (
            'cbar_ticks must be sorted for discrete plot.'
        )
        cbar_ticks = np.array([*cbar_ticks, np.max(cbar_ticks) + 1])
        norm = colors.BoundaryNorm(cbar_ticks, ncolors=len(cbar_ticks) - 1)
        cbar_ticks = (cbar_ticks[0:-1] + cbar_ticks[1:]) / 2
        cbar_spacing = 'uniform'

    if log_scale:
        norm = colors.LogNorm(vmin=col_min, vmax=col_max)

    fmt = mticker.ScalarFormatter(useOffset=False, useMathText=True)
    fmt.set_powerlimits((-4, 6))
    cbar_args = {
        'spacing': cbar_spacing,
        'fraction': cbar_fraction,
        'ticks': cbar_ticks,
        'aspect': cbar_aspect,
        'shrink': cbar_shrink,
        'pad': cbar_pad,
        'orientation': cbar_orientation,
        'format': '%g',
        'extend': cbar_extend,
    }

    if cbar_inset:
        cax = inset_axes(
            ax,
            width='50%',
            height='3%',
            loc='lower left',
            bbox_to_anchor=(0.05, 0.15, 1, 1),
            bbox_transform=ax.transAxes,
        )
        cbar_args['cax'] = cax

    if cbar_label is not None:
        cbar_args['label'] = cbar_label

    if colourbar is False:
        cbar_args = None

    cmap = plt.get_cmap(cmap).copy()
    cmap.set_bad(nan_colour)

    if not contour:
        res = dat.plot(
            ax=ax,
            transform=dat_proj,
            vmin=col_min,
            vmax=col_max,
            cmap=cmap,
            norm=norm,
            cbar_kwargs=cbar_args,
            add_colorbar=colourbar,
            zorder=-10,
        )
    else:
        if col_min is not None:
            col_min = np.floor(col_min)
            col_max = np.ceil(col_max)
            levs = np.linspace(col_min, col_max, num_contours, dtype=np.int64)
        else:
            levs = num_contours

        res = dat.plot.contourf(
            ax=ax,
            transform=dat_proj,
            vmin=col_min,
            vmax=col_max,
            cmap=cmap,
            norm=norm,
            cbar_kwargs=cbar_args,
            add_colorbar=colourbar,
            levels=levs,
        )

    if stippling is not None:
        ax.autoscale(False)  # noqa: FBT003
        hatches = [None, '\\\\\\']
        if np.all(stippling == True):
            hatches = ['\\\\\\', '\\\\\\']
        stippling.plot.contourf(
            transform=dat_proj,
            ax=ax,
            add_colorbar=False,
            levels=3,
            colors='none',
            hatches=hatches,
        )

    if xlims is not None:
        ax.set_xlim(xlims)
    if ylims is not None:
        ax.set_ylim(ylims)
    if colourbar == True and tick_labels is not None:
        assert len(tick_labels) == len(cbar_ticks), (
            'Labels and ticks must have same length'
        )
        res.colorbar.ax.set_yticklabels(tick_labels)
    if left_title is not None:
        if title_inset:
            title = f'{left_title} {title if title is not None else ""}'
        else:
            ax.set_title(left_title, fontsize=plt.rcParams['font.size'], loc='left')
    if title is not None:
        if title_inset:
            ax.set_title('')
            if title_inset_pos == 'upper':
                ax.annotate(
                    text=title,
                    xy=(0.05, 0.9),
                    xycoords='axes fraction',
                    fontweight='bold',
                    fontsize=plt.rcParams['font.size'],
                )
            elif title_inset_pos == 'lower':
                ax.annotate(
                    text=title,
                    xy=(0.05, 0.07),
                    xycoords='axes fraction',
                    fontweight='bold',
                    fontsize=plt.rcParams['font.size'],
                )
            elif title_inset_pos == 'left_lower':
                ax.annotate(
                    text=title.replace(')', ''),
                    xy=(0.01, 0.01),
                    xycoords='axes fraction',
                    fontweight='bold',
                    fontsize=plt.rcParams['font.size'],
                )
            else:
                assert 1 == 0, 'title_inset_pos must be upper or lower or left_lower.'  # noqa: PLR0133
        else:
            ax.set_title(title, fontsize=plt.rcParams['font.size'])
    if coastlines:
        if country is not None:
            shpfilename = shapereader.natural_earth(
                resolution='10m',
                category='cultural',
                name='admin_0_countries',
            )
            df = geopandas.read_file(shpfilename)
            poly = df.loc[df['ADMIN'] == country]['geometry'].values[0]
            ax.add_geometries(
                poly,
                crs=ccrs.PlateCarree(),
                facecolor='none',
                edgecolor=coastlines_colour,
                linewidth=0.75,
            )
        else:
            ax.coastlines(color=coastlines_colour)
    if grid:
        locator = None
        if num_ticks is not None:
            locator = mticker.MaxNLocator(nbins=num_ticks + 1)
        gl = ax.gridlines(
            crs=disp_proj,
            draw_labels=True,
            alpha=0.5,
            xlocs=locator,
            ylocs=locator,
        )
        gl.top_labels = gl.right_labels = False
        gl.left_labels = ticks_left
        gl.bottom_labels = ticks_bottom
    if axis_off:
        ax.axis('off')

    if annotations is not None:
        for text, [x, y, xadj, yadj, ha] in annotations.items():
            if np.abs(xadj) >= 1 or np.abs(yadj) >= 1:
                if ha in ('right', 'left'):
                    ax.plot(
                        [x, x + xadj - (0.2 * np.sign(xadj))],
                        [y, y + yadj + 0.2],
                        color='black',
                    )
                elif ha == 'center':
                    ax.plot(
                        [x, x + xadj - (0.2 * np.sign(xadj))],
                        [y, y + yadj - 0.2],
                        color='black',
                    )
                    if yadj < 0:
                        print(
                            'Warning: ha=center and negative y adjustment '
                            'are not supported.',
                        )
                else:
                    assert 1 == 0, 'Invalid value of ha.'  # noqa: PLR0133
            ax.annotate(xy=(x + xadj, y + yadj), text=text, ha=ha)

    return res


def plot_map(
    dat,
    dat_proj=ccrs.PlateCarree(),
    disp_proj=ccrs.PlateCarree(),
    figsize=(12, 8),
    grid=True,
    ncols=1,
    nrows=1,
    title=None,
    share_scale=False,
    colour_scale=None,
    cbar_ticks=None,
    tick_labels=None,
    file=None,
    scale_label='',
    share_axes=False,
    ticks_left=True,
    ticks_bottom=True,
    wspace=0.05,
    hspace=0.05,
    stippling=None,
    cbar_adjust=0.862,
    cbar_pad=0.015,
    col_labels=None,
    row_labels=None,
    xlims=None,
    ylims=None,
    show=True,
    shared_scale_quantiles=(0, 1),
    row_label_rotation=90,
    row_label_scale=1.33,
    row_label_offset=0.03,
    row_label_adjust=0.02,
    letter_labels=False,
    cbar_extend='neither',
    title_inset_pos='upper',
    **kwargs,
):
    """Plot data on a map.

    Arguments:
        dat: DataSet to plot or list of datasets to plot.
        dat_proj, dist_proj: Data and display projections.
        figsize: Figure size width x height.
        grid: Show grid?
        ncols/nrows: Number of columns/rows to plot.
        title: Title(s) for the plot(s).
        share_scale: Make the range of values in each plot the same?
        colour_scale: Tuple with min/max values to use on scale. Overwritten by
                      share_scale.
        cbar_ticks: Ticks for the colourbar.
        tick_labels: Colour bar tick labels.
        file: If specified save to 'file' instead of showing onscreen.
        scale_label: The label for a shared scale.
        share_axes: Share left/bottom axes?
        ticks_left, ticks_bottom: Display ticks on the left/bottom of plots?
        wspace, hspace: gridspec wspace and hspace arguments.
        stippling: Stippling per axis.
        cbar_adjust: Amount to shrink plots by to add cbar for shared scale.
        cbar_pad: Padding between figure and colour bar.
        col_labels/row_labels: Labels for each column/row; overwrites individial
                               plot titles.
        xlims: x limits.
        ylims: y limits.
        show: Show the map?
        row_label_rotation: Rotation for row labels.
        row_label_scale: Scale factor for gap between row labels.
        row_label_offset: Offset for first row label.
        row_label_adjust: Adjust setting for row label space on left of plot.
        letter_labels: Use a letter to label each subplot?
        cbar_extend: Extend argument to colorbar().
        col_labels: Labels for each column.
        dat_proj: Data projection.
        disp_proj: Display projection.
        nrows: Number of rows.
        ncols: Number of coluns.
        row_labels: Labels for each row.
        hspace: Vertical spacing between facets.
        wspace: Horizontal spacing between facets.
        shared_scale_quantiles: Quantiles to use in the shared scale.
        ticks_bottom: Display bottom ticks?
        ticks_left: Display left ticks?
        kwargs: Extra arguments to plot_map_to_ax.

    Return: The axis plotted to.

    """
    fig, ax = plt.subplots(
        figsize=figsize,
        ncols=ncols,
        nrows=nrows,
        subplot_kw={'projection': disp_proj},
        gridspec_kw={'wspace': wspace, 'hspace': hspace},
    )
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)

    letters = [
        'a)',
        'b)',
        'c)',
        'd)',
        'e)',
        'f)',
        'g)',
        'h)',
        'i)',
        'j)',
        'k)',
        'l)',
        'm)',
        'n)',
        'o)',
        'p)',
        'q)',
        'r)',
        's)',
        't)',
        'u)',
        'v)',
        'w)',
        'x)',
        'y)',
        'z)',
    ]

    if not isinstance(dat, list):
        im = plot_map_to_ax(
            dat=dat,
            ax=ax,
            grid=grid,
            dat_proj=dat_proj,
            disp_proj=disp_proj,
            title=title,
            stippling=stippling,
            colour_scale=colour_scale,
            cbar_ticks=cbar_ticks,
            cbar_pad=cbar_pad,
            tick_labels=tick_labels,
            ticks_left=ticks_left,
            ticks_bottom=ticks_bottom,
            xlims=xlims,
            ylims=ylims,
            cbar_extend=cbar_extend,
            **kwargs,
        )
    else:
        assert ncols * nrows >= len(dat), 'Not enough cols/rows to fit all plots.'

        if share_scale and colour_scale is None:
            all_vals = np.array([])

            for d in dat:
                all_vals = np.concatenate([all_vals, np.array(d.values.flat)])

            colour_scale = (
                np.nanquantile(all_vals, shared_scale_quantiles[0]),
                np.nanquantile(all_vals, shared_scale_quantiles[1]),
            )
            assert not (np.isnan(colour_scale[0]) or np.isnan(colour_scale[1])), (
                'share_scale cannot be used with subplots missing data.'
            )

        for i, d in enumerate(dat):
            ax_title = None
            if title is not None:
                ax_title = title[i]

            tb = ticks_bottom
            tl = ticks_left
            if share_axes:
                if i < (ncols * nrows) - ncols:
                    tb = False
                if i % ncols != 0:
                    tl = False

            stipple = None if stippling is None else stippling[i]
            proj = dat_proj if not isinstance(dat_proj, list) else dat_proj[i]
            xlim = xlims if not isinstance(xlims, list) else xlims[i]
            ylim = ylims if not isinstance(ylims, list) else ylims[i]

            left_title = None
            if letter_labels:
                left_title = letters.pop(0)

            im = plot_map_to_ax(
                dat=d,
                ax=ax.flat[i],
                grid=grid,
                dat_proj=proj,
                disp_proj=disp_proj,
                title=ax_title,
                colour_scale=colour_scale[i] if isinstance(colour_scale, list) else colour_scale,
                cbar_extend=cbar_extend[i] if isinstance(cbar_extend, list) else cbar_extend,
                cbar_pad=cbar_pad,
                cbar_ticks=cbar_ticks,
                tick_labels=tick_labels,
                colourbar=(not share_scale),
                stippling=stipple,
                xlims=xlim,
                ylims=ylim,
                ticks_left=tl,
                ticks_bottom=tb,
                left_title=left_title,
                title_inset_pos=title_inset_pos[i] if isinstance(title_inset_pos, list) else title_inset_pos,
                **kwargs,
            )

        while i + 1 < len(ax.flat):
            fig.delaxes(ax.flat[i + 1])
            i = i + 1

        if share_scale:
            fig.subplots_adjust(right=cbar_adjust)
            cbar_ax = fig.add_axes([cbar_adjust + cbar_pad, 0.23, 0.02, 0.55])
            fmt = mticker.ScalarFormatter(useOffset=False, useMathText=True)
            fmt.set_powerlimits((-4, 6))
            cb = fig.colorbar(
                im,
                ax=ax,
                cax=cbar_ax,
                ticks=cbar_ticks,
                label=scale_label,
                format=fmt,
                extend=cbar_extend,
            )
            if tick_labels is not None:
                assert len(tick_labels) == len(cbar_ticks), (
                    'Labels and ticks must have same length'
                )
                cb.ax.set_yticklabels(tick_labels)

        if col_labels is not None or row_labels is not None:
            for a in ax.flat:
                a.set_title('')

            if col_labels is not None:
                axes = ax if ax.ndim == 1 else ax[0, :]
                for a, lab in zip(axes, col_labels):
                    a.set_title(lab, fontsize=plt.rcParams['font.size'])

            if row_labels is not None:
                fig.subplots_adjust(left=row_label_adjust)
                lab_ax = fig.add_axes(
                    [0, 0.11, row_label_adjust, 0.78],
                    autoscale_on=True,
                )
                lab_ax.axis('off')

                for i, lab in enumerate(row_labels):
                    p = row_label_offset + (i / len(row_labels)) * row_label_scale
                    lab_ax.annotate(
                        lab,
                        xy=(0.5, 1 - p),
                        rotation=row_label_rotation,
                        xycoords='axes fraction',
                        ha='center',
                        va='center',
                    )

    if file is not None:
        plt.savefig(fname=file, bbox_inches='tight')

        if show:
            plt.show()
        plt.close()
    elif show:
        plt.show()

    return ax


def annual_stats(
    d,
    factor,
    day_vars=proxies,
    mean_vars=None,
    quantile_vars=None,
    chunks=None,
    time_chunk=500,
):
    """Calculate annual statistics for a dataset.

    Arguments:
        d: Data on which to calculate statistics.
        factor: The factor by which to multiply the mean number of days
                (for 'day_vars' variables) to get the number of days per period. Eg.
                365 for annual, 91 for seasonal.
        day_vars: Variables for which the statistic is mean number of days per year.
                  Expected to be boolean or binary.
        mean_vars: Variables for which to calculate the mean per year.
        quantile_vars: {percentile: variables} dictionary showing which percentiles to
                       use for extreme values of each variable.
        chunks: Chunks to use for quantile calculation.
        time_chunk: Chunk for times in daily timeseries.

    Returns: statistics per year.

    """
    if mean_vars is None:
        mean_vars = [
            'mixed_100_cape',
            'mixed_100_cin',
            'mixed_100_lifted_index',
            'lapse_rate_700_500',
            'temp_500',
            'melting_level',
            'shear_magnitude',
        ]
    if quantile_vars is None:
        quantile_vars = {
            0.01: ['mixed_100_cin', 'mixed_100_lifted_index', 'lapse_rate_700_500'],
            0.99: ['mixed_100_cape', 'temp_500', 'melting_level', 'shear_magnitude'],
        }
    if chunks is None:
        chunks = {'time': -1, 'lat': 15, 'lon': 15}

    # Annual hail-prone days.
    daily = (
        d[day_vars].resample(time='D').max(keep_attrs=True).chunk({'time': time_chunk})
    )

    # Remove all 29 Februaries, because resample can introduce one if it is
    # missing, which is set to nan; but bool(np.nan) == True so it introduces an
    # erroneous True value on that day.
    daily = daily.where(
        ~np.logical_and(daily.time.dt.month == 2, daily.time.dt.day == 29),  # noqa: PLR2004
        drop=True,
    )
    with xarray.set_options(keep_attrs=True):
        days = daily.groupby('time.year').mean(keep_attrs=True) * factor
        days = days.chunk(-1)

    # Annual ingredient means.
    means = xarray.Dataset()
    if mean_vars is not None:
        means = d[mean_vars].groupby('time.year').mean(keep_attrs=True)
        for v in mean_vars:
            means = means.rename({v: f'mean_{v}'})

    # Annual ingredient extremes.
    extremes = xarray.Dataset()
    if quantile_vars is not None:
        for q in quantile_vars:
            for v in quantile_vars[q]:
                quant_dat = d[v].chunk(chunks)
                quant_dat = quant_dat.persist()
                quants = (
                    quant_dat.groupby('time.year')
                    .quantile(q, keep_attrs=True)
                    .drop('quantile')
                    .load()
                )
                extremes = xarray.merge([extremes, quants])
                extremes[v].attrs['description'] = f'Percentile {q}'
                extremes = extremes.rename({v: f'extreme_{v}'})

    return xarray.merge([days, means, extremes])


def epoch_stats(
    d,
    season_factors=None,
    proxy_vars=proxies,
    proxy_names=proxy_names,
):
    """Calculate annual and seasonal statistics.

    Arguments:
        d: Data to work on.
        season_factors: Factor argument (length in days) for each season.
        proxy_vars: Proxy variables.
        proxy_names: a var: name dictionary giving variable names.

    Returns: A single xarray object with annual and seasonal stats.

    """
    if season_factors is None:
        season_factors = {'DJF': 90, 'MAM': 92, 'JJA': 92, 'SON': 91}

    print('Hail days per month...')
    prox = d.reset_coords()[proxy_vars].load()

    # Resample to daily true/false hail-prone conditions per location.
    daily = prox.resample(time='D').max(keep_attrs=True)

    # Remove all 29 Februaries, because resample can introduce one if it is
    # missing, which is set to nan; but bool(np.nan) == True so it introduces an
    # erroneous True value on that day.
    daily = daily.where(
        ~np.logical_and(daily.time.dt.month == 2, daily.time.dt.day == 29),  # noqa: PLR2004
        drop=True,
    )

    # Resample to number of hail-prone days per month.
    days_per_month = daily.resample(time='M').sum(keep_attrs=True)
    days_per_month['year'] = days_per_month.time.dt.year
    days_per_month['month'] = days_per_month.time.dt.month
    days_per_month = days_per_month.set_index(time=['year', 'month']).unstack('time')
    for v in proxy_vars:
        days_per_month[v].attrs['long_name'] = (
            f'Hail-prone days during month ({proxy_names[v]})'
        )
        days_per_month[v].attrs['units'] = 'days per month'

    print('Seasonal...')
    seasonal = []
    for s in ['DJF', 'MAM', 'JJA', 'SON']:
        print(f'{s}...')
        seasonal.append(
            annual_stats(
                d.where(d.time.dt.season == s),
                factor=season_factors[s],
            ).expand_dims({'season': [s]}),
        )
    seasonal = xarray.combine_nested(
        seasonal,
        concat_dim='season',
        combine_attrs='no_conflicts',
    )
    seasonal = seasonal.rename({n: f'seasonal_{n}' for n in seasonal.data_vars})
    seasonal = seasonal.chunk(-1)

    print('Annual...')
    annual = annual_stats(d=d, factor=365)
    annual = annual.rename({n: f'annual_{n}' for n in annual.data_vars})
    annual = annual.chunk(-1)

    return xarray.merge([seasonal, annual, days_per_month])


def process_epoch(
    epoch_name,
    model_name,
    exp,
    epoch_dates,
    expected_times=365 * 20 * 4,
    data_dir='/g/data/up6/tr2908/future_hail_global/CMIP_conv/',
    prox_dir='/g/data/up6/tr2908/future_hail_global/CMIP_proxies/',
    non_na_vars=proxies,
):
    """Process an epoch for a given simulation.

    Arguments:
        epoch_name: Name of the epoch.
        model_name: The model name.
        exp: Name of the experiment (e.g. historical/ssp585).
        epoch_dates: The epoch start/end times as a tuple.
        expected_times: Number of times expected in the epoch.
        data_dir: Base data directory.
        prox_dir: Directory for proxy data.
        non_na_vars: A variable to check there are no NAs in.

    """
    sim_dir = f'{model_name}/{exp}'
    files = glob(f'{data_dir}/{sim_dir}/*.nc')
    if len(files) == 0:
        print(f'No files available for {sim_dir}.')
        return None

    # Open convective data.
    dat = xarray.open_mfdataset(
        sorted(files),
        parallel=True,
        chunks={'time': 500},
        combine='nested',
        concat_dim='time',
    ).sortby('time')

    # Open separately processed proxy data.
    prox_file = f'{prox_dir}/{model_name}_{exp}_proxies.nc'
    prox = xarray.open_dataset(prox_file).sortby('time')
    dat = xarray.merge([dat, prox])

    dat = dat.sel(time=slice(f'{epoch_dates[0]}-01-01', f'{epoch_dates[1]}-12-31'))

    # Convert from longitude 0-360 to -180-180.
    lon_attrs = dat.lon.attrs
    dat['lon'] = dat.lon.where(dat.lon < 180, other=dat.lon - 360)  # noqa: PLR2004
    dat = dat.sortby('lon')
    dat.lon.attrs = lon_attrs

    if dat.time.size != expected_times:
        msg = (
            f'Warning: {sim_dir} epoch {epoch_name}: number of times returned '
            f'({dat.time.size}) does not match expected size ({expected_times}).'
        )
        print(msg)
        return None

    # Check no NAs in non_na_var.
    for v in non_na_vars:
        assert not np.any(np.isnan(dat))[v], f'NaN found in {non_na_vars}'

    # Get annual/seasonal stats for the epoch.
    stats = epoch_stats(d=dat)
    stats = stats.expand_dims({'model': [model_name], 'epoch': [epoch_name]})
    stats.attrs['epoch_dates'] = f'{epoch_dates[0]}-{epoch_dates[1]}'

    return stats


def make_landsea_mask(
    lsm_file='/g/data/oi10/replicas/CMIP6/CMIP/MRI/MRI-ESM2-0/historical/r1i1p1f1/fx/sftlf/gn/v20190603/sftlf_fx_MRI-ESM2-0_historical_r1i1p1f1_gn.nc',
    out_res=1,
    percent_land_required=25,
):
    """Calculate a land-sea mask based on a CMIP dataset's land-sea mask.

    Arguments:
        lsm_file: The CMIP land-sea mask file to open and regrid.
        out_res: Output resolution to grid to (in degrees).
        percent_land_required: How much of a grid cell needs to be considered land to be
                               a "land" pixel in the returned map?

    Returns: A land-sea mask.

    """
    lsm = xarray.open_mfdataset(lsm_file).load()

    # Regrid to output resolution.
    out_lon = np.arange(-180.0, 180.0 + out_res, out_res)
    out_lat = np.arange(-90, 90 + out_res, out_res)
    out_grid = xarray.Dataset(
        coords={
            'lat': (out_lat[:-1] + out_lat[1:]) / 2,
            'lon': (out_lon[:-1] + out_lon[1:]) / 2,
            'lat_b': out_lat,
            'lon_b': out_lon,
        },
    )

    regridder = xe.Regridder(lsm, out_grid, 'bilinear', periodic=True)
    lsm = regridder(lsm)

    lsm['land'] = lsm.sftlf > percent_land_required
    lsm = lsm.drop('sftlf')
    lsm.land.attrs['long_name'] = 'Land mask'
    lsm.land.attrs['description'] = f'Defined as sftlf > 50 in {lsm_file}.'

    return lsm.land.load()


def plot_seasonal_maps(
    dat,
    variable,
    lat_range=slice(None, None),
    lon_range=slice(None, None),
    figsize=(12, 8),
    cmap=hail_cmap,
    **kwargs,
):
    """Plot maps per model and per season - rows per model and columns per season.

    Argumnets:
        d: The data to plot.
        variable: Variable to plot.
        lat/lon_range: Ranges of latitude/longitude.
        figsize: Figure width x height.
        kwargs: Extra arguments to plot_map.

    """
    d = [
        dat[variable]
        .mean('year_num')
        .sel(model=m, season=s, lat=lat_range, lon=lon_range)
        for m, s in itertools.product(dat.model.values, dat.season.values)
    ]
    titles = [
        f'{m} ({s})' for m, s in itertools.product(dat.model.values, dat.season.values)
    ]

    _ = plot_map(
        d,
        title=titles,
        figsize=figsize,
        nan_colour='white',
        cmap=cmap,
        ncols=dat.season.size,
        nrows=dat.model.size,
        share_scale=True,
        share_axes=True,
        col_labels=dat.season.values,
        row_labels=dat.model.values,
        **kwargs,
    )


def ttest(dat, variables, fut_epoch, hist_epoch='historical', sig_level=0.05):
    """Apply Welch's t-test for across a given axis between epochs.

    Arguments:
        dat: The data to work on - for one model.
        variables: The variables to apply the t test to.
        fut_epoch: The future epoch to compare historical data to.
        hist_epoch: Name of the historical epoch.
        sig_level: The p value to require for significance.

    Returns: the t test statistic and the significance result.

    """
    res = []
    for variable in variables:
        statres, pval = sp.stats.ttest_ind(
            a=dat.sel(epoch=fut_epoch)[variable].values,
            b=dat.sel(epoch=hist_epoch)[variable].values,
            equal_var=False,
        )

        res_dims = list(dat.sel(epoch=fut_epoch)[variable].dims)[1:]
        r = xarray.Dataset(
            {
                variable + '_ttest_stat': (res_dims, statres),
                variable + '_sig': (res_dims, pval < sig_level),
            },
            coords={x: dat[x].values for x in res_dims},
        )
        res.append(r)

    res = xarray.merge(res)
    res.attrs['p-value for significance'] = sig_level
    return res


def select_all_models(dstore_files=None, **kwargs):
    """Return modules that are suitable for this work, from a list of catalogue files.

    Arguments:
        dstore_files: Tuples of (catalog_file, data_dir) to use.
        **kwargs: Extra arguments to select_models().

    Returns: A list of suitable model details.

    """
    if dstore_files is None:
        dstore_files = [
            (
                '/g/data/dk92/catalog/v2/esm/cmip6-oi10/catalog.json',
                '/g/data/oi10/replicas',
            ),
            (
                '/g/data/dk92/catalog/v2/esm/cmip6-fs38/catalog.json',
                '/g/data/fs38/publications',
            ),
        ]

    mods = []
    for [f, d] in dstore_files:
        mods.append(select_models(datastore_file=f, **kwargs))
        mods[-1]['CMIP6_dir'] = d

    return pd.concat(mods).reset_index(drop=True)


def select_models(
    datastore_file='/g/data/dk92/catalog/v2/esm/cmip6-oi10/catalog.json',
    variables=None,
    experiments=None,
    tables=None,
):
    """Return models that are suitable for use in this work.

    Arguments:
        datastore_file: The datastore file to use.
        variables: Required variables.
        experiments: Required experiments.
        tables: Required tables.

    """
    if variables is None:
        variables = ['tas', 'ta', 'uas', 'ua', 'vas', 'va', 'huss', 'hus', 'ps']
    if experiments is None:
        experiments = ['historical', 'ssp585']
    if tables is None:
        tables = ['3hr', '6hrLev']

    cmip6 = intake.open_esm_datastore(datastore_file)

    subset = cmip6.search(
        experiment_id=experiments,
        variable_id=variables,
        table_id=tables,
    )

    subset = subset.df.groupby(
        ['source_id', 'experiment_id', 'institution_id', 'member_id', 'project_id'],
    ).variable_id.unique()

    # Subset to models that have all variables required.
    subset = subset[subset.apply(len).values == len(variables)]

    # Subset to models that have both historical and ssp585 experiments.
    counts = subset.groupby(['source_id', 'member_id']).count()
    assert np.all(counts <= 2), 'More than two options for some model(s).'  # noqa: PLR2004
    counts = counts[counts == 2]  # noqa: PLR2004
    chosen_models = counts.index.values
    subset = (
        subset.reset_index().set_index(['source_id', 'member_id']).loc[chosen_models]
    )

    # Make the description string.
    subset = subset.reset_index()
    subset['desc'] = (
        'CMIP6.'
        + subset['project_id']
        + '.'
        + subset['institution_id']
        + '.'
        + subset['source_id']
        + '.'
        + subset['experiment_id']
        + '.'
        + subset['member_id']
    )

    # Tidy up.
    subset = subset.drop(columns=['variable_id', 'project_id'])
    return subset.rename(
        columns={'source_id': 'model', 'member_id': 'ensemble', 'experiment_id': 'exp'},
    )


def make_run_scripts(
    runs,
    scripts_dir='scripts/',
    template='scripts/process_CMIP_template.sh',
    yrs=9,
):
    """Make scripts to produce convective indices for each model run.

    Arguments:
        runs: As returned by define_runs().
        scripts_dir: The directory in which to write scripts.
        template: The script template to copy and modify.
        yrs: The years to use around the central year of each run.

    """
    # Make scripts for each run.
    for _, row in runs.iterrows():
        # Make scripts directory per model.
        script_dir = f'{scripts_dir}/{row.model}/'
        if not os.path.exists(script_dir):
            os.mkdir(script_dir)

        exp = row.exp.replace(' ', '_').replace('(', '').replace(')', '')

        for start in [row.start_year, row.end_year - yrs]:
            script = f'{script_dir}/process_{row.model}_{exp}'
            script = f'{script}_{start}-{start + yrs}.sh'

            # Copy the template.
            os.system(f'cp {template} {script}')

            # Adapt the template for the model.
            os.system(f'sed -i s/DESC/{row.model}_{start}-{start + yrs}/g {script}')
            os.system(f'sed -i s/MODEL/{row.desc}/g {script}')
            os.system(f'sed -i s/START/{start}/g {script}')
            os.system(f'sed -i s/END/{start + yrs}/g {script}')


def make_postprocessing_scripts(
    runs,
    scripts_dir='scripts/',
    template='scripts/post_process_CMIP_template.sh',
):
    """Make scripts to produce convective indices for each model run.

    Arguments:
        runs: As returned by define_runs().
        scripts_dir: The directory in which to write scripts.
        template: The script template to copy and modify.

    """
    # Make scripts for each run.
    for _, row in runs.iterrows():
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


def plot_run_years(runs, figsize=(10, 2.8), legend_y=9.5, file=None, show=True):
    """Make a plot to show the (overlapping) run years per model.

    Arguments:
        runs: As from define_runs().
        figsize: Figure size to use.
        legend_y: Y position for legend.
        file: Plot output file.
        show: Show the plot?

    """
    rename_exps = {
        'historical': 'Historical',
        'ssp585 (2C)': 'SSP5-8.5 (2C)',
        'ssp585 (3C)': 'SSP5-8.5 (3C)',
    }

    runs = runs.copy()
    for key, val in rename_exps.items():
        runs.loc[runs.exp == key, 'exp'] = val

    _, axs = plt.subplots(figsize=figsize, ncols=1, nrows=len(np.unique(runs.model)))

    for i, model in enumerate(np.unique(runs.model)):
        mod_runs = runs.loc[runs.model == model]
        for _, run in mod_runs.iterrows():
            axs[i].fill_betweenx(
                [0, 1],
                run.start_year,
                run.end_year + 1,
                label=run.exp,
                alpha=0.5,
                zorder=1,
            )

        if i < len(np.unique(runs.model)) - 1:
            axs[i].set_xticks([])
        axs[i].set_yticks([])
        axs[i].set_xlim(1980, 2100)
        axs[i].set_ylabel(model, rotation=0, ha='right', va='center')

    plt.legend(loc='upper right', bbox_to_anchor=(1.15, legend_y), framealpha=1)

    if file is not None:
        plt.savefig(fname=file, bbox_inches='tight')

        if show:
            plt.show()
        plt.close()
    elif show:
        plt.show()


def warming_years(future_models, warming_degrees=2, baseline_range=None):
    """Calculate warming periods.

    For each model, determine the 20 year period around when that model
    reaches a certain warming level over pre-industrial temperatures.

    Arguments:
        future_models: A list of full model descriptions to find years for.
        warming_degrees: The degrees warming to find the year range for.
        baseline_range: Baseline year range.

    Returns: pd.DataFrame with model description, start_year, end_year.

    """
    if baseline_range is None:
        baseline_range = [1980, 1999]

    # Retrieve warming data.
    warming_levels = wl.warming_window(
        warming_amount=warming_degrees,
        project='CMIP6',
        baseline_range=baseline_range,
        baseline_experiment='historical',
        future_experiment='ssp585',
    ).reset_index()

    # New dataframe for results.
    all_levels = pd.DataFrame()

    for mod in future_models:
        _, _, _, model, _, ensemble = mod.split('.')

        levels = warming_levels.loc[
            warming_levels.model == f'{model}_{ensemble}'
        ].copy()
        levels['desc'] = mod
        levels = levels.drop(columns='model')
        all_levels = pd.concat([all_levels, levels])

    return all_levels.reset_index(drop=True)


def define_runs(
    models,
    hist_start=1980,
    hist_end=1999,
    warming_degrees=None,
    tables=None,
    variables=None,
):
    """Define model runs.

    Define model runs with start/end years for the period in which they reached
    certain warming levels, and a historical period. Exclude models that don't cover
    the required times.

    Arguments:
        models: The models to use.
        hist_start: The start year of the historical period to use.
        hist_end: The end year of the historical period to use.
        warming_degrees: Number of degrees warming.
        tables: CMIP6 tables to check for temporal coverage.
        variables: CMIP6 variables to check for temporal coverage.

    Returns: A DataFrame with one row per run.

    """
    if warming_degrees is None:
        warming_degrees = [2, 3]
    if tables is None:
        tables = [['6hrLev', '6H'], ['3hr', '3H']]
    if variables is None:
        variables = ['va', 'ta', 'hus', 'ps', 'vas', 'huss', 'tas']

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
    all_h = hist
    for deg in warming_degrees:
        future = models.loc[models.exp != 'historical'].copy()
        warming_period = warming_years(
            future_models=future.desc.values,
            warming_degrees=deg,
            baseline_range=[hist_start, hist_end],
        )
        warming_period = warming_period.set_index('desc')[['start_year', 'end_year']]
        future = future.set_index('desc')
        future = future.join(warming_period).reset_index()
        future['epoch_name'] = f'{deg}C'
        future['exp_name'] = future.exp
        future['exp'] = future.exp + f' ({deg}C)'
        all_h = pd.concat([all_h, future])
    all_h = all_h.reset_index(drop=True)

    # Exclude models that are not on pressure levels or do not have the
    # temperal coverage required.
    exclude = []
    for i, run in all_h.iterrows():
        if np.isin(run.model, exclude):
            continue

        base_path = f'{run.CMIP6_dir}/{run.desc.replace(".", "/")}'

        for v in variables:
            for [p, res] in tables:  # noqa: B007
                path = f'{base_path}/{p}/{v}'
                if os.path.exists(path):
                    break
            assert os.path.exists(path), (
                f'Could not find path/res combinations for {v} under {base_path}.'
            )

            grids = [os.path.basename(x) for x in sorted(glob(f'{path}/*'))]
            assert len(grids) == 1, f'Multiple grids to choose from for {path}/{v}.'
            grid = grids[0]

            path = f'{path}/{grid}'
            version = [os.path.basename(x) for x in sorted(glob(f'{path}/v*'))][
                -1
            ]  # Use latest version.
            path = f'{path}/{version}'

            files = sorted(glob(f'{path}/*.nc'))
            min_time = times_in_CMIP_file(
                filename=files[0],
                res=res,
                calendar='standard',
            ).min()
            max_time = times_in_CMIP_file(
                filename=files[-1],
                res=res,
                calendar='standard',
            ).max()

            if min_time > pd.to_datetime(
                f'{run.start_year}-1-1',
            ) or max_time < pd.to_datetime(f'{run.end_year}-12-31'):
                print(
                    f'{run.model} {run.exp} requires {run.start_year}-{run.end_year}, '
                    f'but data for {v} covers only {min_time.year}-{max_time.year}. '
                    f'Excluding {run.model}.',
                )
                exclude.append(run.model)
                break

            # Check model is on pressure levels.
            if v == 'ta':
                ta = xarray.open_dataset(files[0])

                # While we're here, collect some metadata.
                all_h.loc[i, 'nominal_resolution'] = ta.attrs['nominal_resolution']
                all_h.loc[i, 'vertical_levels'] = len(ta.lev.values)

                sn = ta.lev.attrs['standard_name']
                ta.close()
                if sn != 'atmosphere_hybrid_sigma_pressure_coordinate':
                    print(
                        f'{run.model} {run.exp} is not on pressure levels. '
                        f'Excluding {run.model}.',
                    )
                    exclude.append(run.model)
                    break

    runs = all_h.iloc[~np.isin(all_h.model, exclude)]
    runs = runs.reset_index(drop=True)

    return runs.reset_index(drop=True)


def make_backup_orography(
    runs,
    CMIP6_dir='/g/data/oi10/replicas',
    backup_orog=(
        'CMIP6/CMIP/CNRM-CERFACS/CNRM-CM6-1/historical/'
        'r1i1p1f2/fx/orog/gr/v20180917/'
        'orog_fx_CNRM-CM6-1_historical_r1i1p1f2_gr.nc'
    ),
    backup_orog_dir='/g/data/up6/tr2908/future_hail_global/interpolated_orography',
    tables=None,
    grid_var='tas',
):
    """Make files for backup orography.

    Backup is interpolated from another CMIP6 model, for those models
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
    if tables is None:
        tables = [['6hrLev', '6H'], ['3hr', '3H']]

    orog = None
    runs['backup_orography'] = False
    for i, run in runs.iterrows():
        base_path = CMIP6_dir + '/' + run.desc.replace('.', '/')
        orog_path = base_path + '/fx/orog/'

        if not os.path.exists(orog_path):
            runs.loc[i, 'backup_orography'] = True

            _, _, _, mod, exp, ens = run.desc.split('.')
            out_file = f'{backup_orog_dir}/orog_{mod}.{exp}.{ens}.nc'
            if os.path.exists(out_file):
                continue

            print('Producing backup orography for ' + run.desc)

            # Open a variable to get the grid to interpolate to.
            for [p, _] in tables:
                path = f'{base_path}/{p}/{grid_var}'
                if os.path.exists(path):
                    break
            assert os.path.exists(path), (
                f'Could not find path/res combinations for {grid_var}.'
            )
            grids = [os.path.basename(x) for x in sorted(glob(f'{path}/*'))]
            assert len(grids) == 1, f'Multiple grids to choose from for {path}.'
            grid = grids[0]
            path = f'{path}/{grid}'
            version = [os.path.basename(x) for x in sorted(glob(f'{path}/v*'))][
                -1
            ]  # Use latest version.
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

            orog.attrs['note'] = (
                'Model orography unavailable, so orography provided by '
                + backup_orog
                + ' and regridded to model grid using xESMF.'
            )

            orog.to_netcdf(out_file)


def read_processed_data(
    data_dir='/g/data/up6/tr2908/future_hail_global/CMIP_conv_annual_stats/',
    data_exp='*common_grid.nc',
    rename_models=None,
    rename_vars=None,
    apply_landmask=False,
):
    """Read all processed data.

    Regrid to a common grid on the way and applying a land mask.

    Arguments:
        data_dir: The data directory to read from.
        data_exp: Expression to match for data.
        rename_models: Models to shorten the names of.
        rename_vars: Variables to rename.
        apply_landmask: Apply the landmask during processing?

    Returns: Dataset, landsea mask.

    """
    if rename_models is None:
        rename_models = {'EC-Earth_Consortium.EC-Earth3': 'EC-Earth3'}
    if rename_vars is None:
        rename_vars = {}

    # Regrid native grids to common grids.
    regrid_global(path=f'{data_dir}/*native_grid.nc')

    # Open all data.
    dat = xarray.open_mfdataset(
        f'{data_dir}/{data_exp}',
        parallel=True,
        engine='netcdf4',
    )

    # Oddly, there seems to be an issue where epochs encoded differently in
    # different input files (<U2 for '2C' and <U10 for 'historical') are, when
    # evaluated and rewritten to NetCDF, written as '<U2' so that historical
    # becomes 'hi'. Enforce all epochs being the same dtype here.
    dat = dat.assign_coords({'epoch': dat.epoch.astype(dat.epoch.dtype)})

    # Shorten model names if required.
    dat = dat.assign_coords(
        {
            'model': [rename_models.get(x, x) for x in dat.model.values],
        },
    )

    # Open landsea mask.
    lsm = make_landsea_mask()

    # Mask to land area only.
    if apply_landmask == True:
        dat = dat.where(lsm == 1)

    # Rename variables.
    dat = dat.rename(rename_vars)

    # Re-arrange the data so that the seasonal/annual/monthly hail days are
    # organised with proxy as a dimension.
    seasonal_proxies = [x for x in list(dat.keys()) if 'seasonal_proxy' in x]
    annual_proxies = [x for x in list(dat.keys()) if 'annual_proxy_' in x]
    prox = [
        x
        for x in list(dat.keys())
        if 'proxy_' in x and x not in seasonal_proxies and x not in annual_proxies
    ]
    rest = [
        x
        for x in list(dat.keys())
        if x not in seasonal_proxies and x not in annual_proxies and x not in prox
    ]

    seasonal = (
        dat[seasonal_proxies]
        .rename({f'{p}': f'{p[15:None]}' for p in seasonal_proxies})
        .to_dataarray(dim='proxy', name='seasonal_hail_days')
    )
    annual = (
        dat[annual_proxies]
        .rename({f'{p}': f'{p[13:None]}' for p in annual_proxies})
        .to_dataarray(dim='proxy', name='annual_hail_days')
    )
    prox = (
        dat[prox]
        .rename({f'{p}': f'{p[6:None]}' for p in prox})
        .to_dataarray(dim='proxy', name='monthly_hail_days')
    )

    dat = xarray.merge([dat[rest], seasonal, annual, prox])

    # Transpose - for the ttest the axis over which the t-test should be applied
    # ('year_num') must be first after model/epoch selection.
    dat = dat.transpose(
        'model',
        'epoch',
        'year_num',
        'proxy',
        'season',
        'month',
        'lat',
        'lon',
    )

    return dat, lsm


def era5_climatology_calc(era5, proxy_vars=proxies):
    """Calculate era5 climatology information.

    Argumnets:
        era5: ERA5 data ready to use.
        proxy_vars: Proxy variable names.
        proxy_names: variable: name dictionary giving proxy names.

    Returns: monthly and annual normalised hail proxy climatology.

    """
    for p in proxy_vars:
        assert not np.any(np.isnan(era5[p])), f'NaNs in ERA5 proxy {p}.'

    # Calculate daily true/false hail proxy results.
    daily = era5[proxy_vars].resample(time='D').max(keep_attrs=True)

    # Remove all 29 Februaries, because resample can introduce one if it is
    # missing, which is set to nan; but bool(np.nan) == True so it introduces an
    # erroneous True value on that day.
    daily = daily.where(
        ~np.logical_and(daily.time.dt.month == 2, daily.time.dt.day == 29),  # noqa: PLR2004
        drop=True,
    )
    with xarray.set_options(keep_attrs=True):
        days = daily.groupby('time.year').mean(keep_attrs=True) * 365
        days = days.chunk(-1)
    res = days.mean('year')

    months = daily.groupby('time.month').mean(keep_attrs=True)
    months['month_factor'] = ('month', [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31])

    for x in proxy_vars:
        res['monthly_' + x] = months[x] * months['month_factor']

    # Rename latitude/longitude for consistency with other data.
    return res.rename({'latitude': 'lat', 'longitude': 'lon'})


def era5_climatology(
    era5_dir='/g/data/up6/tr2908/future_hail_global/era5_conv/',
    era5_file_def='era5_1deg_19*.nc',
    era5_proxy_file='era5_proxies.nc',
    cache_file='/g/data/up6/tr2908/future_hail_global/era5_climatology.nc',
    landmask=None,
    rename=None,
    yrs=20,
):
    """Calculate the ERA5 climatology of mean annual hail-prone days.

    Arguments:
        era5_dir: Processed convective files directory for ERA5.
        era5_file_def: The definition of files to read in for the climatology.
        era5_proxy_file: The proxy definition file.
        cache_file: A cache file to write/read to/from (or None for no cache).
        landmask: Optionally mask for land regions using landmask.lsm.
        rename: Variables to rename.
        yrs: Years of data to expect.

    Returns: Mean annual hail-prone days from ERA5.

    """
    if rename is None:
        rename = {}

    if cache_file is None or not os.path.exists(cache_file):
        dat = xarray.open_mfdataset(f'{era5_dir}/{era5_file_def}', parallel=True)
        prox = xarray.open_dataset(f'{era5_dir}/{era5_proxy_file}')
        prox = prox.transpose('time', 'latitude', 'longitude')
        prox = prox.sel(time=dat.time)
        dat = xarray.merge([dat, prox])
        assert len(dat.time) == 365 * yrs * 4, (
            'Incorrect number of times in ERA5 historic period.'
        )

        dat = era5_climatology_calc(dat)

        if cache_file is not None:
            write_output(dat, attrs={}, file=cache_file)

    if cache_file is not None:
        dat = xarray.open_dataset(cache_file)

    if landmask is not None:
        dat = dat.where(landmask == True).load()

    # Reorganise so proxy is a dimension.
    dat = dat.rename(rename)
    monthly = [x for x in dat if 'monthly' in x]
    prox = [x for x in dat if x not in monthly]
    prox = (
        dat[prox]
        .rename({f: f[6:None] for f in prox})
        .to_dataarray(name='annual_hail_days', dim='proxy')
    )
    monthly = (
        dat[monthly]
        .rename({f: f[14:None] for f in monthly})
        .to_dataarray(name='monthly_hail_days', dim='proxy')
    )
    return xarray.merge([prox, monthly])


def monthly_era5_anoms(
    era5,
    era5_dir='/g/data/up6/tr2908/future_hail_global/era5_conv/',
    anomaly_years=None,
):
    """Calculate ERA5 hail proxy anomalies on a monthly basis.

    Take the mean over all proxies.

    Arguments:
        era5: ERA5 climatology data from era5_climatology().
        era5_dir: The ERA5 data directory.
        anomaly_years: The years for which to find anomalies.

    """
    if anomaly_years is None:
        anomaly_years = [2015, 2022]

    anoms = []
    for year in anomaly_years:
        year_dat = era5_climatology(
            era5_dir=era5_dir,
            era5_file_def=f'*{year}*.nc',
            yrs=1,
            cache_file=None,
        )
        year_dat = year_dat.sel(proxy=era5.proxy)
        year_anoms = (year_dat - era5).mean('proxy')
        year_anoms = year_anoms.expand_dims({'year': [year]})
        anoms.append(year_anoms)

    return xarray.merge(anoms).load()


def plot_era5_anomalies(
    anoms,
    year,
    lats,
    lons,
    figsize=(12, 9),
    ncols=3,
    nrows=4,
    scale_label='Hail-prone day anomaly compared to 1980-1999 climatology',
    **kwargs,
):
    """Plot maps of monthly ERA5 hail proxy anomalies over a selected region.

    Arguments:
        anoms: Anomalies to plot.
        lats: Slice of latitudes to include.
        lons: Slice of longitudes to include.
        figsize: Figure width x height.
        ncols: Number of columns.
        nrows: Number of rows.
        scale_label: Legend key.
        year: Plot for this year.
        **kwargs: Arguments to plot_map().

    """
    months = np.arange(12) + 1
    month_names = {
        1: 'January',
        2: 'February',
        3: 'March',
        4: 'April',
        5: 'May',
        6: 'June',
        7: 'July',
        8: 'August',
        9: 'September',
        10: 'October',
        11: 'November',
        12: 'December',
    }

    _ = plot_map(
        [
            anoms.sel(year=year, month=m, lat=lats, lon=lons).monthly_hail_days
            for m in months
        ],
        title=[f'{month_names[m]} {year}' for m in months],
        ncols=ncols,
        nrows=nrows,
        figsize=figsize,
        hspace=0.22,
        wspace=0.01,
        cmap='RdBu_r',
        divergent=True,
        share_scale=True,
        share_axes=True,
        grid=True,
        contour=False,
        scale_label=scale_label,
        nan_colour='white',
        **kwargs,
    )


def epoch_differences(
    dat,
    variables,
    epochs=None,
    outstring='_',
    cache_dir='/g/data/up6/tr2908/future_hail_global/CMIP_changes/',
    expand_dims=None,
):
    """Calculate differences between historical epoch and warming epochs.

    Arguments:
        dat: The data to work on.
        variables: Which variables to select.
        epochs: The future epochs to test.
        outstring: string to put in the middle of output file name.
        cache_dir: The cache directory to write changes to.
        expand_dims: Optionally pass this argument to expand_dims before saving
                     differences.

    Returns: mean differences, significances of differences, and historical mean
             for each model.

    """
    if epochs is None:
        epochs = ['2C', '3C']

    res = []
    for model in dat.model.values:
        print(f'Processing {model}...')
        out_file = f'{cache_dir}/{model}{outstring}epoch_diffs.nc'
        if not os.path.exists(out_file):
            d = dat.sel(model=model).chunk({'lat': 50, 'lon': 50})
            reference = d.sel(epoch='historical')[variables]
            reference_mean = reference.mean(['year_num']).load()

            mean_diffs = []
            sigs = []

            for e in epochs:
                # Select the epoch.
                epoch = d.sel(epoch=e)[variables]
                epoch_mean = epoch.mean(['year_num']).load()

                # Difference in means between reference and epoch.
                mean_diff = epoch_mean - reference_mean

                # Significance of differences.
                tt = ttest(dat=d, fut_epoch=e, variables=variables)

                mean_diffs.append(mean_diff.expand_dims({'epoch': [e]}))
                sigs.append(tt.expand_dims({'epoch': [e]}))

            mean_diffs = xarray.merge(mean_diffs)
            sigs = xarray.merge(sigs)

            for v in list(mean_diffs.data_vars):
                mean_diffs = mean_diffs.rename({v: f'{v}_mean_diff'})
                reference_mean = reference_mean.rename({v: f'{v}_mean_reference'})

            out = xarray.merge([mean_diffs, sigs, reference_mean])

            if expand_dims is not None:
                out = out.expand_dims(expand_dims)

            write_output(dat=out, file=out_file, attrs=out.attrs)

        res.append(xarray.open_dataset(out_file).expand_dims({'model': [model]}))

    return xarray.concat(res, dim='model')


def plot_diffs_for_epoch(
    diffs,
    epoch,
    file=None,
    var='annual_hail_days_mean_diff',
    stipple_var='annual_hail_days_sig',
    scale_label='$\Delta$ mean annual hail-prone days',
    figsize=(12, 12),
    row_label_adjust=0.16,
    row_label_scale=1.28,
    row_label_offset=-0.056,
):
    """Plot a grid with differences by proxy (columns) and model (rows).

    Arguments:
        diffs: The data to plot.
        epoch: The epoch to plot for.
        file: Output file for plot.
        var: The variable to plot differences for.
        stipple_var: The variable to use for stipples.
        scale_label: The label for the scale.
        figsize: Figure size width x height.
        row*: Arguments to plot_map() for row label positions.
        row_label_adjust: Adjust argument for label.
        row_label_scale: Scale argument for label.
        row_label_offset: Offset argument for label.

    """
    d = diffs.sel(epoch=epoch)

    _ = plot_map(
        [
            d.sel(model=m, proxy=p)[var]
            for m, p in itertools.product(d.model.values, d.proxy.values)
        ],
        stippling=[
            d.sel(model=m, proxy=p)[stipple_var]
            for m, p in itertools.product(d.model.values, d.proxy.values)
        ],
        title=[
            f'{m}, {proxy_dims[p]}'
            for m, p in itertools.product(d.model.values, d.proxy.values)
        ],
        figsize=figsize,
        disp_proj=ccrs.Robinson(),
        ncols=len(d.proxy),
        nrows=len(d.model),
        share_scale=True,
        share_axes=True,
        grid=False,
        contour=True,
        col_labels=[proxy_dims[f] for f in list(d.proxy.values)],
        row_labels=list(d.model.values),
        row_label_rotation=0,
        row_label_adjust=row_label_adjust,
        row_label_scale=row_label_scale,
        row_label_offset=row_label_offset,
        cmap='RdBu_r',
        divergent=True,
        scale_label=scale_label,
        file=file,
    )


def crop_months_array(x):
    """Calculate cropping months.

    Given a start month and an end month in x, return an array of length 12 with 1s
    where cropping occurs and 0 where it does not.

    Arguments;
        x: Should contain start and end months.

    Returns: 12-length binary array.

    """
    start = x['start']
    end = x['end']

    assert start != end, 'start == end'
    if start < end:
        months = np.arange(start, end + 1)
    else:
        months = np.concatenate([np.arange(1, end + 1), np.arange(start, 13)])

    month_array = np.repeat(0, 12)
    month_array[months - 1] = 1

    return month_array


def crop_periods(
    crop_periods_file='/g/data/up6/tr2908/future_hail_global/MIRCA2000/growing_periods_listed/CELL_SPECIFIC_CROPPING_CALENDARS_30MN.TXT.gz',
    crop_codes=None,
):
    """Open cropping periods from MIRCA data.

    Arguments:
        crop_periods_file: The MIRCA data file to open.
        crop_codes: Codes for each crop type.

    Returns: A pandas table with crop periods per latitude/longitude/crop.

    """
    if crop_codes is None:
        crop_codes = {
            1: 'Wheat',
            2: 'Maize',
            3: 'Rice',
            4: 'Barley',
            5: 'Rye',
            6: 'Millet',
            7: 'Sorghum',
            8: 'Soybeans',
            9: 'Sunflower',
            10: 'Potatoes',
            11: 'Cassava',
            12: 'Sugar cane',
            13: 'Sugar beet',
            14: 'Oil palm',
            15: 'Rapeseed / Canola',
            16: 'Groundnuts / Peanuts',
            17: 'Pulses',
            18: 'Citrus',
            19: 'Date palm',
            20: 'Grapes / Vine',
            21: 'Cotton',
            22: 'Cocoa',
            23: 'Coffee',
            24: 'Others perennial',
            25: 'Fodder grasses',
            26: 'Others annual',
        }

    crop_periods = pd.read_csv(crop_periods_file, sep='\t')

    # Irrigated crops have crop number from 1-26.
    # Rain-fed crops have crop number from 27-52.

    crop_periods['crop_type'] = str(np.nan)
    crop_periods['crop_name'] = np.nan
    crop_periods.loc[crop_periods.crop <= 26, 'crop_type'] = 'Irrigated'  # noqa: PLR2004
    crop_periods.loc[crop_periods.crop >= 27, 'crop_type'] = 'Rainfed'  # noqa: PLR2004
    crop_periods['crop_name'] = crop_periods.crop % 26
    crop_periods.loc[crop_periods.crop_name == 0, 'crop_name'] = 26
    crop_periods.crop_name = [crop_codes[x] for x in crop_periods.crop_name]

    return crop_periods


def cropping_mask(
    shape,
    cp=crop_periods(),
    crop_res=0.5,
    cache_file='/g/data/up6/tr2908/future_hail_global/CMIP_crop_stats/crop_months_mask.nc',
):
    """Make a mask of cropping months for a given dataset shape.

    Arguments:
        shape: The DataArray to use for the shape of the mask, containing
               lat/lon/month as dims.
        cp: Crop periods per location.
        crop_res: Resolution (degrees) of crop information.
        cache_file: File to use for cache.

    Returns: A DataArray the same shape as 'shape' with a new dimension 'crop',
             containing a mask for months in which cropping occurs per point per crop.

    """
    if not os.path.exists(cache_file):
        crop_mask = []
        for crop in cp.crop_name.unique():
            print(crop)
            periods = cp.loc[cp.crop_name == crop]

            # For each location, keep unique start/end months for the cropping period.
            # Remove duplicates (caused by eg. both rainfed and irrigated crops being
            # present).
            periods = periods.loc[~periods.duplicated(['lat', 'lon', 'start', 'end'])]

            # Make a list of all the cropping months for each location.
            periods['cropping_months'] = periods[['start', 'end']].apply(
                crop_months_array,
                axis=1,
            )

            # Find the union of all crop months for each location.
            crop_months = (
                periods.groupby(['lat', 'lon'])
                .cropping_months.apply(lambda x: np.amax(np.stack(x), 0))
                .reset_index()
            )

            # Convert to a geopandas object with polygons
            crop_months = gp.GeoDataFrame(
                crop_months,
                geometry=gp.points_from_xy(crop_months.lon, crop_months.lat),
            )
            crop_months = crop_months.set_geometry(
                crop_months.buffer(distance=crop_res, cap_style=3),
            )

            # Get the data grid locations as points.
            dat_grid = shape.reset_coords()[['lat', 'lon']].to_dataframe().reset_index()
            dat_grid = gp.GeoDataFrame(
                dat_grid,
                geometry=gp.points_from_xy(dat_grid.lon, dat_grid.lat),
            )

            # Find data points within each cropping region polygon and create a
            # lookup table.
            lookup = gp.sjoin(dat_grid, crop_months, how='inner')
            lookup = lookup.drop(
                columns=['geometry', 'index_right', 'lat_right', 'lon_right'],
            )
            lookup = lookup.rename(columns={'lat_left': 'lat', 'lon_left': 'lon'})

            # Generate a mask for this crop, of the same shape as the yearly data.
            mask = xarray.full_like(shape, np.nan).load()
            mask.name = 'cropping'

            # Assign mask values.
            lats = xarray.DataArray(lookup['lat'], dims='pt')
            lons = xarray.DataArray(lookup['lon'], dims='pt')
            months = xarray.DataArray(
                np.stack(lookup.cropping_months),
                dims=['pt', 'month'],
                coords={'month': np.arange(12) + 1},
            )
            mask.loc[{'lat': lats, 'lon': lons}] = months

            mask = mask.expand_dims({'crop': [crop]}).load()
            crop_mask.append(mask)

        crop_mask = xarray.concat(crop_mask, dim='crop')
        crop_mask.attrs = []
        crop_mask.attrs['long_name'] = 'Crop months mask'
        crop_mask.attrs['description'] = (
            'Derived from MIRCA2000 cropping times at 0.5 resolution, '
            'aligned with data resolution.'
        )
        crop_mask = xarray.Dataset({'cropping': crop_mask})
        write_output(dat=crop_mask, file=cache_file, attrs=None)

    res = xarray.open_dataset(cache_file).cropping
    assert res.dims[1:] == shape.dims, 'Mismatching dimensions in cached crop mask.'
    assert res.shape[1:] == shape.shape, 'Mismatching shape in cached crop mask.'
    return res


def assert_epochs(
    runs,
    data_dir='/g/data/up6/tr2908/future_hail_global/CMIP_conv_annual_stats/',
    data_exp='*common_grid.nc',
):
    """Ensure all data files have the correct epoch.

    Arguments:
        runs: Run definition from define_runs().
        data_dir: Data directory.
        data_exp: Expression to match files to read.

    """
    for file in glob(f'{data_dir}/{data_exp}'):
        d = xarray.open_dataset(file)
        line = np.logical_and(
            runs.desc == d.attrs['CMIP_model_spec'],
            runs.epoch_name == d.epoch,
        )
        assert len(runs.loc[line]) == 1, 'Multiple or no lines selected'
        assert (
            d.attrs['epoch_dates']
            == f'{runs.loc[line, "start_year"].values[0]}-{runs.loc[line, "end_year"].values[0]}'  # noqa: E501
        ), f'Incorrect epoch in {file}.'
        d.close()
        del d


def multi_model_mean_diffs(dat, variables, completion='_mean_diff', sig_thresh=0.5):
    """Calculate mean differences.

    Calculate the multi-model, multi-proxy mean difference and indicator of
    significance. A difference is considered significant if more than a certain
    proportion of the models and proxy combinations have both a) significant
    differences in the mean and b) their mean difference has the same sign as
    the multi-model mean difference.

    Arguments:
        dat: Data to work on (differences per model, significance of differences).
        variables: Variables to process.
        completion: Completion of variables to select which mean to use,
                    '_mean_rel_diff' for example.
        sig_thresh: The proportion of model/proxy combinations that have to be
                    significant and agree with sign for the result to be significant.

    Return: the mean differences, significance indicator, and mean reference
    value, for each point.

    """
    diffs = dat[[f'{x}{completion}' for x in variables]]
    sigs = dat[[f'{x}_sig' for x in variables]]
    refs = dat[[f'{x}_mean_reference' for x in variables]]

    diffs = diffs.rename({f'{x}{completion}': x for x in variables})
    sigs = sigs.rename({f'{x}_sig': x for x in variables})
    refs = refs.rename({f'{x}_mean_reference': x for x in variables})

    # Mean differences across all models.
    mean_diffs = diffs.mean(['model', 'proxy'])
    mean_sign = np.sign(mean_diffs)
    mean_refs = refs.mean(['model', 'proxy'])

    # Where does the sign of the per-model difference agree with the mean sign?
    sign_agrees = np.sign(diffs) == mean_sign

    # Where are the differences significant AND the sign matches the mean differences?
    sig = np.logical_and(sign_agrees, sigs)

    # How many models have both sig diffs and matching sign? Consider mean
    # difference significant if more than threshold of model/proxy combinations
    # have both these conditions true.
    significance = (
        sig.sum(['model', 'proxy']) >= sig.count(['model', 'proxy']) * sig_thresh
    )

    # Calculate the mean difference in relative terms (percent of reference).
    mean_diffs_rel = mean_diffs / mean_refs * 100
    mean_diffs_rel = mean_diffs_rel.where(mean_refs != 0)

    return mean_diffs, mean_diffs_rel, significance, mean_refs


def plot_relative_changes_crops(
    ch,
    cmap_colours=None,
    panelsize=(12, 4),
):
    """Plot changes in three classes (zero, increase, decrease) by crop.

    Arguments:
        ch: Crop data to plot, usually crop_hail_prone_proportion.
        cmap_colours: Colours to use in the discrete colourmap.
        panelsize: Size of the panel.

    """
    if cmap_colours is None:
        cmap_colours = ['cornflowerblue', 'greenyellow', 'indianred']

    mask = ~np.isnan(ch)
    zero_changes = xarray.full_like(other=ch, fill_value=0).where(ch == 0).where(mask)
    positive_changes = (
        xarray.full_like(other=ch, fill_value=1).where(ch > 0).where(mask)
    )
    negative_changes = (
        xarray.full_like(other=ch, fill_value=-1).where(ch < 0).where(mask)
    )

    changes = positive_changes.where(np.isnan(zero_changes), other=zero_changes)
    changes = changes.where(np.isnan(negative_changes), other=negative_changes)

    # Make a colour map to use.
    tri_cmap = LinearSegmentedColormap.from_list('', cmap_colours)
    norm = BoundaryNorm(boundaries=[-1, -0.5, 0.5, 1], ncolors=tri_cmap.N)

    for crop in changes.crop.values:
        _ = plot_map(
            [changes.sel(crop=crop, epoch=e) for e in changes.epoch.values],
            norm=norm,
            cbar_ticks=[-0.75, 0, 0.75],
            tick_labels=['Decrease', 'No change', 'Increase'],
            title=[f'{crop}, {e}' for e in changes.epoch.values],
            figsize=panelsize,
            nan_colour='white',
            grid=False,
            share_scale=True,
            nrows=1,
            ncols=2,
            disp_proj=ccrs.Robinson(),
            cmap=tri_cmap,
            divergent=True,
        )

    return changes


def crop_hail_stats(
    dat,
    cache_dir='/g/data/up6/tr2908/future_hail_global/CMIP_crop_stats/',
):
    """Calculate hail proxy stats for cropping months by location.

    Arguments:
        dat: Data to work on - should contain 'monthly_hail_days'.
        cache_dir: Directory for per-crop cache files.

    Returns: DataSet with hail prone days per cropping period per year and
    average proportion of cropping time that was hail prone, per year and
    per crop.

    """
    # Select data to analyse.
    days_per_month = dat[['monthly_hail_days']]
    days_per_month = days_per_month.chunk(
        {
            'model': 1,
            'proxy': 1,
            'epoch': 1,
            'year_num': -1,
            'month': -1,
            'lat': -1,
            'lon': -1,
        },
    )
    days_per_month['month_days'] = (
        'month',
        [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31],
    )
    days_per_month = days_per_month.rename(
        {'monthly_hail_days': 'crop_hail_prone_days'},
    )

    # Find cropping times per crop per location.
    shape = days_per_month.isel(year_num=0).crop_hail_prone_days
    crop_mask = cropping_mask(
        shape=shape,
        cache_file=f'{cache_dir}/crop_months_mask.nc',
    )
    crop_mask = crop_mask.chunk(
        {
            'crop': 1,
            'model': 1,
            'proxy': 1,
            'epoch': 1,
            'month': -1,
            'lat': -1,
            'lon': -1,
        },
    )

    # Process cropping info per crop, and cache.
    for crop in crop_mask.crop.values:
        crop_name = crop.replace(' / ', '_').replace(' ', '_').lower()
        outfile = f'{cache_dir}/crop_stats_{crop_name}.nc'

        if not os.path.exists(outfile):
            print(f'Processing {crop}...')
            d = days_per_month.where(crop_mask.sel(crop=[crop]) == 1)

            # Sum all the hail days for each year and divide by the number of days in
            # the cropping season; all per point.
            d['crop_hail_prone_proportion'] = d.crop_hail_prone_days.sum(
                'month',
            ) / d.month_days.sum('month')

            d = d.drop('month_days')
            d.crop_hail_prone_days.attrs['long_name'] = (
                'Hail-prone days during cropping period month'
            )
            d.crop_hail_prone_days.attrs['units'] = 'days per period'
            d.crop_hail_prone_proportion.attrs['long_name'] = (
                'Hail-prone proportion of cropping period'
            )
            d.crop_hail_prone_proportion.attrs['units'] = ''

            # Compute and load for this crop.
            d = d.load()
            write_output(dat=d, file=outfile, attrs=None)
            del d

    # Read cache files.
    res = xarray.open_mfdataset(f'{cache_dir}/crop_stats*.nc', parallel=True)

    # As for read_processed_data, explicitly code the crop dtype to be the right length
    # to avoid future concatenation because input files have different crop lengths.
    res = res.assign_coords({'crop': res.crop.astype(res.crop.dtype)})
    assert len(res.crop.values) == len(crop_mask.crop.values), 'Crop number mismatch.'

    return res


def plot_crop_lines(
    dat,
    diffs,
    sig,
    lat,
    lon,
    crops,
    epoch='3C',
    figsize=(12, 6.2),
    legend_renamer=None,
    crop_renamer=None,
    legend_col_length=8,
    file=None,
    average_buffer=2,
    subset_buffer=10,
):
    """Plot changes in hail-prone days.

    For a given location, plot changes in hail-prone days per month,
    cropping months, and a map of the selected point.

    Arguments:
        dat: Data to plot, must include 'monthly_hail_days' and 'crop_hail_days'.
        diffs: Differences to show on inset plot.
        sig: Significances of differences to show on inset plot.
        lat, lon: The point to examine.
        crops: Crops to select.
        figsize: Figure width x height.
        legend_renamer: Names for each element in the legend.
        crop_renamer: Rename crop names? Dict with from: to pairs.
        epoch: Epoch to plot for.
        lat: Latitude to select.
        lon: Longitude to select.
        legend_col_length: Number of rows in first legend (for epoch).
        file: Plot to file?
        average_buffer: average monthly changes over this number of lat/long degrees
                        around the location.
        subset_buffer: show this buffer around the location in inset maps.

    """
    if legend_renamer is None:
        legend_renamer = {
            'epoch': 'Epoch',
            'proxy': 'Proxy',
            '2C': '2 $^{\circ}$C',
            '3C': '3 $^{\circ}$C',
            'Raupach2023_updated': 'Raupach',
            'Raupach2023_updated_noconds': 'RaupachNEC',
            'SHIP_0.1': 'SHIP',
            'Eccel2012': 'Eccel',
        }
    if crop_renamer is None:
        crop_renamer = {
            'Fodder grasses': '  Fodder',
            'Groundnuts / Peanuts': 'Peanuts',
            'Barley': '  Barley',
        }

    line_dat = dat.monthly_hail_days.chunk(
        {'model': -1, 'epoch': -1, 'year_num': -1, 'proxy': -1, 'month': -1},
    ).sel(
        lat=slice(lat - average_buffer, lat + average_buffer),
        lon=slice(lon - average_buffer, lon + average_buffer),
        epoch=['historical', epoch],
    )
    subset_dat = xarray.Dataset(
        {
            'hpp': diffs.crop_hail_prone_proportion,
            'sig': sig.crop_hail_prone_proportion,
        },
    )
    subset_dat = subset_dat.chunk({'epoch': 1, 'crop': 1})
    subset_dat = subset_dat.sel(
        lat=slice(lat - subset_buffer, lat + subset_buffer),
        lon=slice(lon - subset_buffer, lon + subset_buffer),
        epoch=epoch,
    )
    crop_dat = (
        dat.crop_hail_prone_days.chunk(
            {
                'model': -1,
                'epoch': -1,
                'year_num': -1,
                'crop': 1,
                'proxy': -1,
                'month': -1,
            },
        )
        .sel(lat=lat, lon=lon, method='nearest')
        .sel(crop=crops, epoch=['historical', epoch])
    )

    line_dat = line_dat.mean(['year_num', 'lat', 'lon']).load()
    crop_dat = crop_dat.mean(['year_num', 'model', 'proxy']).load()

    diffs = line_dat - line_dat.sel(epoch='historical')
    diffs = diffs.drop_sel(epoch='historical')
    diffs = diffs.to_dataframe().reset_index()

    fig = plt.figure(figsize=figsize, constrained_layout=False)
    gs = fig.add_gridspec(nrows=6, ncols=5)

    months_ax = fig.add_subplot(gs[0, 0:4])
    line_ax = fig.add_subplot(gs[1:5, 0:4])
    subset_axes = [
        fig.add_subplot(gs[0:2, 4], projection=ccrs.PlateCarree()),
        fig.add_subplot(gs[2:4, 4], projection=ccrs.PlateCarree()),
        fig.add_subplot(gs[4:6, 4], projection=ccrs.PlateCarree()),
    ]

    for i, crop in enumerate(crops):
        c = crop_dat.sel(crop=crop).isel(epoch=0)
        c = c.where(np.isnan(c), other=i)

        c.name = 'crop_month'
        sns.pointplot(
            c.to_dataframe(),
            x='month',
            y='crop_month',
            ax=months_ax,
            marker='x',
            color='black',
            markersize=5,
            linewidth=1.5,
        )

    # Top row: lines showing cropping months.
    months_ax.set_frame_on(False)
    months_ax.set_yticks([0, 1, 2], [crop_renamer.get(x, x) for x in crops])
    months_ax.set_ylabel('')
    months_ax.set_xlabel('')
    months_ax.set_ylim(-0.5, 2.5)
    months_ax.tick_params(axis='x', bottom=False, labelbottom=False)
    months_ax.tick_params(axis='y', right=False)

    # Bottom rows: lines showing monthly changes in hail-prone days.
    line_ax.set_ylabel('$\Delta$ Hail-prone days')
    line_ax.set_xlabel('Month')
    line_ax.spines[['right', 'top']].set_visible(False)
    line_ax.axhline(y=0, color='black')
    sns.pointplot(
        data=diffs,
        ax=line_ax,
        x='month',
        y='monthly_hail_days',
        err_kws={'linewidth': 1.75},
        linewidth=1.2,
        hue='proxy',
        dodge=0.3,
        errorbar=('pi', 100),
    )
    h, labs = line_ax.get_legend_handles_labels()
    labs = [legend_renamer[lab] for lab in labs]
    line_ax.legend(h[0:legend_col_length], labs[0:legend_col_length], frameon=False)
    sns.move_legend(line_ax, 'lower center', ncol=4, bbox_to_anchor=(0.5, -0.4))

    # Right plots: crop changes spatially.
    for i, crop in enumerate(crops):
        rect = Rectangle(
            (lon - average_buffer, lat - average_buffer),
            width=average_buffer * 2,
            height=average_buffer * 2,
            edgecolor='fuchsia',
            facecolor='none',
            linewidth=2,
        )

        plot_map_to_ax(
            dat=subset_dat.sel(crop=crop).hpp,
            stippling=subset_dat.sel(crop=crop).sig,
            ax=subset_axes[i],
            grid=True,
            cmap='RdBu_r',
            divergent=True,
            cbar_label='$\Delta$ HPP [%]',
            title=crop,
            cbar_aspect=15,
            cbar_shrink=0.65,
            cbar_pad=0.03,
            ticks_bottom=i == len(crops) - 1,
        )
        subset_axes[i].add_patch(rect)

    assert np.max(crop_dat.diff('crop')) == 0, (
        f'Errant differences between crop values with diff  '
        f'of {np.max(crop_dat.diff("crop"))}'
        '(if nan then perhaps crop is not at this location)'
    )

    plt.subplots_adjust(hspace=0.5, wspace=0.7)
    if file is not None:
        plt.savefig(fname=file, bbox_inches='tight')
    plt.show()


def conv_properties(dat, vert_dim='model_level_number'):
    """Calculate convective properties needed in this work.

    Arguments:
        dat: An xarray Dataset containing pressure, temperature, and
             specific humidity, wind data, and height ('height' for all
             variables except wind, 'wind_height' for wind levels).
        vert_dim: The name of the vertical dimension in the dataset.

    Returns: Dataset containing convection properties for each point.

    """
    print('Calculating dewpoint...')
    dat['dewpoint'] = metpy.calc.dewpoint_from_specific_humidity(
        pressure=dat.pressure,
        temperature=dat.temperature,
        specific_humidity=dat.specific_humidity,
    )
    dat['dewpoint'] = dat.dewpoint.metpy.convert_units('K')
    dat['dewpoint'] = dat.dewpoint.metpy.dequantify()

    print('Calculating mixed-parcel CAPE and CIN (100 hPa)...')
    mixed_cape_cin_100, mixed_profile_100, _ = parcel.mixed_layer_cape_cin(
        pressure=dat.pressure,
        temperature=dat.temperature,
        dewpoint=dat.dewpoint,
        vert_dim=vert_dim,
        depth=100,
        prefix='mixed_100',
    )

    print('Calculating lifted indices...')
    mixed_li_100 = parcel.lifted_index(
        profile=mixed_profile_100,
        vert_dim=vert_dim,
        prefix='mixed_100',
        description=('Lifted index using fully-mixed ' + 'lowest 100 hPa parcel.'),
    )

    print('700-500 hPa lapse rate...')
    lapse = parcel.lapse_rate(
        pressure=dat.pressure,
        temperature=dat.temperature,
        height=dat.height_asl,
        vert_dim=vert_dim,
    )
    lapse.name = 'lapse_rate_700_500'

    print('Temperature at 500 hPa...')
    temp_500 = parcel.isobar_temperature(
        pressure=dat.pressure,
        temperature=dat.temperature,
        isobar=500,
        vert_dim=vert_dim,
    )
    temp_500.name = 'temp_500'

    print('Freezing level height...')
    flh = parcel.freezing_level_height(
        temperature=dat.temperature,
        height=dat.height_asl,
        vert_dim=vert_dim,
    )

    print('Melting level height...')
    mlh, _ = parcel.melting_level_height(
        pressure=dat.pressure,
        temperature=dat.temperature,
        dewpoint=dat.dewpoint,
        height=dat.height_asl,
        vert_dim=vert_dim,
    )

    print('0-6 km vertical wind shear...')
    shear = parcel.wind_shear(
        surface_wind_u=dat.surface_wind_u,
        surface_wind_v=dat.surface_wind_v,
        wind_u=dat.wind_u,
        wind_v=dat.wind_v,
        height=dat.wind_height_above_surface,
        shear_height=6000,
        vert_dim=vert_dim,
    )

    print('Calculating most-unstable CAPE and CIN...')
    mu_cape_cin, _, mu_parcel = parcel.most_unstable_cape_cin(
        pressure=dat.pressure,
        temperature=dat.temperature,
        dewpoint=dat.dewpoint,
        vert_dim=vert_dim,
        depth=250,
        prefix='mu',
    )

    print('Calculating deep convective indices...')
    mixed_dci_100 = parcel.deep_convective_index(
        pressure=dat.pressure,
        temperature=dat.temperature,
        dewpoint=dat.dewpoint,
        lifted_index=mixed_li_100.mixed_100_lifted_index,
        vert_dim=vert_dim,
        prefix='mixed_100',
        description=(
            'Deep convective index using fully-mixed ' + 'lowest 100 hPa parcel.'
        ),
    )

    print('Calculating mixing ratio of most unstable parcel...')
    mu_mixing_ratio = metpy.calc.mixing_ratio_from_specific_humidity(
        specific_humidity=metpy.calc.specific_humidity_from_dewpoint(
            pressure=mu_parcel.pressure * units.hPa,
            dewpoint=mu_parcel.dewpoint * units.K,
        ),
    ).metpy.dequantify()
    mu_mixing_ratio.attrs['long_name'] = 'Mixing ratio'
    mu_mixing_ratio.attrs['description'] = 'Mixing ratio of most unstable parcel'
    mu_mixing_ratio.name = 'mu_mixing_ratio'

    print('Merging results...')
    return xarray.merge(
        [
            mixed_cape_cin_100,
            mu_cape_cin,
            mu_mixing_ratio,
            mixed_li_100,
            lapse,
            temp_500,
            flh,
            mlh,
            shear,
            mixed_dci_100,
        ],
    )


def storm_proxies(
    dat,
    check_negative_cape=False,
    proxies=None,
):
    """Calculate storm proxies.

    Arguments:
        dat: Data with convective properties.
        check_negative_cape: Check for negative cape values? (Default: False).
        proxies: A list of proxies to compute and which study they are from.

    Returns: DataSet with proxy values (binary, 1=proxy triggered, 0=proxy untriggered).

    """
    if proxies is None:
        proxies = {
            'proxy_Kunz2007': 'Kunz 2007',
            'proxy_Eccel2012': 'Eccel 2012',
            'proxy_Mohr2013': 'Mohr 2013',
            'proxy_SHIP_0.1': 'SHIP > 0.1',
            'proxy_SHIP_0.5': 'SHIP > 0.5',
        }

    # Ignore negative CAPE.
    dat = dat.rename({'shear_magnitude': 'S06'})

    if check_negative_cape:
        assert np.all(dat.mixed_100_cape >= 0), 'Negative CAPE found.'
        assert np.all(dat.mu_cape >= 0), 'Negative MUCAPE found.'

    out = xarray.Dataset()

    # Proxy calculations.

    # Kunz 2007.
    if 'proxy_Kunz2007' in proxies:
        out['proxy_Kunz2007'] = np.logical_or(
            dat.mixed_100_lifted_index <= -2.07,  # noqa: PLR2004
            np.logical_or(dat.mu_cape >= 1474, dat.mixed_100_dci >= 25.7),  # noqa: PLR2004
        )

    # Eccel 2012.
    if 'proxy_Eccel2012' in proxies:
        out['proxy_Eccel2012'] = np.logical_and(
            dat.mixed_100_cape * dat.S06 > 10000,  # noqa: PLR2004
            dat.mixed_100_cin > -50,  # noqa: PLR2004
        )

    # Mohr 2013.
    if 'proxy_Mohr2013' in proxies:
        out['proxy_Mohr2013'] = np.logical_or(
            dat.mixed_100_lifted_index <= -1.6,  # noqa: PLR2004
            dat.mixed_100_cape >= 439,  # noqa: PLR2004
        )
        out['proxy_Mohr2013'] = np.logical_or(
            out.proxy_Mohr2013,
            dat.mixed_100_dci >= 26.4,  # noqa: PLR2004
        )

    # Significant hail parameter.
    if 'proxy_SHIP_0.1' in proxies or 'proxy_SHIP_0.5' in proxies:
        out['ship'] = parcel.significant_hail_parameter(
            mucape=dat.mu_cape,
            mixing_ratio=dat.mu_mixing_ratio,
            lapse=dat.lapse_rate_700_500,
            temp_500=dat.temp_500,
            shear=dat.S06,
            flh=dat.freezing_level,
        )
        if 'proxy_SHIP_0.1' in proxies:
            out['proxy_SHIP_0.1'] = out.ship > 0.1  # noqa: PLR2004
        if 'proxy_SHIP_0.5' in proxies:
            out['proxy_SHIP_0.5'] = out.ship > 0.5  # noqa: PLR2004
        out.ship.attrs['long_name'] = 'Significant hail parameter (SHIP)'

    for proxy, val in proxies.items():
        out[proxy].attrs['long_name'] = 'Proxy ' + val

    return out


def plot_regional_crop_changes(
    diffs,
    sig,
    lats,
    lons,
    region_names,
    file,
    figsize=(13.6, 13.3),
):
    """Plot box plots of significant crop changes by region.

    Arguments:
        diffs: The differences to plot.
        sig: Significance of differences.
        lats: Regional latitude bounds.
        lons: Regional longitude bounds.
        region_names: Title names per region.
        file: Output file.
        figsize: Output figure size.

    """
    res = []
    for region in lats.keys():  # noqa: SIM118
        for e in diffs.epoch.values:
            d = diffs.crop_hail_prone_proportion.sel(
                epoch=e,
                lat=lats[region],
                lon=lons[region],
            )
            s = sig.crop_hail_prone_proportion.sel(
                epoch=e,
                lat=lats[region],
                lon=lons[region],
            )
            d = d.where(s == True)
            d = (
                d.expand_dims({'region': [region_names[region]]})
                .to_dataframe()
                .reset_index()
            )
            res.append(d)
    res = pd.concat(res).reset_index(drop=True)
    del d, s

    letters = [
        'a)',
        'b)',
        'c)',
        'd)',
        'e)',
        'f)',
        'g)',
        'h)',
        'i)',
        'j)',
        'k)',
        'l)',
        'm)',
        'n)',
        'o)',
        'p)',
        'q)',
        'r)',
        's)',
        't)',
        'u)',
        'v)',
        'w)',
        'x)',
        'y)',
        'z)',
    ]
    rename_leg = {'2C': '2 $^{\circ}$C', '3C': '3 $^{\circ}$C'}

    regions = sorted(np.unique(res.region.values))
    print(regions)
    if 'global' in regions:
        regions = ['global'] + [x for x in regions if x != 'global']

    _, axs = plt.subplots(
        nrows=len(regions),
        figsize=figsize,
        gridspec_kw={'hspace': 0.35},
    )
    for i, r in enumerate(regions):
        d = res[res.region == r]
        d = d.sort_values('crop').reset_index()
        sns.boxplot(
            data=d,
            x='crop',
            y='crop_hail_prone_proportion',
            hue='epoch',
            ax=axs[i],
            legend=(i == len(np.unique(res.region.values)) - 1),
            width=0.75,
            showfliers=False,
            whis=[10, 90],
            gap=0,
            palette='viridis_r',
        )

        xmin, xmax = axs[i].get_xlim()
        for j in np.arange(len(np.unique(res.crop))):
            if j % 2 == 0:
                axs[i].axvspan(j - 0.5, j + 0.5, facecolor='grey', alpha=0.2)
        axs[i].set_xlim(xmin, xmax)

        axs[i].set_title(r)
        axs[i].set_title(letters.pop(0), loc='left')
        axs[i].set_ylabel('$\Delta$ HPP [%]')
        axs[i].set_xlabel('')
        axs[i].set_xticklabels([])
        axs[i].axhline(y=0, color='black', linewidth=0.75)
        if i == len(np.unique(res.region.values)) - 1:
            axs[i].set_xticks(np.arange(len(np.unique(d.crop))))
            axs[i].set_xticklabels(np.unique(d.crop), rotation=90)

    # Do legend.
    handles, labels = axs[i].get_legend_handles_labels()
    axs[i].legend(handles, [rename_leg[lab] for lab in labels], title='Epoch')
    sns.move_legend(axs[i], 'upper left', bbox_to_anchor=(0.85, -1.1))

    # Save plot.
    plt.savefig(fname=file, bbox_inches='tight')


def calc_detrended_annual(
    detrended_proxy_dir='/g/data/up6/tr2908/future_hail_global/CMIP_detrended/proxy_results/',
    annual_detrended_dir='/g/data/up6/tr2908/future_hail_global/CMIP_detrended/annual_stats',
):
    """Load annual statistics for proxy data with detrended ingredients.

    Use the common global grid, and cache outputs.

    Arguments:
        detrended_proxy_dir: Directory with detrended proxy files to process.
        annual_detrended_dir: Directory for output annual detrended files.

    Returns: All annual detrended data.

    """
    # Each file in the detrended proxy directory has proxy results for a scenario
    # in which one ingredient has been detrended.
    for f in [os.path.basename(x) for x in glob(f'{detrended_proxy_dir}/*.nc')]:
        annual_file = f'{annual_detrended_dir}/{f.replace(".nc", "_native_grid.nc")}'
        if not os.path.exists(annual_file):
            print(f'Calculating annual stats for {f}')

            # Calculate annual stats.
            d = xarray.open_dataset(f'{detrended_proxy_dir}/{f}')
            d = d.reset_coords(drop=True)
            model = f.split('_')[0]
            d = d.chunk({'time': 1000, 'lat': -1, 'lon': -1})
            d = d.expand_dims({'model': [model], 'epoch': ['3C']})
            d = d.drop_vars('ship')

            # Calculate annual hail days.
            annual = annual_stats(
                d=d,
                factor=365,
                mean_vars=None,
                quantile_vars=None,
                day_vars=[x for x in list(d.variables) if 'proxy' in x],
            )
            annual_proxies = [x for x in list(annual.keys()) if ['proxy_' in x]]

            # Re-arrange the data so that the annual hail days are organised with
            # proxy as a dimension.
            annual = (
                annual[annual_proxies]
                .rename({f'{p}': f'{p[6:None]}' for p in annual_proxies})
                .to_dataarray(dim='proxy', name='annual_hail_days')
            )

            # Get year number in abstract form.
            annual['year_num'] = ('year', np.arange(1, annual.year.size + 1))
            annual = annual.swap_dims({'year': 'year_num'}).reset_coords()
            assert annual.year_num.size == 20, 'Incorrect number of years in annual.'  # noqa: PLR2004
            write_output(dat=annual, file=annual_file)

    # Regrid annual files from native to global 1 degree grid.
    regrid_global(path=f'{annual_detrended_dir}/*_native_grid.nc')


def detrended_changes(
    hist_dat,
    annual_detrended_dir='/g/data/up6/tr2908/future_hail_global/CMIP_detrended/annual_stats',
    cache_dir='/g/data/up6/tr2908/future_hail_global/CMIP_detrended/detrended_changes',
):
    """Calculate changes for models with detrended ingredients.

    Arguments:
        hist_dat: Data containing the historical dataset to compare to.
        annual_detrended_dir: The directory containing files output by
                              calc_detrended_annual().
        cache_dir: Directory to cache changes in.

    Returns: all changes organsed by model and detrended ingredient.

    """
    files = glob(f'{annual_detrended_dir}/*common_grid.nc')
    cache_files = glob(f'{cache_dir}/*.nc')

    if len(files) != len(cache_files):
        for file in files:
            print(f'Calculating changes for {file}')
            detrended = xarray.open_dataset(file)
            detrended_ing = re.search(
                'detrended_(.*)_common_grid.nc',
                os.path.basename(file),
            ).group(1)

            # Collect historic and 3D detrended results to compare.
            hist = hist_dat.sel(
                model=detrended.model,
                proxy=detrended.proxy,
                epoch=['historical'],
            )[['annual_hail_days']]
            detrended = detrended.chunk(hist.chunks)
            detrended = xarray.merge(
                [detrended.annual_hail_days, hist.annual_hail_days],
            )

            epoch_diffs = epoch_differences(
                dat=detrended,
                variables=['annual_hail_days'],
                epochs=['3C'],
                cache_dir=cache_dir,
                outstring=f'_{detrended_ing}_',
                expand_dims={'detrended_ing': [detrended_ing]},
            )
            del epoch_diffs

    # Read all cache files.
    res = []
    for file in cache_files:
        d = xarray.open_dataset(file)
        d = d.expand_dims({'model': [d.model.values]})
        res.append(d)

    return xarray.combine_by_coords(res)


def plot_drivers(
    drivers,
    ing_names=None,
    days_threshold=None,
    refs=None,
    refs_title=None,
    **kwargs,
):
    """Plot drivers with optional subsetting to remove insignificant drivers.

    Args:
        drivers: The drivers to plot.
        ing_names: Ingredient names for plotting.
        days_threshold: Require at least this many days in any of the plots to
                        show an ingredient (optional).
        kwargs: Arguments to plot_map.
        refs: If given, plots to provide as a row of reference values (ie changes
              with no drivers excluded).
        refs_title: Title for the reference row.

    """
    if ing_names is None:
        ing_names = {
            'freezing_level': 'FLH',
            'lapse_rate_700_500': 'LR',
            'melting_level': 'MLH',
            'mixed_100_cape': 'MLCAPE',
            'mixed_100_cin': 'MLCIN',
            'mixed_100_dci': 'DCI',
            'mixed_100_lifted_index': 'LI',
            'mu_cape': 'MUCAPE',
            'mu_cin': 'MUCIN',
            'mu_mixing_ratio': 'MUMR',
            'shear_magnitude': 'S06',
            'temp_500': 'T500',
        }
    if refs is None:
        refs = []

    # If references are provided, also plot column sums.
    refs_title = ['Sum', refs_title] if refs_title is not None else []
    sums = []
    if refs != []:
        sums = [drivers.sum('detrended_ing').sel(proxy=p) for p in drivers.proxy.values]

    if days_threshold is not None:
        drivers = drivers.copy(deep=True)
        null_drivers = (
            np.round(np.abs(drivers).max(['lat', 'lon', 'proxy']), 0)
            .to_dataframe()
            .reset_index()
        )
        null_drivers = null_drivers[
            null_drivers.annual_hail_days_mean_diff < days_threshold
        ]
        for _, row in null_drivers.iterrows():
            drivers = drivers.drop_sel(detrended_ing=row.detrended_ing)

    _ = plot_map(
        [
            drivers.sel(detrended_ing=ing, proxy=p)
            for ing, p in itertools.product(
                drivers.detrended_ing.values,
                drivers.proxy.values,
            )
        ]
        + sums
        + refs,
        cmap='RdBu_r',
        divergent=True,
        nrows=len(drivers.detrended_ing)
        + np.min([1, len(refs)])
        + np.min([1, len(sums)]),
        ncols=len(drivers.proxy),
        share_scale=True,
        hspace=0.1,
        disp_proj=ccrs.Robinson(),
        grid=False,
        row_labels=[ing_names[x] for x in drivers.detrended_ing.values] + refs_title,
        col_labels=[proxy_dims[f] for f in drivers.proxy.values],
        **kwargs,
    )
