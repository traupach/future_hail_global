# Functions to apply the proxy of Raupach et al., 2023 (doi: 10.1175/MWR-D-22-0127.1) with 
# optional limiting of the proxy variation with melting level height.

import json
import math
import xarray
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ruff: noqa: E712

def apply_Raupach_proxy(dat, results_file, extra_conds_file, load=False, subset_conds=None, 
                        band_limits=[2000,None], **kwargs):
    """
    Apply the full trained proxy with optional extra conditions.
    
    Arguments:
    
        - dat: Data work apply the proxy to.
        - results_file: File with proxy settings.
        - extra_conds_file: File with extra conditions (or None to not apply conditions).
        - load: Load the data into memory?
        - band_limits: Limits for the band values.
        - **kwargs: Extra argumenst to apply_proxy().
        
    Returns:
        - True/False for each data point indicating whether the proxy was satisfied.
    """
    
    # Read proxy settings.
    with open(results_file, 'r') as f:
        era5_results = json.load(f)
        
    if extra_conds_file is None:
        extra_conds = pd.DataFrame()
    else:
        extra_conds = pd.read_csv(extra_conds_file)
        if subset_conds is not None:
            extra_conds = extra_conds.iloc[subset_conds,:]

    # Apply the proxy.
    proxy_res = apply_proxy(dat, 
                            band_var=era5_results['band_variable'], 
                            alpha_a=era5_results['hail_alpha_a'],
                            alpha_b=era5_results['hail_alpha_b'], 
                            beta_a=era5_results['hail_beta_a'], 
                            beta_b=era5_results['hail_beta_b'],
                            extra_conditions=extra_conds, 
                            band_limits=band_limits,
                            **kwargs)
    
    proxy_res.name = 'hail_proxy'
    proxy_res.attrs['long_name'] = 'Hail-prone atmospheric conditions'
    proxy_res.attrs['description'] = 'Hail proxy results'
    proxy_res.attrs['proxy_band_var']= era5_results['band_variable'] 
    proxy_res.attrs['proxy_alpha_a'] = era5_results['hail_alpha_a']
    proxy_res.attrs['proxy_alpha_b'] = era5_results['hail_alpha_b']
    proxy_res.attrs['proxy_beta_a'] = era5_results['hail_beta_a'] 
    proxy_res.attrs['proxy_beta_b'] = era5_results['hail_beta_b']
    for i, row in extra_conds.iterrows():
        proxy_res.attrs['proxy_extra_cond_' + str(i)] = f'{row.variable} {row.sign} {row.threshold}'

    bl = 'None'
    if band_limits is not None:
        bl = f'[{band_limits[0]}, {band_limits[1]}]'
    proxy_res.attrs['proxy_band_limits'] = bl
    
    if load:
        proxy_res = proxy_res.load()
        
    return proxy_res

def proxy_by_band(band, alpha_a, alpha_b, beta_a, beta_b, band_limits=None):
    """

    Find alpha and beta using their relationship to another 'band' variable.

    Arguments:
        - band: Data to use to determine values of alpha and beta.
        - alpha_a, alpha_b: Values of a and b in alpha = a*band_var + b.
        - beta_a, beta_b: Values of a and b in log10(beta) = a*band_var + b.
        - band_limits: If band values are outside these limits, return the values of 
                       alpha/beta respectively at the band limits.

    Returns: 
    
        - alpha, beta for each value in band.
    """
    
    # Determine alpha and beta from the band variable.
    alpha = band * alpha_a + alpha_b
    beta = 10**(band * beta_a) * 10**(beta_b)
    
    # Apply band limits if required.
    if band_limits is not None:
        band_min = band_limits[0]
        band_max = band_limits[1]

        if band_min is not None:
            min_alpha = band_limits[0] * alpha_a + alpha_b
            min_beta = 10**(band_limits[0] * beta_a) * 10**(beta_b)
            alpha = alpha.where(band >= band_limits[0], other=min_alpha) 
            beta = beta.where(band >= band_limits[0], other=min_beta)        
        if band_max is not None:
            max_alpha = band_limits[1] * alpha_a + alpha_b
            max_beta = 10**(band_limits[1] * beta_a) * 10**(beta_b)
            alpha = alpha.where(band <= band_limits[1], other=max_alpha)
            beta = beta.where(band <= band_limits[1], other=max_beta)

    return alpha, beta

def apply_proxy(dat, band_var, alpha_a, alpha_b, beta_a, beta_b, 
                x='mixed_100_cape', y='shear_magnitude',
                extra_conditions=None, band_limits=None): 
    """
    Apply a proxy in which proxy parameters depend on a third variable. 
    
    The proxy has the form:
    
        x * y^alpha >= beta
        
    The values of alpha and log10(beta) are determined using linear relationships another variable.
    
    Arguments:
    
        - dat: The data to apply the proxy to (DataFrame).
        - band_var: Variable to use to determine values of alpha and beta.
        - alpha_a, alpha_b: Values of a and b in alpha = a*band_var + b.
        - beta_a, beta_b: Values of a and b in log10(beta) = a*band_var + b.
        - x: Variable to use for 'x'.
        - y: Variable to use for 'y'.
        - extra_conditions: A DataFrame of extra conditions to apply. Must contain variable, 
                            sign ('>' or '<'), and threshold. All conditions must hold for the
                            proxy to be True.
        - band_limits: If band values are outside these limits, return the values of 
                       alpha/beta respectively at the band limits.
        
    Returns: 
    
        - True/False for each data point indicating whether the proxy was satisfied.
    """

    alpha, beta = proxy_by_band(band=dat[band_var], alpha_a=alpha_a, alpha_b=alpha_b, 
                                beta_a=beta_a, beta_b=beta_b, band_limits=band_limits)
    
    # Do the proxy calculation.
    res = dat[x] * dat[y]**alpha >= beta

    # Apply extra conditions if required.
    if extra_conditions is not None:        
        for row in extra_conditions.itertuples():
            assert row.sign == '<' or row.sign == '>', 'Invalid sign.'
            if row.sign == '>':
                res = res.where(dat[row.variable] > row.threshold, other=False)
            else:
                res= res.where(dat[row.variable] < row.threshold, other=False)

    return res

def plot_power_law(x, p, t, ax, label):
    """
    Plot the line x * y^p = t onto an axis with a given label.
    """

    y = (t/x)**(1/p)
    ax.plot(x, y, label=label)

def prox_performance(dat, proxy, extra_conditions):
    """
    Test proxy performance and show effects of extra conditions.

    Arguments:
        - dat: The data to be analyzed. Should include a 'true_hail' column and a column named as per the `proxy` parameter.
        - proxy: A string representing the column in `dat` that acts as the proxy for hail detection.
        - extra_conditions: A DataFrame where each row represents an extra condition to be applied to `dat`.
    """
    
    def stats(d):
        hit = d[np.logical_and(d.true_hail == True, d[proxy] == True)]
        miss = d[np.logical_and(d.true_hail == True, d[proxy] == False)]
        true_neg = d[np.logical_and(d.true_hail == False, d[proxy] == False)]
        false_pos = d[np.logical_and(d.true_hail == False, d[proxy] == True)]
        return hit, miss, true_neg, false_pos

    def skill(h, m, n, f):
        POD = h/(h+m)
        FAR = f/(h+f)
        HSS = 2 * (h*n - m*f) / (m**2 + f**2 + 2*h*n + (m+f)*(h+n))
        SR = 1-FAR
        bias = POD/SR
        CSI = 1/(1/SR + 1/POD - 1)
        
        print(f'POD: {np.round(POD, 2)}')
        print(f'FAR: {np.round(FAR, 2)}')
        print(f'HSS: {np.round(HSS, 2)}')
        print(f'SR: {np.round(SR, 2)}')
        print(f'bias: {np.round(bias, 2)}')
        print(f'CSI: {np.round(CSI, 2)}')

    hit, miss, true_neg, false_pos = stats(dat)
    print('Before extra conds:')
    skill(h=len(hit), m=len(miss), n=len(true_neg), f=len(false_pos))
    print('')

    for i, row in extra_conditions.iterrows():
        assert row.sign == '<', 'Unexpected sign.'

        print((f'{row.variable} < {row.threshold} removes {np.round((hit[row.variable] >= row.threshold).mean()*100, 1)}% of hits ' + 
           f'and {np.round((false_pos[row.variable] >= row.threshold).mean()*100, 0)}% of false positives.'))

        dat.loc[dat[row.variable] >= row.threshold, proxy] = False
        hit, miss, true_neg, false_pos = stats(dat)        
    
    print('\nAfter extra conds:')
    skill(h=len(hit), m=len(miss), n=len(true_neg), f=len(false_pos))

def plot_proxy_discrims(era5_results, MLH_vals=[200, 1000, 1500, 2000, 3000, 8000], file=None, figsize=(12,3)):
    """
    Plot the discriminator for the proxy by melting level height value.

    Arguments:
        era5_results: The ERA5-based proxy results.
        MLH_vals: The values of melting level height to plot for.
        file: File to save to.
    """
    
    fig, axs = plt.subplots(ncols=2, figsize=figsize)
    CAPE = np.arange(1,10000)
    for MLH in MLH_vals:
        alpha, beta = proxy_by_band(band=np.array([MLH]), 
                                    alpha_a = era5_results['hail_alpha_a'],
                                    alpha_b = era5_results['hail_alpha_b'], 
                                    beta_a = era5_results['hail_beta_a'], 
                                    beta_b = era5_results['hail_beta_b'])

        print(f'MLH: {MLH}, alpha: {alpha}, beta {beta}.')
        plot_power_law(x=CAPE, p=alpha, t=beta, ax=axs[0], label=f'{MLH} m')
        plot_power_law(x=CAPE, p=alpha, t=beta, ax=axs[1], label=f'{MLH} m')
    
    axs[0].set_xscale('log')
    axs[0].set_yscale('log')
    axs[0].set_ylim(1,75)
    axs[0].set_xlim(1,10000)
    axs[1].set_ylim(-5,75)
    axs[1].set_xlim(-300,10000)
    axs[0].set_title('Log scales')
    axs[1].set_title('Linear scales')
    
    for ax in axs:
        ax.set_xlabel('CAPE [J kg$^{-1}$]')
        ax.set_ylabel('S$_{06}$ [m s$^{-1}$]')
    
    plt.legend(loc='upper right', ncol=2, title='Melting level height')

    if file is not None:
        plt.savefig(fname=file, bbox_inches='tight')

    plt.tight_layout()
    plt.show()

def era5_proxies(climatol_dir, landsea_file, results_file, extra_conds_file, 
                 chunks={'time': 500, 'latitude': -1, 'longitude': -1},
                 UTC_hours= ['03', '06', '09']):
    """
    Read ERA5 processed data and return a daily timeseries. Calculate new 
    and old hail proxies. For each day the maximum hail proxy value is 
    returned for all UTC times processed.
    
    Arguments:
        climatol_dir: Processed ERA5 climatology directory.
        landsea_file: Land sea mask file.
        results_file: File with proxy settings.
        extra_conds_file: File with extra conditions (or None to not apply conditions).
        chunks: Chunks for the output.
        UTC_hours: UTC hours for which processed data exist.
        
    Returns: climatol_all, climatol_land, perc_nonphys where
        climatol_all contains all points
        climatol_land contains only land points
        perc_nonphys is the percentage of points that were deemed non-physical 
        (had negative q in the ERA5 data).
    """
    
    climatol = {}
    num_nonphys = 0
    total_pts = 0

    land_sea_mask = xarray.open_dataset(landsea_file)
    if np.isin('time', land_sea_mask.keys()):
        # Modified LSM does not have time dimension; if using ERA5 LSM select the first time.
        land_sea_mask = land_sea_mask.isel(time=0)
    
    for UTC in UTC_hours:
        dat = xarray.open_mfdataset(f'{climatol_dir}/*UTC{UTC}.nc', parallel=True)
        num_nonphys = num_nonphys + np.logical_not(dat.physical_mask).sum()
        total_pts = total_pts + math.prod(list(dat.dims.values()))
        dat = dat.sortby('time').transpose('time', 'latitude', 'longitude')
        dat = dat.chunk(chunks)

        dat['old_prox'] = apply_Raupach_proxy(dat=dat, results_file=results_file, 
                                              extra_conds_file=extra_conds_file, 
                                              load=True, band_limits=None)

        dat['new_prox'] = apply_Raupach_proxy(dat=dat, results_file=results_file, 
                                              extra_conds_file=extra_conds_file, 
                                              load=True)
        
        # Round down to the start of each day.
        dat['time'] = dat.time.dt.floor('1D')

        # Ignore non-physical points in the ERA5 data.
        climatol[UTC] = dat.where(dat.physical_mask == True)
        del dat

    # Average all fields except the proxy.
    with xarray.set_options(keep_attrs=True):
        for i, k in enumerate(list(climatol.keys())):
            if i == 0:
                climatol_all = climatol[k]
            else:
                climatol_all = climatol_all + climatol[k]

        climatol_all = climatol_all / len(list(climatol.keys()))
        climatol_all['old_prox'] = climatol_all.old_prox.where(climatol_all.old_prox == 0, other=1)
        climatol_all['new_prox'] = climatol_all.new_prox.where(climatol_all.new_prox == 0, other=1)
        
        # If any of the combined points were non-physical, mark the output as non-physical and mask it out.
        climatol_all['physical_mask'] = climatol_all.physical_mask.where(climatol_all.physical_mask == 1, other=0)
        climatol_all = climatol_all.where(climatol_all.physical_mask)
        del climatol 
    
    perc_nonphys = num_nonphys / total_pts * 100
    climatol_land = climatol_all.where(land_sea_mask.lsm > 0.5)
    return climatol_all, climatol_land, perc_nonphys