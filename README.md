# Software for "Shifting hail hazard under global warming"

This repository contains the complete code used for the research study "Shifting hail hazard under global warming" by T.H. Raupach, R. Portmann, C. Siderius, and S.C. Sherwood. 

## System requirements

Required python packages:
* xarray_parcel by T. H. Raupach (https://doi.org/10.5281/zenodo.15081094) version >= v1.0.7.
* warming_levels by T. H. Raupach (https://doi.org/10.5281/zenodo.10785698) version >= v1.0.1. 
* python-cmethods by B. T. Schwertfeger (https://doi.org/10.5281/zenodo.12168002) version >= 2.3.0.

Python code is operating-system independent; this version was tested using Linux (Rocky Linux 8.10 (Green Obsidian) on x86_64) with standard hardware, xarray_parcel version v1.0.7, warming_levels version v1.0.1, and python-cmethods version 2.3.0.

## Installation guide

Installation takes only a few minutes.

* Install required packages.
* Clone the git repository.
* Run the future_hail.ipynb notebook in a JupyterLab session. Follow instructions in the document for running extra scripts.

## Demo

The demo in `demo/future_hail_demo.ipynb` shows calculation of convective indices and proxy results for a single time step for a single CMIP6 model.

## License

Code is by T. H. Raupach. The paper and supplementary material are by T.H. Raupach, R. Portmann, C. Siderius, and S.C. Sherwood.

<p xmlns:cc="http://creativecommons.org/ns#" >This work is licensed under <a href="https://creativecommons.org/licenses/by-nc/4.0/?ref=chooser-v1" target="_blank" rel="license noopener noreferrer" style="display:inline-block;">CC BY-NC 4.0 <img style="height:22px!important;margin-left:3px;vertical-align:text-bottom;" src="https://mirrors.creativecommons.org/presskit/icons/cc.svg?ref=chooser-v1" alt=""><img style="height:22px!important;margin-left:3px;vertical-align:text-bottom;" src="https://mirrors.creativecommons.org/presskit/icons/by.svg?ref=chooser-v1" alt=""><img style="height:22px!important;margin-left:3px;vertical-align:text-bottom;" src="https://mirrors.creativecommons.org/presskit/icons/nc.svg?ref=chooser-v1" alt=""></a>.</p>
