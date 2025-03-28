#!/bin/bash
# 
# Post-process CMIP6 convective data.
#
# Author: Tim Raupach <t.raupach@unsw.edu.au>

#PBS -q hugemem
#PBS -P li18
#PBS -l storage=gdata/hh5+gdata/up6+gdata/oi10+gdata/dk92+gdata/w42
#PBS -l ncpus=48
#PBS -l walltime=04:00:00
#PBS -l mem=384GB
#PBS -j oe
#PBS -W umask=0022
#PBS -l wd
#PBS -l jobfs=150GB
#PBS -N job_post_process_CMIP
#PBS -r y

module use /g/data3/hh5/public/modules
module load conda/analysis3-22.10

export PYTHONPATH=$PYTHONPATH:../../modules:../../../xarray_parcel/:../../../aus400_hail/:../../../warming_levels/

# Start the dask scheduler.
dask scheduler --scheduler-file sched_"${PBS_JOBID}".json &
while ! [[ -f sched_"${PBS_JOBID}".json ]]; do sleep 10; done

# Use mpirun to run dask workers in this environment.
dask worker --nworkers 8 --nthreads 4 --memory-limit 0.125 --scheduler-file sched_"${PBS_JOBID}".json &

sleep 10

# Run the python script to do the work.
time python3 ../post_process_CMIP.py '3C' 'MRI-ESM2-0' 'ssp585' 2062 2081
