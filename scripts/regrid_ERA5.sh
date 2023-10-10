#!/bin/bash
# 
# Regrid ERA5 data to 1x1 degree.
#
# Author: Tim Raupach <t.raupach@unsw.edu.au>

#PBS -q normal
#PBS -P li18
#PBS -l storage=gdata/hh5+gdata/up6+gdata/oi10+gdata/dk92+gdata/w42+gdata/rt52
#PBS -l ncpus=48
#PBS -l walltime=24:00:00
#PBS -l mem=192GB
#PBS -j oe
#PBS -W umask=0022
#PBS -l wd
#PBS -l jobfs=100GB
#PBS -r y
#PBS -N job_regrid_ERA5

module use /g/data3/hh5/public/modules
module load conda/analysis3-unstable

export PYTHONPATH=$PYTHONPATH:../modules:../../xarray_parcel/:../../aus400_hail/:../../warming_levels/

# Start the dask scheduler.
dask scheduler --scheduler-file sched_"${PBS_JOBID}".json &
while ! [[ -f sched_"${PBS_JOBID}".json ]]; do sleep 10; done

# Use mpirun to run six dask workers in this environment.
dask worker --nworkers 12 --nthreads 8 --memory-limit 0.083 --scheduler-file sched_"${PBS_JOBID}".json &

sleep 10

# Run the python script to do the work.
time python3 regrid_ERA5.py