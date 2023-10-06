#!/bin/bash
# 
# Post-process CMIP6 convective data.
#
# Author: Tim Raupach <t.raupach@unsw.edu.au>

#PBS -q normal
#PBS -P up6
#PBS -l storage=gdata/hh5+gdata/up6+gdata/oi10+gdata/dk92+gdata/w42
#PBS -l ncpus=48
#PBS -l walltime=03:00:00
#PBS -l mem=192GB
#PBS -j oe
#PBS -W umask=0022
#PBS -l wd
#PBS -l jobfs=100GB
#PBS -N job_post_process_CMIP
#PBS -r y

module use /g/data3/hh5/public/modules
module load conda/analysis3-22.10

export PYTHONPATH=$PYTHONPATH:../../modules:../../../xarray_parcel/:../../../aus400_hail/:../../../warming_levels/

# Start the dask scheduler.
dask scheduler --scheduler-file sched_"${PBS_JOBID}".json &
while ! [[ -f sched_"${PBS_JOBID}".json ]]; do sleep 10; done

# Use mpirun to run dask workers in this environment.
dask worker --nworkers 12 --nthreads 4 --memory-limit 0.166 --scheduler-file sched_"${PBS_JOBID}".json &

sleep 10

# Run the python script to do the work.
time python3 ../post_process_CMIP.py 'EPOCH_NAME' 'MODEL_NAME' 'EXP' START_YEAR END_YEAR
