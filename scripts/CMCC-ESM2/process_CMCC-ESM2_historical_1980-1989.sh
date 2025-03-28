#!/bin/bash
# 
# Process CMIP6 data to get convective indices.
#
# Author: Tim Raupach <t.raupach@unsw.edu.au>

#PBS -q normal
#PBS -P up6
#PBS -l storage=gdata/hh5+gdata/up6+gdata/oi10+gdata/dk92+gdata/w42
#PBS -l ncpus=48
#PBS -l walltime=07:00:00
#PBS -l mem=192GB
#PBS -j oe
#PBS -W umask=0022
#PBS -l wd
#PBS -l jobfs=100GB
#PBS -N job_CMCC-ESM2_1980-1989
#PBS -r y
#PBS -J 1980-1989

module use /g/data3/hh5/public/modules
module load conda/analysis3-22.10

export PYTHONPATH=$PYTHONPATH:../../modules:../../../xarray_parcel/:../../../aus400_hail/:../../../warming_levels/

# Start the dask scheduler.
dask scheduler --scheduler-file sched_"${PBS_JOBID}".json &
while ! [[ -f sched_"${PBS_JOBID}".json ]]; do sleep 10; done

# Use mpirun to run six dask workers in this environment.
dask worker --nworkers 6 --nthreads 8 --memory-limit 0.166 --scheduler-file sched_"${PBS_JOBID}".json &

sleep 10

# Run the python script to do the work.
time python3 ../process_CMIP.py 'CMIP6.CMIP.CMCC.CMCC-ESM2.historical.r1i1p1f1' ${PBS_ARRAY_INDEX}
