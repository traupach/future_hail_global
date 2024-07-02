#!/bin/bash
# 
# Detrend proxy ingredients for a 3C future run. 
#
# Author: Tim Raupach <t.raupach@unsw.edu.au>

#PBS -q normal
#PBS -P w42
#PBS -l storage=gdata/hh5+gdata/up6+gdata/oi10+gdata/dk92+gdata/w42
#PBS -l ncpus=48
#PBS -l walltime=03:00:00
#PBS -l mem=192GB
#PBS -j oe
#PBS -W umask=0022
#PBS -l wd
#PBS -l jobfs=50GB
#PBS -N job_detrend_ingredients
#PBS -r y
#PBS -J 0-7

module use /g/data3/hh5/public/modules
module load conda/analysis3-24.04

# Start the dask scheduler.
dask scheduler --scheduler-file sched_"${PBS_JOBID}".json &
while ! [[ -f sched_"${PBS_JOBID}".json ]]; do sleep 10; done

# Use mpirun to run dask workers in this environment.
dask worker --nworkers 8 --nthreads 4 --memory-limit 0.125 --scheduler-file sched_"${PBS_JOBID}".json &

sleep 10

# Run the python script to do the work.
time python3 detrend_ingredients.py ${PBS_ARRAY_INDEX}
