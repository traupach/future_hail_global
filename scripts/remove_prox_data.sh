#!/bin/bash
# 
# Remove proxy data from existing nc files in the current directory.
#
# Author: Tim Raupach <t.raupach@unsw.edu.au>

#PBS -q normal
#PBS -P li18
#PBS -l storage=gdata/hh5+gdata/up6+gdata/oi10+gdata/dk92+gdata/w42
#PBS -l ncpus=48
#PBS -l walltime=04:00:00
#PBS -l mem=192GB
#PBS -j oe
#PBS -W umask=0022
#PBS -l wd
#PBS -l jobfs=150GB
#PBS -N job_remove_proxy
#PBS -r y

module use /g/data3/hh5/public/modules
module load nco
module load netcdf
module load parallel

mkdir noproxy
parallel --jobs 200% ncks -O -x -v proxy_SHIP_0.1,proxy_SHIP_0.5,ship,proxy_Mohr2013,proxy_Eccel2012,proxy_Kunz2007,hail_proxy_noconds,hail_proxy {} noproxy/{} ::: *.nc
mv noproxy/*.nc .