#!/bin/bash
#
#Test script for running forcing engine tests
#Usage: /opt/pbs/default/bin/qsub create_forcing_test.sh 
#
#PBS -N AnA_REALTIME
#PBS -A NRAL0017
#PBS -l walltime=01:30:00
#PBS -q premium
#PBS -o AnA_REALTIME.out
#PBS -e AnA_REALTIME.err
#PBS -l select=1:ncpus=36:mpiprocs=36:mem=109GB

export WGRIB2=/glade/u/home/zhangyx/software/grib2/wgrib2/wgrib2
 
rm -f *.err *.out

module unload mpt
module load impi
mpiexec python ../genForcing.py ./template_forcing_engine_AnA_v2.config 2.2 AnA
