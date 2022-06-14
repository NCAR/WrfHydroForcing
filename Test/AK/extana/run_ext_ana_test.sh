#!/bin/bash
#
#Test script for running forcing engine tests
#Usage: /opt/pbs/default/bin/qsub run_ext_ana_test.sh
#
#PBS -N EXT_ANA_TEST 
#PBS -A NRAL0017
#PBS -l walltime=01:30:00
#PBS -q premium
#PBS -o EXT_ANA_TEST.out
#PBS -e EXT_ANA_TEST.err
#PBS -l select=1:ncpus=36:mpiprocs=36:mem=109GB

export WGRIB2=/glade/u/home/zhangyx/software/grib2/wgrib2/wgrib2
export PATH=/glade/u/home/bpetzke/anaconda3/envs/wrfhydro/bin:$PATH

rm -f *.err *.out

module unload mpt
module load impi
mpiexec -n 36 python ../../../genForcing.py ./template_forcing_engine_AK_ExtAnA_hourly.config 2.2 ExtAnA
