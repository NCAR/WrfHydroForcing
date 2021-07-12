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

rm -f *.err *.out

module unload mpt
module load impi
#mpiexec python ../genForcing.py ./template_forcing_engine_ExtAnA.2.config 2.2 ExtAnA
#mpiexec python ../genForcing.py ./template_forcing_engine_ExtAnA.1.config.orig 2.2 ExtAnA
mpiexec python ../genForcing.py ./template_forcing_engine_ExtAnA.3.config.orig 2.2 ExtAnA
