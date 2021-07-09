#!/bin/bash
#
#Test script for running forcing engine tests
#Usage: /opt/pbs/default/bin/qsub run_mm_test.sh
#
#PBS -N MM_TEST 
#PBS -A NRAL0017
#PBS -l walltime=01:30:00
#PBS -q premium
#PBS -o MM_TEST.out
#PBS -e MM_TEST.err
#PBS -l select=1:ncpus=36:mpiprocs=36:mem=109GB

export WGRIB2=/glade/u/home/zhangyx/software/grib1/wgrib

rm -f *.err *.out

module unload mpt
module load impi
mpiexec python ../genForcing.py ./template_forcing_engine_Medium.config 2.2 AnA
