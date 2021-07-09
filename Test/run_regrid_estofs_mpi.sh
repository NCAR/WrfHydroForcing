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

rm -f *.err *.out

module load impi
#mpiexec -np 35 python -m mpi4py.futures ../Util/regrid_estofs_futures.py  /glade/scratch/rcabell/coastal/estofs.t00z.fields.cwl.nc /glade/scratch/bpetzke/ForcingEngine/gr3_2_esmf/schism.hgrid.nc /glade/scratch/bpetzke/ForcingEngine/regrid_estofs/estofs.t00z.fields.cwl.regrid.nc 
mpiexec -n 1 -usize 35 python -m mpi4py.futures ../Util/regrid_estofs_futures.py  /glade/scratch/rcabell/coastal/estofs.t00z.fields.cwl.nc /glade/scratch/bpetzke/ForcingEngine/gr3_2_esmf/schism.hgrid.nc /glade/scratch/bpetzke/ForcingEngine/regrid_estofs/estofs.t00z.fields.cwl.regrid.nc 
#mpiexec python ../Util/MPIpool.py
