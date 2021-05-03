#!/bin/bash

#Script to allow running the FE in a single MPI thread for debugging

mpiexec -n 1 xterm -e "python -m pudb ../genForcing.py ./template_forcing_engine_AnA_v2.config 2.2 AnA"
