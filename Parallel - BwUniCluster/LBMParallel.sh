#!/bin/bash
#SBATCH --nodes=4
#SBATCH --time=00:10:00
#SBATCH --partition=multiple
#SBATCH --ntasks-per-node=25
#SBATCH --mail-user=et110@uni-freiburg.de
#SBATCH --mail-type=END,FAIL
#SBATCH --output=LBMParallel.out
#SBATCH --error=LBMParallel.err
echo "Loading Pythona module and mpi module"
module load devel/python/3.9.2_gnu_10.2
module load mpi/openmpi/4.1
module list
startexe="mpirun --bind-to core --map-by core -report-bindings python3 ./LBMParallel.py"
exec $startexe
