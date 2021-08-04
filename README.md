# HPC in Fluid Dynamics
This is the official git repository containing the submission of Emrald Abraham Thadathil (Mat. No. 5252633) for the "HPC with Python" course. The goal of this course was to use the Lattice Boltzmann Method to simulate the flow of a fluid under various conditions.

## Milestones

The "Milestones" folder contains `.ipynb`  that are all the experiments mentioned in ILIAS. All of them can be run on your local device if you have Jupyter notebook and Python installed.

## Parallel

To execute the code in Parallel using BwUnicluster, download the files in "Parallel - BwUniCluster". Login to BwUnicluster and run `sbatch LBMParallel.sh`. The current setting is to run on 25 processors across 4 nodes, i.e, 4x25 = 100 processors, time limit of 10 minutes. The status of execution can be found using the command `squeue`. The `sinfo_t_idle` command can be used to find out information about the no. of nodes with free processors.

To visualize the final output, copy the "Visualization.py" file into the same folder as the output "ux.npy" and "uy.npy" file and then run it from the terminal using `python Visualization.py`.

