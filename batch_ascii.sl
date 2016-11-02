#!/bin/bash -l

#SBATCH -p regular

#SBATCH -N 2

#SBATCH -t 00:30:00

#SBATCH -L SCRATCH   #note: specify license need for the file systems your job needs, such as SCRATCH,project

#SBATCH -C haswell  

srun -n 32 -c 4 python pickle2ascii.py
