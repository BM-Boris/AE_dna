#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=11
#SBATCH --partition=day-long-cpu
#SBATCH --time=23:59:00
#SBATCH --mem=170GB
#SBATCH --job-name=ae
#SBATCH --error=sbatch_outs/serial.%J.err
#SBATCH --output=sbatch_outs/serial.%J.out
#
##Comands to run
python3 main.py
