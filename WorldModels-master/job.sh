#!/bin/sh
#SBATCH --partition=SCSEGPU_UG
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --mem=8000M
#SBATCH --job-name=ADSPTest
#SBATCH --output=test.out
#SBATCH --error=testError.err

/home/FYP/kwinata002/.conda/envs/adsp/bin/python3.7 04_train_rnn.py --new_model
