#!/usr/bin/env bash
#SBATCH -A cs525
#SBATCH -p academic
#SBATCH -N 1
#SBATCH -c 8
#SBATCH --gres=gpu:1
#SBATCH -C A30
#SBATCH -t 24:00:00
#SBATCH --mem 12G
#SBATCH --job-name="P3"
#SBATCH --output=dqn_good.log     # Save standard output

source activate myenv
python main.py --train_dqn --filename dqn_good --num_episodes 80000 --decay_rate 0.0000002 --epsilon_min 0.01 --update_freq 10000 --buffer_len 100000 --learning_rate 1e-5