#!/bin/bash
#SBATCH --job-name=gcn_node_classify
#SBATCH --mail-type=ALL

#SBATCH --gres=gpu:1
#SBATCH --mem=64g

#SBATCH --mail-type=END,FAIL,BEGIN
#SBATCH --mail-user=anilkumar.swamy-intern@naverlabs.com

#SBATCH --output=/tmp-network/user/aswamy/hype/gcn_node_classify_%j.log
#SBATCH --error=/tmp-network/user/aswamy/hype/gcn_node_classify_%j.err

#SBATCH --workdir=/nfs/team/mlo/aswamy/code/gcn_node_classification

source /etc/proxyrc

source /home/aswamy/tools/anaconda3/bin/activate pygeom

# Start the code
stdbuf -o0 -i0 -e0 python train.py "$@"

cp /tmp-network/user/aswamy/hype/gcn_node_classify_${SLURM_JOB_ID}.log /nfs/team/mlo/aswamy/logs/hype/gcn_node_classify_${SLURM_JOB_ID}.log     