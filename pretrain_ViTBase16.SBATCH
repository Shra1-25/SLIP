#!/bin/bash
#
#SBATCH --job-name=training
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=20:10:00
#SBATCH --mem=128GB
#SBATCH --mail-type=END
#SBATCH --mail-user= ## Please add your email
#SBATCH --output=train_Dermatomyositis.out

module purge
## Please change python file name in line 18

singularity exec --nv \
	    --overlay /scratch/ssc10020/overlay-50G-10M.ext3:ro \
	    /scratch/work/public/singularity/cuda11.2.2-cudnn8-devel-ubuntu20.04.sif \
	    /bin/bash -c "source /ext3/miniconda3/bin/activate; conda activate slip; python run_with_submitit.py \
		--root /scratch/ssc10020/IndependentStudy/SLIP/dataset/ISIC/ \
		--model SLIP_VITB16 \
		--lr 3e-3 \
		--wd 0.1 "