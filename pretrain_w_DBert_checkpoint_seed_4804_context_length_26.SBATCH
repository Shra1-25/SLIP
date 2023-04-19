#!/bin/bash
#
#SBATCH --job-name=training
#SBATCH --gres=gpu:rtx8000:1
#SBATCH --nodes 1
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=5
#SBATCH --time=20:10:00
#SBATCH --mem=64GB
#SBATCH --mail-type=END
#SBATCH --output=pretrain_CBIS_w_DBert_checkpoint_seed_4804_context_length_26_lr_1e-4_wd_1e-6.out

module purge
## Please change python file name in line 18

singularity exec --nv \
	    --overlay /scratch/ssc10020/overlay-50G-10M.ext3:ro \
	    /scratch/work/public/singularity/cuda11.2.2-cudnn8-devel-ubuntu20.04.sif \
	    /bin/bash -c "source /ext3/miniconda3/bin/activate; conda activate slip; python main.py \
		--root /scratch/ssc10020/IndependentStudy/SLIP/dataset/CBIS_DDSM/ \
		--dataset cbis \
		--model SLIP_VITB16 \
		--seed 4804 \
		--lr 1e-4 \
		--wd 1e-6 \
		--resume pretrain_CBIS_w_DBert_checkpoint_seed_4804_context_length_26_lr_1e-4_wd_1e-6.pt \
		--save-model-name-tag pretrain_CBIS_w_DBert_lr_1e-4_wd_1e-6 \
		--context-length 26 \
		--output-dir /scratch/ssc10020/IndependentStudy/SLIP/pretrain_CBIS_w_DBert_checkpoint_seed_4804_context_length_26"