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
#SBATCH --output=eval_linear_w_DBert_checkpoint_seed_65967_context_length_26.out

module purge
## Please change python file name in line 18

singularity exec --nv \
	    --overlay /scratch/ssc10020/overlay-50G-10M.ext3:ro \
	    /scratch/work/public/singularity/cuda11.2.2-cudnn8-devel-ubuntu20.04.sif \
	    /bin/bash -c "source /ext3/miniconda3/bin/activate; conda activate slip; python main_linear.py \
		--pretrained /scratch/ssc10020/IndependentStudy/SLIP/pretrain_w_DBert_checkpoint_seed_65967_context_length_26/pretrain_w_DBertcheckpoint_best_seed_65967_context_len_26.pt \
		--seed 65967 \
		--arch vit_base_patch16_384 \
		--output-dir /scratch/ssc10020/IndependentStudy/SLIP/linear_w_DBert_chkpt_seed_65967_context_len_26/ \
		--resume /scratch/ssc10020/IndependentStudy/SLIP/linear_w_DBert_chkpt_seed_65967_context_len_26/linear_checkpoint.pt"