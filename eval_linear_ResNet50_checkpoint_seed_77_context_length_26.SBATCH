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
#SBATCH --output=./multimodal_logs/eval_linear_CBIS_multimodal_ResNet50_checkpoint_seed_77_context_length_26_lr_1e-4_wd_1e-6.out

module purge
## Please change python file name in line 18

singularity exec --nv \
	    --overlay /scratch/ssc10020/overlay-50G-10M.ext3:ro \
	    /scratch/work/public/singularity/cuda11.2.2-cudnn8-devel-ubuntu20.04.sif \
	    /bin/bash -c "source /ext3/miniconda3/bin/activate; conda activate slip; python main_linear_multimodal.py \
		--root /scratch/ssc10020/IndependentStudy/SLIP/dataset/CBIS_DDSM/ \
		--dataset cbis \
		--model SLIP_ResNet50 \
		--pretrained /scratch/ssc10020/IndependentStudy/SLIP/pretrain_CBIS_ResNet50_checkpoint_seed_77_context_length_26/pretrain_CBIS_ResNet50_lr_1e-4_wd_1e-6checkpoint_best_seed_77_context_len_26.pt \
		--num-classes 3 \
		--seed 77 \
		--arch resnet50 \
		--save-model-name-tag linear_CBIS_multimodal_ResNet50_lr_1e-4_wd_1e-6_ \
		--output-dir /scratch/ssc10020/IndependentStudy/SLIP/linear_CBIS_ResNet50_chkpt_seed_77_context_len_26/ \
		--resume /scratch/ssc10020/IndependentStudy/SLIP/linear_CBIS_ResNet50_chkpt_seed_77_context_len_26/linear_CBIS_ResNet50_lr_1e-4_wd_1e-6_checkpoint_best_seed_77_context_len_26.pt"