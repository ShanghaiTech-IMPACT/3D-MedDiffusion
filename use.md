source /home/guoxu4/miniconda3/etc/profile.d/conda.sh
conda activate meddiff

cd 3D-MedDiffusion-main/
python train/train_PatchVolume_wandb.py --config config/PatchVolume_4x_wandb.yaml
