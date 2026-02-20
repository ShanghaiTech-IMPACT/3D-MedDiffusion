# Using Lung SBRT CT Data (DICOM → 3D-MedDiffusion)

Your data under `Accelerated_SIEMAC/CT_Data/LungSBRTPatient` is **DICOM slices** (one CT volume). The repo expects **3D NIfTI** (`.nii.gz`) volumes normalized to **[0, 1]**.

## Step 1: Convert DICOM → NIfTI

Run from the repo root:

```bash
conda activate 3DMedDiffusion

# One patient folder → one .nii.gz
python scripts/dicom_to_nifti.py \
  --input "/media/liyong/f001be89-9498-4619-9827-7607b8ac9501/home/liyong/Arnav/IPO-IMPT_Optimization/Accelerated_SIEMAC/CT_Data/LungSBRTPatient" \
  --output "data/LungSBRT" \
  --single-folder
```

Optional: adjust CT window (default HU -1000 to 1000):

```bash
python scripts/dicom_to_nifti.py --input /path/to/LungSBRTPatient --output data/LungSBRT --single-folder --hu-min -150 --hu-max 250
```

This writes e.g. `data/LungSBRT/LungSBRTPatient.nii.gz` (values in [0,1]).

## Step 2: Point configs at the NIfTI directory

Configs are already set to use `data/LungSBRT`:

- **PatchVolume (AE) training:** `config/PatchVolume_data_LungSBRT.json`  
  - Set `dataset.root_dir` in `config/PatchVolume_8x.yaml` (or `PatchVolume_4x.yaml`) to the **absolute** path to this JSON, or edit the JSON so the value is the **absolute** path to `data/LungSBRT`.

- **BiFlowNet / latent generation:** `config/Singleres_dataset_LungSBRT.json`  
  - Class `"0"` = Lung SBRT; value = **absolute** path to `data/LungSBRT`.

If you use a different output directory, edit the JSONs so the paths point to the folder that **contains** the `.nii.gz` files.

## Step 3: Pipeline

1. **Train PatchVolume AE** (optional if you already have a checkpoint):
   - In `PatchVolume_8x.yaml`: set `dataset.root_dir` to `config/PatchVolume_data_LungSBRT.json` (or full path to that JSON), and `default_root_dir` to your log/ckpt dir.
   - Run Stage 1, then Stage 2 as in the main README.

2. **Encode volumes to latents:**
   ```bash
   python train/generate_training_latent.py \
     --data-path config/Singleres_dataset_LungSBRT.json \
     --AE-ckpt checkpoints/your_AE.ckpt \
     --batch-size 4
   ```
   This creates `data/LungSBRT_latents/` with latent `.nii.gz` files.

3. **Train BiFlowNet** (optional):
   - Use `--data-path config/Singleres_dataset_LungSBRT.json` (the dataset loads from `*_latents` when `generate_latents=False`).
   - Set `--num-classes 1` for a single class (Lung SBRT).

4. **Inference:** Use the main inference scripts with your AE and BiFlowNet checkpoints; for a single class, class index is `0`.

## One patient only

With only one volume (one patient), AE and BiFlowNet training will overfit unless you add more data. The conversion and configs still let you:

- Run latent generation and inspect reconstructions with a **pretrained** AE.
- Try fine-tuning or use the pipeline as a template when you have more scans.
