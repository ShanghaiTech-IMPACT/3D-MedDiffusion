"""
Convert a folder of DICOM slices (one CT volume) to a single 3D NIfTI file.
Output is normalized to [0, 1] so 3D-MedDiffusion's dataloader (img*2-1) gets [-1,1].

Usage:
  python scripts/dicom_to_nifti.py --input /path/to/LungSBRTPatient --output /path/to/data/LungSBRT
  python scripts/dicom_to_nifti.py --input /path/to/CT_Data --output /path/to/data/CT_volumes  # multiple patient folders
"""

import argparse
import os
import glob
import numpy as np
import pydicom
import nibabel as nib


def is_image_dicom(path: str) -> bool:
    """Skip RT plan, structure set, etc.; keep CT image slices."""
    try:
        dcm = pydicom.dcmread(path, stop_before_pixels=True)
        return hasattr(dcm, "pixel_array") or (hasattr(dcm, "SOPClassUID") and "Image" in str(dcm.get("SOPClassUID", "")))
    except Exception:
        return False


def load_series(dicom_dir: str):
    """Load one DICOM series from a directory; return 3D array and affine (voxel→world)."""
    paths = glob.glob(os.path.join(dicom_dir, "*.dcm"))
    paths = [p for p in paths if is_image_dicom(p)]
    if not paths:
        raise FileNotFoundError(f"No image DICOMs found in {dicom_dir}")

    slices = []
    for p in paths:
        try:
            dcm = pydicom.dcmread(p)
            if not hasattr(dcm, "pixel_array"):
                continue
            slices.append((dcm, p))
        except Exception:
            continue

    if not slices:
        raise ValueError(f"Could not read any image slices from {dicom_dir}")

    # Sort by slice position (ImagePositionPatient or InstanceNumber)
    def sort_key(item):
        dcm = item[0]
        if hasattr(dcm, "ImagePositionPatient") and dcm.ImagePositionPatient is not None:
            return float(dcm.ImagePositionPatient[2])
        return getattr(dcm, "InstanceNumber", 0)

    slices.sort(key=sort_key)
    pixels = np.stack([s[0].pixel_array.astype(np.float32) for s in slices], axis=0)

    # Rescale to HU if possible
    dcm0 = slices[0][0]
    intercept = float(getattr(dcm0, "RescaleIntercept", 0))
    slope = float(getattr(dcm0, "RescaleSlope", 1))
    pixels = pixels * slope + intercept

    # Build affine from first slice
    spacing = [1.0, 1.0, 1.0]
    if hasattr(dcm0, "PixelSpacing") and dcm0.PixelSpacing is not None:
        spacing[0], spacing[1] = float(dcm0.PixelSpacing[0]), float(dcm0.PixelSpacing[1])
    if hasattr(dcm0, "SliceThickness"):
        spacing[2] = float(dcm0.SliceThickness)
    origin = [0.0, 0.0, 0.0]
    if hasattr(dcm0, "ImagePositionPatient") and dcm0.ImagePositionPatient is not None:
        origin = [float(x) for x in dcm0.ImagePositionPatient]
    # NIfTI: (i,j,k) -> (x,y,z). Assume axial: row=X, col=Y, slice=Z
    affine = np.eye(4)
    affine[0, 0] = spacing[0]
    affine[1, 1] = spacing[1]
    affine[2, 2] = spacing[2]
    affine[:3, 3] = origin

    return pixels, affine


def normalize_ct(vol: np.ndarray, hu_min: float = -1000.0, hu_max: float = 1000.0) -> np.ndarray:
    """Clip HU and scale to [0, 1]."""
    vol = np.clip(vol, hu_min, hu_max)
    vol = (vol - hu_min) / (hu_max - hu_min)
    return vol.astype(np.float32)


def convert_folder(
    input_path: str,
    output_dir: str,
    hu_min: float = -1000.0,
    hu_max: float = 1000.0,
    one_volume_per_subfolder: bool = True,
):
    """
    Convert DICOM(s) to NIfTI.
    - If input_path is a folder of .dcm files → one volume, one .nii.gz.
    - If one_volume_per_subfolder and input_path has subdirs with .dcm → one .nii.gz per subdir.
    """
    os.makedirs(output_dir, exist_ok=True)

    if one_volume_per_subfolder:
        subdirs = [d for d in glob.glob(os.path.join(input_path, "*")) if os.path.isdir(d)]
        has_dcm = any(glob.glob(os.path.join(input_path, "*.dcm")))
        if has_dcm and not subdirs:
            # Single series in input_path
            subdirs = [input_path]
        elif not subdirs:
            subdirs = [input_path]
    else:
        subdirs = [input_path]

    for d in subdirs:
        try:
            vol, affine = load_series(d)
        except Exception as e:
            print(f"Skipping {d}: {e}")
            continue
        vol = normalize_ct(vol, hu_min=hu_min, hu_max=hu_max)
        # NIfTI: (x,y,z) convention; we have (slice, row, col) -> (k,i,j) -> store as (i,j,k) for (x,y,z)
        vol = np.transpose(vol, (1, 2, 0))  # (row, col, slice) -> (x,y,z) in RAS-like order
        nii = nib.Nifti1Image(vol, affine)
        name = os.path.basename(d.rstrip(os.sep)) + ".nii.gz"
        out_path = os.path.join(output_dir, name)
        nib.save(nii, out_path)
        print(f"Saved {out_path} shape {vol.shape}")


def main():
    p = argparse.ArgumentParser(description="DICOM series folder(s) -> 3D NIfTI [0,1] for 3D-MedDiffusion")
    p.add_argument("--input", "-i", required=True, help="Path to folder of DICOMs or parent of patient folders")
    p.add_argument("--output", "-o", required=True, help="Output directory for .nii.gz files")
    p.add_argument("--hu-min", type=float, default=-1000.0, help="HU window min (default -1000)")
    p.add_argument("--hu-max", type=float, default=1000.0, help="HU window max (default 1000)")
    p.add_argument("--single-folder", action="store_true", help="Treat input as one series only (no subfolders)")
    args = p.parse_args()
    convert_folder(
        args.input,
        args.output,
        hu_min=args.hu_min,
        hu_max=args.hu_max,
        one_volume_per_subfolder=not args.single_folder,
    )


if __name__ == "__main__":
    main()
