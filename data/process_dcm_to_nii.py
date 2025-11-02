import os
from os.path import join
import torch
import torchio as tio
import torch.nn.functional as F
import numpy as np
from glob import glob
from concurrent.futures import ThreadPoolExecutor

def pad(im, padding_value=0):
    cx, cy = int(im.shape[0] / 2), int(im.shape[1] / 2)
    ss = [[128 - cx, 128 + im.shape[0] - cx], [128 - cy, 128 + im.shape[1] - cy]]
    empty1 = np.ones([256, 256, im.shape[-1]]) * padding_value
    empty1[ss[0][0]:ss[0][1], ss[1][0]:ss[1][1]] = im
    return empty1

def padz(im, padding_value=0):
    cz = im.shape[-1] // 2
    ss = [128 - cz, 128 + im.shape[-1] - cz]
    empty1 = np.ones([256, 256, 256]) * padding_value
    empty1[:, :, ss[0]:ss[1]] = im
    return empty1

def process_volume(path, output_dir, crop_size=128):
    num = int(os.path.basename(path).split('.nii.gz')[0].split('-')[-1])
    img = tio.ScalarImage(path)
    img_data = img.data
    origin_shape, origin_spacing = np.asarray(img.shape[1:]), np.asarray(img.spacing)

    new_spacing = (1.25, 1.25, 1.25)
    target_shape = np.floor(origin_shape * origin_spacing / new_spacing).astype(np.int16)
    new_affine = img.affine.copy()
    new_affine[0, 0], new_affine[1, 1], new_affine[2, 2] = (
        np.sign(new_affine[0, 0]) * new_spacing[0],
        np.sign(new_affine[1, 1]) * new_spacing[1],
        np.sign(new_affine[2, 2]) * new_spacing[2],
    )

    img_data = F.interpolate(
        img_data.to(torch.float32).unsqueeze(0),
        tuple(target_shape),
        mode='trilinear'
    ).squeeze().numpy()

    norm_data_max = np.percentile(img_data, 99.9)
    norm_data = np.clip(img_data, -1000, norm_data_max)
    img_data = (norm_data - norm_data.min()) / (norm_data.max() - norm_data.min())

    if img_data.shape[0] < 256 or img_data.shape[1] < 256:
        img_data = pad(img_data)
    else:
        center = (img_data.shape[0] // 2, img_data.shape[1] // 2)
        img_data = img_data[
            center[0] - crop_size:center[0] + crop_size,
            center[1] - crop_size:center[1] + crop_size, :
        ]

    if img_data.shape[-1] < 240:
        return
    elif img_data.shape[-1] < 256:
        img_data = padz(img_data)

    idx = 0
    for z in range(0, img_data.shape[-1], 256):
        z_ = min(z, img_data.shape[-1] - 256)
        new_data = img_data[..., z_:z_ + 256]
        new_img = tio.ScalarImage(tensor=torch.from_numpy(new_data).unsqueeze(0), affine=new_affine)
        new_name = f'LIDC_{str(num).zfill(3)}_{str(idx).zfill(3)}.nii.gz'
        new_img.save(join(output_dir, new_name))
        idx += 1

def main():
    paths = glob('./dataset/*')
    output_dir = './LIDC_256_cubed'
    os.makedirs(output_dir, exist_ok=True)

    for path in paths:
        process_volume(path, output_dir)


if __name__ == "__main__":
    main()
