import os
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
from functools import partial
import dicom2nifti
from tqdm import tqdm
import logging
from datetime import datetime

# --- 1. 请在这里配置您的路径 ---

# 包含所有LIDC-IDRI-XXXX文件夹的根目录
# 例如: 'D:/datasets/LIDC-IDRI'
INPUT_DICOM_ROOT = Path('3D-MedDiffusion-main/data/LIDC-IDRI/LIDC-IDRI-DICOM-DATA') 

# 保存转换后.nii.gz文件的输出目录
OUTPUT_NIFTI_ROOT = Path('3D-MedDiffusion-main/data/LIDC-IDRI/result')

# --- 2. 配置要处理的患者ID范围 ---

# 根据您的要求，从131到1012
PATIENT_ID_RANGE = range(131, 1013)

# ------------------------------------

def convert_patient_series(patient_dir: Path, output_root: Path):
    """
    自动扫描单个患者目录中的所有DICOM序列，并将其转换为NIfTI格式。

    Args:
        patient_dir (Path): 单个患者的根目录 (例如, '.../LIDC-IDRI-0131')。
        output_root (Path): 保存NIfTI文件的总输出目录。
    """
    patient_id = patient_dir.name
    try:
        # dicom2nifti会自动扫描子目录，找到并转换所有合法的DICOM序列。
        # reorient=True 会将图像方向调整到标准方向，这对于后续处理通常是必要的。
        dicom2nifti.convert_directory(
            dicom_directory=str(patient_dir),
            output_folder=str(output_root),
            compression=True,  # 输出为 .nii.gz 格式
            reorient=True
        )
        return {"status": "success", "patient_id": patient_id, "error": None}
    except dicom2nifti.exceptions.ConversionError as e:
        # 捕获DICOM转换特定错误（如缺失文件、切片不匹配等）
        error_type = "CONVERSION_ERROR"
        return {"status": "failed", "patient_id": patient_id, "error": f"{error_type}: {str(e)}"}
    except Exception as e:
        # 捕获其他类型的错误
        error_type = "UNKNOWN_ERROR"
        return {"status": "failed", "patient_id": patient_id, "error": f"{error_type}: {str(e)}"}

def main():
    """
    主执行函数：构建患者列表，并使用多进程并行转换。
    """
    # 确保输出目录存在
    OUTPUT_NIFTI_ROOT.mkdir(exist_ok=True)

    # 根据指定的范围，生成所有需要处理的患者目录路径列表
    patient_dirs_to_process = []
    for i in PATIENT_ID_RANGE:
        patient_id = f"LIDC-IDRI-{i:04d}"
        current_path = INPUT_DICOM_ROOT / patient_id
        if current_path.is_dir():
            patient_dirs_to_process.append(current_path)
        else:
            # 如果某个ID的文件夹不存在，打印警告
            print(f"警告: 未找到目录 {current_path}")

    if not patient_dirs_to_process:
        print("错误: 在指定范围内未找到任何有效的患者目录，请检查INPUT_DICOM_ROOT设置。")
        return

    # 使用较少的CPU核心进行并行处理，减少错误输出交错
    worker_count = max(4, (os.cpu_count() or 1) // 4)  # 使用1/4的核心，最少4个
    print(f"开始转换 {len(patient_dirs_to_process)} 个患者数据，使用 {worker_count} 个CPU核心...")

    # 使用functools.partial来固定convert_patient_series函数的output_root参数
    process_func = partial(convert_patient_series, output_root=OUTPUT_NIFTI_ROOT)

    # 创建一个进程池并执行转换任务
    with ProcessPoolExecutor(max_workers=worker_count) as executor:
        # 使用tqdm来显示进度条
        results = list(tqdm(executor.map(process_func, patient_dirs_to_process), total=len(patient_dirs_to_process)))

    # 分析处理结果
    success_count = sum(1 for r in results if r["status"] == "success")
    failed_count = len(results) - success_count
    failed_results = [r for r in results if r["status"] == "failed"]
    
    # 创建错误日志文件
    log_file = OUTPUT_NIFTI_ROOT / "conversion_errors.log"
    with open(log_file, 'w', encoding='utf-8') as f:
        f.write(f"DICOM转换错误日志 - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 60 + "\n\n")
        
        if failed_count > 0:
            f.write(f"失败患者总数: {failed_count}\n\n")
            for result in failed_results:
                f.write(f"患者ID: {result['patient_id']}\n")
                f.write(f"错误信息: {result['error']}\n")
                f.write("-" * 40 + "\n")
        else:
            f.write("所有患者转换成功，无错误记录。\n")
    
    # 打印处理结果总结
    print("\n--- 转换完成 ---")
    print(f"成功转换: {success_count} 个患者")
    print(f"转换失败: {failed_count} 个患者")
    
    if failed_count > 0:
        print(f"\n失败详情已保存到: {log_file}")
        print("失败患者列表:")
        for result in failed_results:
            print(f"- {result['patient_id']}: {result['error'].split(':')[0] if ':' in result['error'] else result['error']}")
    else:
        print("所有患者转换成功！")

if __name__ == "__main__":
    main()