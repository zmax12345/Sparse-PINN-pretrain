import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os


def analyze_and_generate_mask(static_csv_path, output_mask_path, threshold_hz=2.0):
    print(f"正在读取静态背景数据: {static_csv_path} ...")

    # 1. 读取数据 (保持与你 dataset 一致的读取格式)
    try:
        df = pd.read_csv(static_csv_path, header=None, names=['row', 'col', 't_in', 't_off'],
                         dtype={'row': np.int32, 'col': np.int32, 't_in': np.int64, 't_off': np.int64},
                         on_bad_lines='skip')
    except Exception as e:
        print(f"读取失败: {e}")
        return

    if df.empty:
        print("警告：数据为空！")
        return

    # 2. 计算录制总物理时长 (假设 t_in 单位是微秒)
    t_start = df['t_in'].min()
    t_end = df['t_in'].max()
    duration_s = (t_end - t_start) / 1_000_000.0  # 转换为秒
    print(f"数据总物理时长: {duration_s:.3f} 秒")
    print(f"总事件数量: {len(df)} 个")

    # 3. 统计每个像素触发的绝对次数
    print("正在统计各像素触发频率...")
    pixel_counts = df.groupby(['row', 'col']).size().reset_index(name='event_count')

    # 4. 计算每个像素的物理频率 (Hz)
    pixel_counts['freq_hz'] = pixel_counts['event_count'] / duration_s

    # 5. 画出频率分布直方图 (核心诊断工具)
    plt.figure(figsize=(10, 6), dpi=150)
    # 为了看清长尾效应，我们将 y 轴设置为对数坐标，并只关注 0-10 Hz 之间的分布
    bins = np.linspace(0, 10, 100)
    plt.hist(pixel_counts['freq_hz'], bins=bins, color='steelblue', edgecolor='black', alpha=0.7)

    plt.axvline(x=threshold_hz, color='red', linestyle='--', linewidth=2, label=f'Threshold ({threshold_hz} Hz)')
    plt.yscale('log')  # y 轴必须是对数，因为正常像素极多，死像素极少
    plt.title("Static Background Pixel Frequency Distribution", fontsize=14)
    plt.xlabel("Firing Frequency (Hz)", fontsize=12)
    plt.ylabel("Number of Pixels (Log Scale)", fontsize=12)
    plt.legend()
    plt.grid(axis='y', alpha=0.3)

    plot_path = "/data/zm/Moshaboli/new_data/other_data/pixel_frequency_distribution.png"
    plt.savefig(plot_path)
    print(f"频率分布直方图已保存至: {plot_path} (请务必打开查看！)")

    # 6. 生成布尔型 Mask 矩阵 (800 x 1280 是相机的全分辨率)
    # 注意：True 表示该像素是死像素（需要被过滤），False 表示正常
    mask = np.zeros((800, 1280), dtype=bool)

    # 找出所有频率大于等于阈值的死像素
    hot_pixels = pixel_counts[pixel_counts['freq_hz'] >= threshold_hz]

    # 在 Mask 矩阵中将这些位置设为 True
    mask[hot_pixels['row'].values, hot_pixels['col'].values] = True

    # 7. 保存并打印统计结果
    np.save(output_mask_path, mask)
    total_pixels = 800 * 1280
    hot_count = len(hot_pixels)
    kill_ratio = (hot_count / total_pixels) * 100

    print("\n" + "=" * 40)
    print(f"Mask 生成报告")
    print("=" * 40)
    print(f"设定阈值: {threshold_hz} Hz")
    print(f"判定为死像素数量: {hot_count} 个")
    print(f"占全画幅比例: {kill_ratio:.4f} %")
    print(f"Mask 文件已成功保存至: {output_mask_path}")
    print("=" * 40)


if __name__ == "__main__":
    # 【请在这里填入你录制的那段“完全静止”的 CSV 数据路径】
    STATIC_DATA_PATH = "/data/zm/Moshaboli/new_data/other_data/mask (2).csv"

    # 【输出的新 Mask 路径】
    OUTPUT_MASK = "/data/zm/Moshaboli/new_data/other_data/new_2hz_mask.npy"

    # 你可以先设为 2.0，运行完看一眼图片，再决定要不要修改
    TARGET_THRESHOLD = 2.0

    analyze_and_generate_mask(STATIC_DATA_PATH, OUTPUT_MASK, threshold_hz=TARGET_THRESHOLD)