import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os


def extract_robust_mask(csv_path, output_mask_path, crop_rect=None, k_sigma=4.0):
    print(f"正在读取诊断数据: {csv_path} ...")

    # 1. 读取原始数据
    try:
        df = pd.read_csv(csv_path, header=None, names=['row', 'col', 't_in', 't_off'],
                         dtype={'row': np.int32, 'col': np.int32, 't_in': np.int64},
                         on_bad_lines='skip')
    except Exception as e:
        print(f"读取失败: {e}")
        return

    # 2. 空间裁剪
    if crop_rect is not None:
        r_min, r_max, c_min, c_max = crop_rect
        df = df[(df['row'] >= r_min) & (df['row'] < r_max) &
                (df['col'] >= c_min) & (df['col'] < c_max)]
        height, width = r_max - r_min, c_max - c_min
        row_offset, col_offset = r_min, c_min
    else:
        row_offset, col_offset = df['row'].min(), df['col'].min()
        height = df['row'].max() - row_offset + 1
        width = df['col'].max() - col_offset + 1

    # 3. 统计热力图
    pixel_counts = df.groupby(['row', 'col']).size().reset_index(name='event_count')
    heatmap = np.zeros((height, width), dtype=np.int32)
    local_rows = pixel_counts['row'] - row_offset
    local_cols = pixel_counts['col'] - col_offset
    heatmap[local_rows.values, local_cols.values] = pixel_counts['event_count'].values

    # 4. 鲁棒统计学异常检测 (Mean + K * Std)
    mean_val = np.mean(heatmap)
    std_val = np.std(heatmap)
    threshold = mean_val + k_sigma * std_val

    # 找出异常坏点
    bad_mask_local = heatmap > threshold
    bad_count = np.sum(bad_mask_local)

    print("\n" + "=" * 40)
    print("异常像素抓捕报告")
    print("=" * 40)
    print(f"全场平均事件数: {mean_val:.2f}")
    print(f"全场标准差: {std_val:.2f}")
    print(f"自适应惩罚阈值 (Mean + {k_sigma}*Std): {threshold:.2f}")
    print(f"成功抓捕坏点数量: {bad_count} 个 (占比 {(bad_count / (height * width)) * 100:.3f}%)")
    print("=" * 40 + "\n")

    # 5. 画出“抓捕确诊图”供你人工核验
    plt.figure(figsize=(15, 8), dpi=200)

    # 子图1：原始热力图
    plt.subplot(2, 1, 1)
    vmax_val = np.percentile(heatmap[heatmap > 0], 99) if len(heatmap[heatmap > 0]) > 0 else 10
    plt.imshow(heatmap, cmap='hot', aspect='auto', vmin=0, vmax=vmax_val)
    plt.title(f"Original Heatmap (Mean: {mean_val:.1f})", fontsize=12)
    plt.colorbar(label='Event Count')

    # 子图2：被抓捕的坏点位置标红
    plt.subplot(2, 1, 2)
    # 用灰度图显示底图，用纯红点标出坏点
    plt.imshow(heatmap, cmap='gray', aspect='auto', vmin=0, vmax=vmax_val, alpha=0.5)
    bad_y, bad_x = np.where(bad_mask_local)
    plt.scatter(bad_x, bad_y, color='red', s=5, marker='x', alpha=0.8)
    plt.title(f"Detected Hot Pixels (Red Crosses, Threshold: {threshold:.1f}, Count: {bad_count})", fontsize=12)

    plot_path = "/data/zm/Moshaboli/new_data/evaluate/event_count/hot_pixels_verification.png"
    plt.tight_layout()
    plt.savefig(plot_path)
    print(f"抓捕确诊图已保存至: {plot_path} (请务必打开看一眼红叉有没有打准！)")

    # 6. 生成并合并到全局 Mask 中
    # 转换回相机全局坐标 (800 x 1280)
    bad_global_rows = bad_y + row_offset
    bad_global_cols = bad_x + col_offset

    # 初始化一个全新的全画幅 False 矩阵
    global_mask = np.zeros((800, 1280), dtype=bool)

    # 如果你有旧的纯静态 Mask，可以取消下面三行的注释来合并
    # if os.path.exists(output_mask_path):
    #     global_mask = np.load(output_mask_path)

    # 将新抓到的坏点设为 True (屏蔽)
    global_mask[bad_global_rows, bad_global_cols] = True

    np.save(output_mask_path, global_mask)
    print(f"\n终极 Mask 已保存至: {output_mask_path}")


if __name__ == '__main__':
    # 【强烈建议：换成你 0.2 mm/s 的 CSV 数据路径！】
    CSV_PATH = "/data/zm/Moshaboli/new_data/no3/0.2mm_clip.csv"

    # 输出的新 Mask 路径
    OUTPUT_MASK = "/data/zm/Moshaboli/new_data/other_data/hot_pixel_mask_strict.npy"

    # 你的裁剪区域
    CROP = (400, 499, 200, 567)

    # 灵敏度调节：如果红叉没覆盖完所有刺眼白点，把 4.0 调小 (比如 3.5)；如果误杀了斜向波纹，把 4.0 调大 (比如 5.0)
    K_SIGMA = 1.0

    extract_robust_mask(CSV_PATH, OUTPUT_MASK, crop_rect=CROP, k_sigma=K_SIGMA)