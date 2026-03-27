import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os


def analyze_event_distribution(csv_path, output_image_path, crop_rect=None):
    print(f"正在读取数据: {csv_path} ...")

    # 1. 读取原始数据
    try:
        df = pd.read_csv(csv_path, header=None, names=['row', 'col', 't_in', 't_off'],
                         dtype={'row': np.int32, 'col': np.int32, 't_in': np.int64},
                         on_bad_lines='skip')
    except Exception as e:
        print(f"读取失败: {e}")
        return

    if df.empty:
        print("警告：数据为空！")
        return

    print(f"总事件数量: {len(df)} 个")

    # 2. 空间裁剪 (提取你关心的 100 x 368 区域)
    if crop_rect is not None:
        r_min, r_max, c_min, c_max = crop_rect
        print(f"正在裁剪区域: Row[{r_min}:{r_max}], Col[{c_min}:{c_max}]")
        df = df[(df['row'] >= r_min) & (df['row'] < r_max) &
                (df['col'] >= c_min) & (df['col'] < c_max)]
        height = r_max - r_min
        width = c_max - c_min
        row_offset = r_min
        col_offset = c_min
    else:
        # 如果不指定裁剪，就自动贴合数据存在的最小边界
        row_offset = df['row'].min()
        col_offset = df['col'].min()
        height = df['row'].max() - row_offset + 1
        width = df['col'].max() - col_offset + 1
        print(
            f"未指定裁剪，自动边界为: Row[{row_offset}:{row_offset + height - 1}], Col[{col_offset}:{col_offset + width - 1}]")

    # 3. 统计该区域内每个像素的事件触发总数
    print("正在统计各像素事件总数...")
    pixel_counts = df.groupby(['row', 'col']).size().reset_index(name='event_count')

    # 生成 100 x 368 的全零矩阵
    heatmap = np.zeros((height, width), dtype=np.int32)

    # 将统计结果填入矩阵
    local_rows = pixel_counts['row'] - row_offset
    local_cols = pixel_counts['col'] - col_offset
    heatmap[local_rows.values, local_cols.values] = pixel_counts['event_count'].values

    # 4. 统计稀疏度 (这是诊断你分块现象的核心指标！)
    total_pixels = height * width
    zero_pixels = np.sum(heatmap == 0)
    sparsity = (zero_pixels / total_pixels) * 100
    avg_events = np.mean(heatmap)

    print("\n" + "=" * 40)
    print("区域物理统计报告")
    print("=" * 40)
    print(f"视场大小: {height} x {width} ({total_pixels} 像素)")
    print(f"完全没有触发过事件的像素(盲区): {zero_pixels} 个")
    print(f"空间极度稀疏率 (绝对盲区占比): {sparsity:.2f} %")
    print(f"每个像素平均累计事件: {avg_events:.2f} 个")
    print("=" * 40 + "\n")

    # 5. 画图可视化
    plt.figure(figsize=(15, 4), dpi=300)

    # 【核心画图逻辑】：为了防止 1 个死像素有 10000 次事件，导致其他几百次事件的正常像素全部变成纯黑
    # 我们计算 99% 的分位数作为最高亮度阈值
    valid_events = heatmap[heatmap > 0]
    if len(valid_events) > 0:
        vmax_val = np.percentile(valid_events, 99)
    else:
        vmax_val = 10

    # 使用 'hot' 伪彩图 (黑->红->黄->白)，绝对纯黑代表 0 事件
    im = plt.imshow(heatmap, cmap='hot', aspect='auto', interpolation='none', vmin=0, vmax=vmax_val)
    cbar = plt.colorbar(im)
    cbar.set_label(f'Event Count\n(Capped at 99th percentile: {vmax_val:.1f})', fontsize=10)

    plt.title(f"Spatial Event Distribution Heatmap ({height}x{width}) | Sparsity: {sparsity:.1f}%", fontsize=14)
    plt.xlabel("Width (Local Col)", fontsize=12)
    plt.ylabel("Height (Local Row)", fontsize=12)
    plt.tight_layout()

    plt.savefig(output_image_path)
    plt.close()
    print(f"-> 绝美热力图已保存至: {output_image_path}")


if __name__ == '__main__':
    # 【请替换为你录制的一段有流速的 CSV 数据路径】
    # 建议选一段中流速 (1.2-1.8 mm/s) 的数据，也就是你觉得分块最暧昧的那段
    CSV_PATH = "/data/zm/Moshaboli/new_data/no3/0.2mm_clip.csv"

    # 输出图片路径
    OUTPUT_IMAGE = "/data/zm/Moshaboli/new_data/evaluate/event_count/event_spatial_heatmap.png"

    # 【关键设置】：输入你 dataset.py 里面切片的具体坐标
    # 格式为 (row_start, row_end, col_start, col_end)
    # 假设你的中心区域是 row 从 350 到 450，col 从 456 到 824
    # 如果你想看全图，把这个改成 CROP = None
    CROP = (400, 499, 200, 567)

    analyze_event_distribution(CSV_PATH, OUTPUT_IMAGE, crop_rect=CROP)