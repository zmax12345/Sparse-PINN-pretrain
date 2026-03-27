import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import UnivariateSpline
import os


def calculate_fwhm(y_data):
    """
    使用样条插值计算一维信号的半高全宽 (FWHM)，实现亚像素级精度
    """
    max_val = np.max(y_data)
    half_max = max_val / 2.0

    # 构建样条插值寻找跨越 half_max 的根
    spline = UnivariateSpline(np.arange(len(y_data)), y_data - half_max, s=0)
    roots = spline.roots()

    if len(roots) >= 2:
        return np.abs(roots[-1] - roots[0])
    elif len(roots) == 1:
        # 极少情况：如果只找到一个交点，返回中心到该点距离的2倍
        center = np.argmax(y_data)
        return 2 * np.abs(roots[0] - center)
    else:
        return 0.0


def compute_spatial_autocorr(image):
    """
    基于 FFT 快速计算 2D 空间自相关
    """
    # 转换为 float 并减去均值，消除零频（DC）的巨大背景峰，突出散斑高频特征
    img_f = image.astype(np.float32)
    img_f -= np.mean(img_f)

    # 傅里叶变换 -> 计算功率谱 -> 逆傅里叶变换得到自相关
    F = np.fft.fft2(img_f)
    power_spectrum = np.abs(F) ** 2
    autocorr = np.fft.ifft2(power_spectrum)

    # 将中心移到图像正中间，并取实部
    autocorr = np.fft.fftshift(np.real(autocorr))

    # 归一化使得中心最高点为 1.0
    autocorr /= np.max(autocorr)
    return autocorr


def process_speckle_video(video_path, num_frames=30, roi_size=400):
    """
    处理 MP4 视频，抽取多帧计算散斑平均尺寸
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"错误：无法打开视频文件 {video_path}")
        return

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"视频总帧数: {total_frames}，准备抽取 {num_frames} 帧进行计算...")

    # 均匀采样的帧索引
    frame_indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)

    speckle_sizes = []
    sample_autocorr = None

    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            continue

        # 转换为灰度图
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # 截取中心 ROI 区域，加速计算且避免边缘像差干扰
        h, w = gray.shape
        start_y, start_x = h // 2 - roi_size // 2, w // 2 - roi_size // 2
        roi = gray[start_y:start_y + roi_size, start_x:start_x + roi_size]

        # 计算 2D 空间自相关
        autocorr = compute_spatial_autocorr(roi)

        # 取出自相关矩阵的中心点坐标
        cy, cx = roi_size // 2, roi_size // 2

        # 在中心截取一个 31x31 像素的小窗口来算 FWHM（因为通常散斑只有几个像素大）
        window = 15
        slice_x = autocorr[cy, cx - window: cx + window + 1]
        slice_y = autocorr[cy - window: cy + window + 1, cx]

        # 分别计算横向和纵向的 FWHM，然后取平均
        dx = calculate_fwhm(slice_x)
        dy = calculate_fwhm(slice_y)

        d_mean = (dx + dy) / 2.0
        if d_mean > 0:
            speckle_sizes.append(d_mean)

        # 保存第一帧的自相关结果用于最终可视化
        if sample_autocorr is None:
            sample_autocorr = autocorr[cy - 20:cy + 21, cx - 20:cx + 21]

    cap.release()

    if len(speckle_sizes) == 0:
        print("未能成功计算任何帧的散斑尺寸。")
        return

    final_speckle_size = np.mean(speckle_sizes)
    std_dev = np.std(speckle_sizes)

    print("\n" + "=" * 40)
    print(f"散斑尺寸计算完成！(共分析了 {len(speckle_sizes)} 帧)")
    print(f"平均散斑大小 (d) = {final_speckle_size:.4f} 像素")
    print(f"标准差 = {std_dev:.4f} 像素")
    print("=" * 40 + "\n")

    # ================= 画图可视化 =================
    plt.figure(figsize=(10, 4))

    # 1. 绘制 2D 自相关中心峰的伪彩色图
    plt.subplot(1, 2, 1)
    plt.imshow(sample_autocorr, cmap='jet')
    plt.title('2D Spatial Autocorrelation Peak')
    plt.colorbar()

    # 2. 绘制 1D 切面和 FWHM 示意图
    plt.subplot(1, 2, 2)
    center_idx = sample_autocorr.shape[0] // 2
    profile = sample_autocorr[center_idx, :]
    x_axis = np.arange(len(profile)) - center_idx
    plt.plot(x_axis, profile, 'b-', label='Autocorrelation')

    # 画一条 0.5 (半高) 的横线
    plt.axhline(y=0.5, color='r', linestyle='--', label='Half Maximum (0.5)')
    plt.title(f'1D Profile (FWHM $\\approx$ {final_speckle_size:.2f} px)')
    plt.xlabel('Pixel Distance')
    plt.ylabel('Correlation Coefficient')
    plt.legend()

    plt.tight_layout()
    plt.savefig("/data/zm/Moshaboli/sanbanwuli_BD/speckle_calibration_1.png", dpi=300)
    plt.show()
    print("标定分析图已保存为 'speckle_calibration.png'")


if __name__ == "__main__":
    # 请在这里替换为你拍摄的 MP4 视频路径
    video_file = "/data/zm/Moshaboli/new_data/other_data/bd5.mp4"

    if os.path.exists(video_file):
        # 默认均匀抽取 30 帧取平均，中心截取 400x400 的 ROI 进行计算
        process_speckle_video(video_file, num_frames=30, roi_size=400)
    else:
        print(f"请先将代码底部的 video_file 路径修改为你的 MP4 视频路径！")