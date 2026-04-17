import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import savgol_filter
import matplotlib.gridspec as gridspec  # 비율 조정을 위한 gridspec 추가

center_wl = 1000
delta_wl = 10
file_name = 'small ng samp 2, 1_5'

# txt 파일 읽기 (가정: df는 적절히 읽은 데이터프레임)
df = pd.read_csv(f'test_data/{file_name}.txt', delimiter='\t')

# 첫 두 열은 x, y 좌표
x_coords = df['Unnamed: 0']
y_coords = df['Unnamed: 1']

# 나머지 열은 각각 wavelengths과 intensities를 의미함
wavelengths = df.columns[2:].astype(float)  # wavelengths 정보
intensities = df.iloc[:, 2:]  # intensities 정보

file_list = []

def normalizedata(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))

for i in range(len(df)):
    # 특정 좌표에서의 스펙트럼 (예: 첫 번째 좌표)
    x = x_coords[i]
    y = y_coords[i]
    print(f'Curr. : {x}, {y}')
    intensity_at_point = intensities.iloc[i].to_numpy()
    intensity_at_point = normalizedata(intensity_at_point)

    # Savitzky-Golay 필터 적용 (window_length=5, polyorder=2는 예시로 설정)
    filtered_intensity = savgol_filter(intensity_at_point, window_length=20, polyorder=2)

    # 관심 영역 설정
    peak_start = center_wl - delta_wl
    peak_end = center_wl + delta_wl
    mask = (wavelengths >= peak_start) & (wavelengths <= peak_end)

    # 피크 밖의 값들을 위한 보간
    linear_interp_filtered = np.interp(wavelengths[mask], [peak_start, peak_end],
                                       [filtered_intensity[wavelengths < peak_start][-1],
                                        filtered_intensity[wavelengths > peak_end][0]])
    # 필터링된 데이터에 반영
    linear_intensity_filtered = filtered_intensity.copy()
    linear_intensity_filtered[mask] = linear_interp_filtered

    # Savitzky-Golay 필터 적용 후 선형 보간 값을 뺀 결과
    difference_intensity_filtered = filtered_intensity - linear_intensity_filtered

    # SUM difference_intensity_filtered Data
    integ_val = round(np.sum(difference_intensity_filtered), 3)

    # 4. 플롯
    plt.figure(figsize=(15, 12))

    # 그리드 설정: 3:1 비율로 너비 설정
    gs = gridspec.GridSpec(3, 2, width_ratios=[3, 1])  # 3:1 비율 적용

    # (1) Savitzky-Golay 필터 결과 + Raw Data
    plt.subplot(gs[0, 0])  # 첫 번째 행, 왼쪽 플롯
    plt.plot(wavelengths, intensity_at_point, label='Raw Data', color='black', linestyle='solid')
    plt.plot(wavelengths, filtered_intensity, label='Savitzky-Golay Filtered', color='blue')
    plt.axvline(x=peak_start, ymin=0, ymax=1, linewidth=1, linestyle="--", color='red')
    plt.axvline(x=peak_end, ymin=0, ymax=1, linewidth=1, linestyle="--", color='red')
    plt.axvspan(peak_start, peak_end, alpha=0.1, color='red')
    plt.title('Savitzky-Golay Filtered Data with Raw Data')
    plt.xlabel('Wavelength (cm^-1)')
    plt.ylabel('Intensity (a.u.)')
    plt.legend()

    # (1) Zoom-in Plot: 첫 번째 플롯의 확대
    plt.subplot(gs[0, 1])  # 첫 번째 행, 오른쪽 플롯
    plt.plot(wavelengths[mask], intensity_at_point[mask], label='Raw Data (Zoom)', color='black', linestyle='solid')
    plt.plot(wavelengths[mask], filtered_intensity[mask], label='Filtered (Zoom)', color='blue')
    plt.title(f'Zoom-in: {center_wl} ± {delta_wl} cm^-1')
    plt.xlabel('Wavelength (cm^-1)')
    plt.ylabel('Intensity (a.u.)')
    plt.legend().remove()

    # (2) Savitzky-Golay 필터에서 선형 보간된 결과
    plt.subplot(gs[1, 0])  # 두 번째 행, 왼쪽 플롯
    plt.plot(wavelengths, filtered_intensity, label='Savitzky-Golay Filtered', color='blue')
    plt.plot(wavelengths, linear_intensity_filtered, label='Linear Interpolation on Filtered Data', color='green')
    plt.axvline(x=peak_start, ymin=0, ymax=1, linewidth=1, linestyle="--", color='gray', alpha=0.3)
    plt.axvline(x=peak_end, ymin=0, ymax=1, linewidth=1, linestyle="--", color='gray', alpha=0.3)
    plt.fill_between(wavelengths[0:600], filtered_intensity[0:600], linear_intensity_filtered[0:600], color='red', alpha=0.5)
    plt.title('Linear Interpolation on Savitzky-Golay Filtered Data')
    plt.xlabel('Wavelength (cm^-1)')
    plt.ylabel('Intensity (a.u.)')
    plt.legend()

    # (2) Zoom-in Plot: 두 번째 플롯의 확대
    plt.subplot(gs[1, 1])  # 두 번째 행, 오른쪽 플롯
    plt.plot(wavelengths[mask], filtered_intensity[mask], label='Filtered (Zoom)', color='blue')
    plt.plot(wavelengths[mask], linear_intensity_filtered[mask], label='Linear Interpolation (Zoom)', color='green')
    plt.fill_between(wavelengths[mask], filtered_intensity[mask], linear_intensity_filtered[mask], color='red', alpha=0.5)
    plt.title(f'Zoom-in: {center_wl} ± {delta_wl} cm^-1')
    plt.xlabel('Wavelength (cm^-1)')
    plt.ylabel('Intensity (a.u.)')
    plt.legend().remove()

    # (3) Savitzky-Golay 필터 값 - 선형 보간 값
    plt.subplot(gs[2, 0])  # 세 번째 행, 왼쪽 플롯
    plt.plot(wavelengths, difference_intensity_filtered, label='Difference (Filtered - Linear)', color='red')
    # plt.axvspan(peak_start, peak_end, alpha=0.1, color='red')
    plt.title(f'Difference: Savitzky-Golay - Linear Interpolation \nPOS. : [{x}, {y}] SUM : {integ_val}')
    plt.xlabel('Wavelength (cm^-1)')
    plt.ylabel('Intensity (a.u.)')
    plt.legend()

    # (3) Zoom-in Plot: 세 번째 플롯의 확대
    plt.subplot(gs[2, 1])  # 세 번째 행, 오른쪽 플롯
    plt.plot(wavelengths[mask], difference_intensity_filtered[mask], label='Difference (Zoom)', color='red')
    plt.title(f'Zoom-in: {center_wl} ± {delta_wl} cm^-1')
    plt.xlabel('Wavelength (cm^-1)')
    plt.ylabel('Intensity (a.u.)')
    plt.legend().remove()

    # 플롯 보여주기 및 저장
    plt.tight_layout()
    # plt.show()
    plt.savefig(f'imgs/{file_name}_({x}_{y}).png')

    # 데이터 프레임 생성
    df = pd.DataFrame({
        'Wavelength': wavelengths,
        'Intensity': intensity_at_point
    })

    # CSV 파일로 저장
    df.to_csv(f's_data/{file_name}_({x}_{y}).csv', index=False)
    file_list.append(f'{file_name}_({x}_{y}).csv')


# pandas의 DataFrame으로 변환 (한 열로 변환)
df = pd.DataFrame(file_list, columns=['Value'])
# CSV 파일로 저장
df.to_csv('peaks_info2.csv', index=False)
