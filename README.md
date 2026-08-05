# SERSPeakFinder

SERS(라만) 스펙트럼에서 **피크 유무를 학습·검출**하는 PyQt6 GUI 도구입니다.  
원본 매핑 데이터를 좌표별 CSV로 분리하고, 수동 라벨링 → 모델 학습 → 배치 검출 → 결과 비교 → 적분 정량까지 한 흐름으로 처리합니다.

[한국어](README.md) | [English](README.en.md)

## 워크플로

| 탭 | 기능 |
|---|---|
| **Splite Plot Data** | 원본 txt/csv를 `(x_y).csv` 스펙트럼 파일로 분리 |
| **Plot Classifier** | 피크 있음(1) / 없음(0) 수동 라벨링 (`A` / `D`) |
| **CNN Training** | 이진 분류 모델 학습, loss·accuracy 실시간 표시 |
| **Run (Peak Detection)** | 학습된 `.pth`로 폴더 단위 배치 추론 |
| **Compare with Prev. Plot** | 모델 결과와 수동 분류 결과 비교·리포트 |
| **Interpolation** | Savitzky–Golay 필터 + 선형 보간 기반 피크 영역 적분 |

## 모델

저장소 이름과 달리 실제 네트워크는 **CNN이 아닌 Fully Connected MLP**입니다.

- 입력: 스펙트럼 CSV를 `StandardScaler` 후 1D 벡터로 flatten
- 구조: Linear `input → 128 → 64 → 32 → 1` + ReLU / Dropout / Sigmoid
- 손실: BCELoss
- 구현: `touch_training.py`의 `BinaryClassifier`

사전 학습 가중치 예시: `small_ng_samp_m*.pth` (기본 설정은 `m6`)

## 요구 사항

- Windows 10+
- Python 3.8+
- 주요 패키지: `PyQt6`, `numpy`, `pandas`, `torch`, `scipy`, `matplotlib`, `scikit-learn`, `natsort`, `torchsummary`, `qbstyles`

GPU(CUDA)가 있으면 학습·추론에 사용하고, 없으면 CPU로 동작합니다.

## 설치 및 실행

```bash
python -m venv .venv
.venv\Scripts\activate
pip install PyQt6 numpy pandas torch scipy matplotlib scikit-learn natsort torchsummary qbstyles
python main.py
```

경로·하이퍼파라미터는 `config.ini`에 저장됩니다. UI에서 수정 후 저장하면 다음 실행 시 복원됩니다.

## 주요 파일

| 파일 | 설명 |
|---|---|
| `main.py` | GUI 진입점, 학습/추론 스레드, 탭 로직 |
| `ui_main.py` / `ui_main.ui` | Qt Designer UI |
| `touch_training.py` | 데이터셋·모델·학습/평가 |
| `matplotlibwidget.py` | Matplotlib 캔버스 위젯 |
| `matplotlib_toolbar.py` | 플롯 툴바 |
| `config.ini` | 최근 사용 경로·파라미터 |
| `*.pth` | 학습된 모델 가중치 |
| `main.spec` | PyInstaller 패키징 설정 |

## 데이터 형식

- **원본**: 탭 구분 txt/csv (좌표 열 + 파장별 intensity)
- **스펙트럼 CSV**: 분리된 개별 파일 (좌표별 1파일)
- **라벨 CSV**: `파일명, 피크여부(0/1)`
- **검출 결과 CSV**: 파일명, 예측값, 확률(confidence)

## 참고

- 학습/데이터 경로는 환경에 맞게 `config.ini`를 수정하세요.
