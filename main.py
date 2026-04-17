import sys, os, io, re
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import glob
import numpy as np
import pandas as pd
from PyQt6 import QtWidgets, QtCore, QtGui
from ui_main import Ui_MainWindow
from natsort import os_sorted
from configparser import ConfigParser
import gc
from scipy.signal import savgol_filter
from scipy.fftpack import fft, ifft

from touch_training import BinaryClassifier, GraphDataset, GraphDataset2
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchsummary import summary


class RunModelThread(QtCore.QThread):
    update_text = QtCore.pyqtSignal(str)

    def __init__(self, model_file, folder_path, batch_size):
        super(RunModelThread, self).__init__()
        self.MODEL_PATH = model_file  # 저장된 모델 경로
        DATA_FOLDER = fr'{folder_path}'  # 예측할 데이터가 있는 폴더
        BATCH_SIZE = batch_size

        self.report_file = []
        self.report_peak = []
        self.report_cs = []

        # 장치 설정
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # print(f"Using device: {self.device}")

        # 데이터셋 생성 (예측용)
        self.dataset = GraphDataset2(DATA_FOLDER)  # info_file 없이 생성
        self.data_loader = DataLoader(self.dataset, batch_size=BATCH_SIZE, shuffle=False)

    def run(self):
        # 모델 로드
        input_size = self.dataset[0][0].shape[0]  # 첫 번째 샘플의 특성 수
        self.model = BinaryClassifier(input_size)

        checkpoint = torch.load(self.MODEL_PATH, map_location=self.device, weights_only=True)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)

        # print(f"Model loaded from {self.MODEL_PATH}")
        # print(f"Best accuracy from training: {checkpoint['best_acc']:.2f}%")
        self.update_text.emit(f"Best accuracy from training: {checkpoint['best_acc']:.2f}%")

        # 예측 수행
        results_df = self.predict_samples()

        self.report_file = results_df['filename'].to_list()
        self.report_peak = results_df['prediction'].to_list()
        self.report_cs = results_df['probability'].to_list()

        # 결과 저장
        # output_file = r'./temp/prediction_results.csv'
        # results_df.to_csv(output_file, index=False)
        # print(f"\nPrediction results saved to {output_file}")

        # 결과 요약 출력
        # print("\nPrediction Summary:")
        self.update_text.emit(f"Total samples : {len(results_df)} "
                              f"[ Predicted positive: {(results_df['prediction'] == 1).sum()} ] "
                              f"[ Predicted negative: {(results_df['prediction'] == 0).sum()} ]")
        # self.update_text.emit(f"Total samples: {len(results_df)}")
        # self.update_text.emit(f"Predicted positive: {(results_df['prediction'] == 1).sum()}")
        # self.update_text.emit(f"Predicted negative: {(results_df['prediction'] == 0).sum()}")
        self.update_text.emit("Done")

    def predict_samples(self):
        self.model.eval()
        predictions = []
        filenames = []
        probabilities = []

        with torch.no_grad():
            for inputs, _, batch_filenames in self.data_loader:
                inputs = inputs.to(self.device)
                outputs = self.model(inputs)
                probs = outputs.squeeze().cpu().numpy()
                preds = (probs > 0.5).astype(int)

                predictions.extend(preds)
                probabilities.extend(probs)
                filenames.extend(batch_filenames)

        # 결과를 데이터프레임으로 변환
        results_df = pd.DataFrame({
            'filename': filenames,
            'prediction': predictions,
            'probability': probabilities
        })

        return results_df

    def predict_samples_2(self):
        self.model.eval()
        predictions = []
        filenames = []
        probabilities = []

        with torch.no_grad():
            total_batches = len(self.data_loader)
            for batch_idx, (inputs, batch_filenames) in enumerate(self.data_loader, start=1):
                inputs = inputs.to(self.device)
                outputs = self.model(inputs)
                probs = outputs.squeeze().cpu().numpy()
                preds = (probs > 0.5).astype(int)

                predictions.extend(preds)
                probabilities.extend(probs)
                filenames.extend(batch_filenames)

                # 진행 상황 업데이트
                self.update_text.emit(f"Batch {batch_idx}/{total_batches} processed.")
                # print(f"Batch {batch_idx}/{total_batches} processed.")

        # 결과를 데이터프레임으로 변환
        results_df = pd.DataFrame({
            'filename': filenames,
            'prediction': predictions,
            'probability': probabilities
        })

        return results_df


class TrainModelThread(QtCore.QThread):
    update_text = QtCore.pyqtSignal(str)

    def __init__(self, folder_path, info_file, test_size=0.2, random_state=42,
                 epochs=300, batch_size=64, validation_split=0.2, save_model='peak_detection_model.pth'):
        super(TrainModelThread, self).__init__()  # QThread 초기화

        # 하이퍼파라미터 설정
        self.BATCH_SIZE = int(batch_size)
        self.EPOCHS = int(epochs)
        self.LEARNING_RATE = 0.001
        self.save_model = save_model

        # 장치 설정
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # 데이터셋 생성
        dataset = GraphDataset(folder_path=folder_path, info_file=info_file)

        # 학습/테스트 데이터 분할
        train_size = int((1-validation_split) * len(dataset))
        test_size = len(dataset) - train_size
        train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

        # 데이터로더 생성
        # self.train_loader = DataLoader(train_dataset, batch_size=self.BATCH_SIZE, shuffle=True)
        self.train_loader = DataLoader(train_dataset, batch_size=self.BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
        # self.test_loader = DataLoader(test_dataset, batch_size=self.BATCH_SIZE, shuffle=True)
        self.test_loader = DataLoader(test_dataset, batch_size=self.BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)

        # 입력 크기 계산
        sample_data, _ = dataset[0]
        self.input_size = sample_data.shape[0]


        # 모델 초기화
        self.model = BinaryClassifier(self.input_size).to(self.device)
        self.criterion = nn.BCELoss()  # Binary Cross Entropy Loss
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.LEARNING_RATE)

        # summary(self.model, input_size=(self.input_size,))

    def run(self):
        # 모델 학습
        best_acc = 0.0
        self.update_text.emit(f"Using device: {self.device}")
        self.update_text.emit(f"Input size: {self.input_size}")
        summary_text = self.capture_summary(self.model, input_size=(self.input_size,))
        self.update_text.emit(f"{summary_text}")
        self.update_text.emit("Starting training...\n")
        for epoch in range(self.EPOCHS):
            train_loss, train_acc = self.train_model(self.model, self.train_loader, self.criterion, self.optimizer, self.device)
            test_loss, test_acc = self.evaluate_model(self.model, self.test_loader, self.criterion, self.device)

            self.update_text.emit(f'Epoch [{epoch + 1}/{self.EPOCHS}]\n'
                                  f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%\n'
                                  f'Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%\n')

            # 최고 성능 모델 저장
            if test_acc > best_acc:
                best_acc = test_acc
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'best_acc': best_acc,
                }, f'{self.save_model}')

        self.update_text.emit(f'Best Test Accuracy: {best_acc:.2f}%')

    def capture_summary(self, model, input_size):
        buffer = io.StringIO()
        sys.stdout = buffer  # 표준 출력 리다이렉션
        summary(model, input_size=input_size)
        sys.stdout = sys.__stdout__  # 표준 출력 복원
        return buffer.getvalue()

    def train_model(self, model, train_loader, criterion, optimizer, device):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.float().to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            outputs = outputs.squeeze()

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            predicted = (outputs > 0.5).float()
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        return running_loss / len(train_loader), 100 * correct / total

    # 평가 함수
    def evaluate_model(self, model, test_loader, criterion, device):
        model.eval()
        running_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs = inputs.to(device)
                labels = labels.float().to(device)

                outputs = model(inputs)
                outputs = outputs.squeeze()

                loss = criterion(outputs, labels)
                running_loss += loss.item()

                predicted = (outputs > 0.5).float()
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        return running_loss / len(test_loader), 100 * correct / total


class MainWindow(QtWidgets.QMainWindow, Ui_MainWindow):
    def __init__(self, parent=None):
        super(MainWindow, self).__init__(parent)
        # Configuration 파일 경로 설정
        self.config_file = 'config.ini'
        self.setupUi(self)

        self.tab_index = 0

        self.ToolbarWidget.read(canvas=self.plot_widget)
        self.ToolbarWidget_2.read(canvas=self.plot_widget_2)
        self.ToolbarWidget_3.read(canvas=self.plot_widget_5)
        self.ToolbarWidget_4.read(canvas=self.plot_widget_6)
        self.ToolbarWidget_5.read(canvas=self.plot_widget_7)

        self.textEdit.setFont(QtGui.QFont("Consolas"))

        # listView용 모델 초기화
        self.list_model = QtGui.QStandardItemModel(self.listView)
        self.listView.setModel(self.list_model)

        # 트리뷰와 모델 설정
        self.model = QtGui.QStandardItemModel()
        self.treeView.setModel(self.model)

        # 트리뷰의 모델 설정 (헤더 추가)
        self.model.setHorizontalHeaderLabels(["File Name"])

        # 클릭 이벤트 연결
        self.listView.clicked.connect(self.display_spectrum_from_selected_file)
        self.treeView.clicked.connect(self.display_spectrum_from_selected_file_2)
        self.tableWidget_2.cellClicked.connect(self.on_table_click)
        self.tableWidget_4.cellClicked.connect(self.on_table_click)
        self.tableWidget_3.cellClicked.connect(self.on_table_click)

        # 버튼 클릭 시 슬롯 연결
        self.pushButton_7.clicked.connect(self.move_to_previous_item)
        self.pushButton_8.clicked.connect(self.move_to_next_item)

        # 버튼 클릭 연결
        self.pushButton_4.clicked.connect(lambda: self.open_file_dialog(self.lineEdit_2))
        self.pushButton_15.clicked.connect(lambda: self.open_file_dialog(self.lineEdit_8))
        self.pushButton_18.clicked.connect(lambda: self.open_file_dialog(self.lineEdit_9))
        self.pushButton_23.clicked.connect(lambda: self.open_file_dialog(self.lineEdit_16))
        self.pushButton_24.clicked.connect(lambda: self.open_file_dialog(self.lineEdit_17))
        self.pushButton_22.clicked.connect(lambda: self.open_file_dialog(self.lineEdit_20))
        self.pushButton_2.clicked.connect(lambda: self.open_folder_dialog(self.lineEdit))
        self.pushButton_5.clicked.connect(self.open_folder_dialog_2)
        self.pushButton_13.clicked.connect(lambda: self.open_folder_dialog(self.lineEdit_6))
        self.pushButton_21.clicked.connect(lambda: self.open_folder_dialog(self.lineEdit_10))
        self.pushButton_28.clicked.connect(lambda: self.open_folder_dialog(self.lineEdit_19))
        self.pushButton_29.clicked.connect(lambda: self.open_folder_dialog(self.lineEdit_21))
        self.pushButton.clicked.connect(self.process_file)
        self.pushButton_11.clicked.connect(self.classifier_run)
        self.pushButton_20.clicked.connect(self.run_detect_run)
        self.pushButton_16.clicked.connect(self.start_training)
        self.pushButton_14.clicked.connect(self.save_config)
        self.pushButton_19.clicked.connect(self.start_detection)
        self.pushButton_25.clicked.connect(self.matched_result)
        self.pushButton_26.clicked.connect(self.save_to_csv_index4)
        self.pushButton_30.clicked.connect(self.process_data)

        # 버튼 클릭 시 슬롯 연결
        self.pushButton_9.clicked.connect(lambda: self.save_file(1))
        self.pushButton_10.clicked.connect(lambda: self.save_file(0))

        # Shortcut key setting
        self.shortcut_prev = QtGui.QShortcut(QtGui.QKeySequence("Up"), self)
        self.shortcut_prev.activated.connect(self.move_to_previous_item)
        self.shortcut_prev.setEnabled(False)
        self.shortcut_next = QtGui.QShortcut(QtGui.QKeySequence("Down"), self)
        self.shortcut_next.activated.connect(self.move_to_next_item)
        self.shortcut_next.setEnabled(False)
        self.shortcut_o = QtGui.QShortcut(QtGui.QKeySequence("A"), self)
        self.shortcut_o.activated.connect(lambda: self.save_file(1))
        self.shortcut_o.setEnabled(False)
        self.shortcut_x = QtGui.QShortcut(QtGui.QKeySequence("D"), self)
        self.shortcut_x.activated.connect(lambda: self.save_file(0))
        self.shortcut_x.setEnabled(False)

        # 숏컷 생성 및 기본 비활성화 상태로 설정
        self.up_shortcut = QtGui.QShortcut(QtGui.QKeySequence(QtCore.Qt.Key.Key_Up), self.tableWidget_2)
        self.up_shortcut.activated.connect(lambda: self.move_to_previous_table_item(self.tableWidget_2))
        self.up_shortcut.setEnabled(False)

        self.down_shortcut = QtGui.QShortcut(QtGui.QKeySequence(QtCore.Qt.Key.Key_Down), self.tableWidget_2)
        self.down_shortcut.activated.connect(lambda: self.move_to_next_table_item(self.tableWidget_2))
        self.down_shortcut.setEnabled(False)

        self.up_shortcut_2 = QtGui.QShortcut(QtGui.QKeySequence(QtCore.Qt.Key.Key_Up), self.tableWidget_4)
        self.up_shortcut_2.activated.connect(lambda: self.move_to_previous_table_item(self.tableWidget_4))
        self.up_shortcut_2.setEnabled(False)

        self.down_shortcut_2 = QtGui.QShortcut(QtGui.QKeySequence(QtCore.Qt.Key.Key_Down), self.tableWidget_4)
        self.down_shortcut_2.activated.connect(lambda: self.move_to_next_table_item(self.tableWidget_4))
        self.down_shortcut_2.setEnabled(False)

        self.up_shortcut_3 = QtGui.QShortcut(QtGui.QKeySequence(QtCore.Qt.Key.Key_Up), self.tableWidget_3)
        self.up_shortcut_3.activated.connect(lambda: self.move_to_previous_table_item(self.tableWidget_3))
        self.up_shortcut_3.setEnabled(False)

        self.down_shortcut_3 = QtGui.QShortcut(QtGui.QKeySequence(QtCore.Qt.Key.Key_Down), self.tableWidget_3)
        self.down_shortcut_3.activated.connect(lambda: self.move_to_next_table_item(self.tableWidget_3))
        self.down_shortcut_3.setEnabled(False)

        # 탭 변경 이벤트 연결
        self.tabWidget.currentChanged.connect(self.on_tab_changed)

        # 설정 파일 로드 시도
        if os.path.exists(self.config_file):
            self.load_config()
        else:
            QtWidgets.QMessageBox.warning(self, "Warning", "Config file not found. Proceeding without it.")

    def save_config(self):
        # ConfigParser 설정
        config = ConfigParser()
        config['LineEdits'] = {
            'lineEdit': self.lineEdit.text(),
            'lineEdit_2': self.lineEdit_2.text(),
            'lineEdit_3': self.lineEdit_3.text(),
            'lineEdit_4': self.lineEdit_4.text(),
            'lineEdit_5': self.lineEdit_5.text(),
            'lineEdit_6': self.lineEdit_6.text(),
            'lineEdit_7': self.lineEdit_7.text(),
            'lineEdit_8': self.lineEdit_8.text(),
            'lineEdit_9': self.lineEdit_9.text(),
            'lineEdit_10': self.lineEdit_10.text(),
            'lineEdit_11': self.lineEdit_11.text(),
            'lineEdit_12': self.lineEdit_12.text(),
            'lineEdit_13': self.lineEdit_13.text(),
            'lineEdit_14': self.lineEdit_14.text(),
            'lineEdit_15': self.lineEdit_15.text(),
            'lineEdit_16': self.lineEdit_16.text(),
            'lineEdit_17': self.lineEdit_17.text(),
            'lineEdit_19': self.lineEdit_19.text(),
            'lineEdit_20': self.lineEdit_20.text(),
            'lineEdit_21': self.lineEdit_21.text(),
            'lineEdit_22': self.lineEdit_22.text(),
            'lineEdit_23': self.lineEdit_23.text(),
            'lineEdit_24': self.lineEdit_24.text(),
            'lineEdit_25': self.lineEdit_25.text(),
        }

        # Config 파일 저장
        with open(self.config_file, 'w') as configfile:
            config.write(configfile)
        # print("Config saved successfully.")

    def load_config(self):
        # Config 파일 로드
        config = ConfigParser()
        config.read(self.config_file)

        if 'LineEdits' in config:
            self.lineEdit.setText(config.get('LineEdits', 'lineEdit', fallback=''))
            self.lineEdit_2.setText(config.get('LineEdits', 'lineEdit_2', fallback=''))
            self.lineEdit_3.setText(config.get('LineEdits', 'lineEdit_3', fallback=''))
            self.lineEdit_4.setText(config.get('LineEdits', 'lineEdit_4', fallback=''))
            self.lineEdit_5.setText(config.get('LineEdits', 'lineEdit_5', fallback=''))
            self.lineEdit_6.setText(config.get('LineEdits', 'lineEdit_6', fallback=''))
            self.lineEdit_7.setText(config.get('LineEdits', 'lineEdit_7', fallback=''))
            self.lineEdit_8.setText(config.get('LineEdits', 'lineEdit_8', fallback=''))
            self.lineEdit_9.setText(config.get('LineEdits', 'lineEdit_9', fallback=''))
            self.lineEdit_10.setText(config.get('LineEdits', 'lineEdit_10', fallback=''))
            self.lineEdit_11.setText(config.get('LineEdits', 'lineEdit_11', fallback=''))
            self.lineEdit_12.setText(config.get('LineEdits', 'lineEdit_12', fallback=''))
            self.lineEdit_13.setText(config.get('LineEdits', 'lineEdit_13', fallback=''))
            self.lineEdit_14.setText(config.get('LineEdits', 'lineEdit_14', fallback=''))
            self.lineEdit_15.setText(config.get('LineEdits', 'lineEdit_15', fallback=''))
            self.lineEdit_16.setText(config.get('LineEdits', 'lineEdit_16', fallback=''))
            self.lineEdit_17.setText(config.get('LineEdits', 'lineEdit_17', fallback=''))
            self.lineEdit_19.setText(config.get('LineEdits', 'lineEdit_19', fallback=''))
            self.lineEdit_20.setText(config.get('LineEdits', 'lineEdit_20', fallback=''))
            self.lineEdit_21.setText(config.get('LineEdits', 'lineEdit_21', fallback=''))
            self.lineEdit_22.setText(config.get('LineEdits', 'lineEdit_22', fallback=''))
            self.lineEdit_23.setText(config.get('LineEdits', 'lineEdit_23', fallback=''))
            self.lineEdit_24.setText(config.get('LineEdits', 'lineEdit_24', fallback=''))
            self.lineEdit_25.setText(config.get('LineEdits', 'lineEdit_25', fallback=''))

    def on_table_click(self):
        table_widget = None
        plot_widget = None
        if self.tab_index == 3:
            table_widget = self.tableWidget_2
            plot_widget = self.plot_widget_5
        elif self.tab_index == 4:
            table_widget = self.tableWidget_4
            plot_widget = self.plot_widget_6
        elif self.tab_index == 5:
            table_widget = self.tableWidget_3
            plot_widget = self.plot_widget_7
        row = table_widget.currentRow()
        if row != -1:
            try:
                column_0_value = ''
                if table_widget == self.tableWidget_2:
                    column_0_value = self.lineEdit_10.text() + '/' + table_widget.item(row, 0).text()
                    pic = table_widget.item(row, 1).text()
                    if pic == '1':
                        line_c = 'b'
                    elif pic == '0':
                        line_c = 'gray'
                elif table_widget == self.tableWidget_4:
                    column_0_value = self.lineEdit_19.text() + '/' + table_widget.item(row, 0).text()
                    pic = table_widget.item(row, 4).text()
                    if pic == 'o':
                        line_c = 'g'
                    elif pic == 'x':
                        line_c = 'gray'
                elif table_widget == self.tableWidget_3:
                    column_0_value = self.lineEdit_21.text() + '/' + table_widget.item(row, 0).text()
                    line_c = 'b'
                # print(f"Selected row: {row}, Column 0 value: {column_0_value}")
                plot_widget.axes.cla()
                if os.path.isfile(column_0_value):
                    if not table_widget == self.tableWidget_3:
                        plot_widget.axes.cla()
                        spectrum_df = pd.read_csv(column_0_value)
                        wavelengths = spectrum_df['Wavelength'].to_numpy()
                        intensities = spectrum_df['Intensity'].to_numpy()
                        plot_widget.axes.plot(wavelengths, intensities, color=f'{line_c}')
                        plot_widget.axes.set_xlabel('Wavelength (cm^-1)', fontsize=12)
                        plot_widget.axes.set_ylabel('Intensity (a.u.)', fontsize=12)
                        plot_widget.axes.tick_params(axis="both", direction="in", pad=10, labelsize=10)
                        peak_start = float(self.lineEdit_4.text()) - float(self.lineEdit_5.text())
                        peak_end = float(self.lineEdit_4.text()) + float(self.lineEdit_5.text())
                        plot_widget.axes.axvline(x=peak_start, ymin=0, ymax=1, linewidth=1, linestyle="--",
                                                        color='k')
                        plot_widget.axes.axvline(x=peak_end, ymin=0, ymax=1, linewidth=1, linestyle="--",
                                                        color='k')
                        plot_widget.axes.set_title(f'{column_0_value}')
                        plot_widget.figure.tight_layout()
                        plot_widget.draw()
                    else:
                        plot_widget.clear_all()
                        window_length = int(self.lineEdit_22.text())
                        polyorder = int(self.lineEdit_23.text())
                        center_wavelength = float(self.lineEdit_24.text())
                        range_value = float(self.lineEdit_25.text())

                        wavelength, intensity, filtered_intensity, interpolated_intensity = self.process_file_interpolation(
                            file_path=column_0_value,
                            window_length=window_length,
                            polyorder=polyorder,
                            center_wavelength=center_wavelength,
                            range_value=range_value
                        )

                        plot_widget.rows = 3
                        plot_widget.cols = 1
                        plot_widget.set_subplot(1)
                        # plot_widget.axes.set_xlabel('Wavelength (cm^-1)', fontsize=12)
                        plot_widget.axes.set_ylabel('Intensity (a.u.)', fontsize=12)
                        plot_widget.axes.set_title('Savitzky-Golay Filtered Data with Raw Data')

                        plot_widget.axes.tick_params(axis="both", direction="in", pad=10, labelsize=10)
                        plot_widget.axes.plot(wavelength, intensity, label='Raw Data', color='black', linestyle='solid')
                        plot_widget.axes.plot(wavelength, filtered_intensity, label='Savitzky-Golay Filtered', color='blue', alpha=0.5)
                        if self.checkBox.isChecked():
                            intensity_fft = fft(intensity)
                            frequencies = np.fft.fftfreq(len(intensity), d=(wavelength[1] - wavelength[0]))
                            cutoff = float(self.lineEdit_18.text())  # 주파수 컷오프
                            intensity_fft[np.abs(frequencies) > cutoff] = 0
                            intensity_fourier = np.real(ifft(intensity_fft))

                            plot_widget.axes.plot(wavelength, intensity_fourier, label='Fourier Transform', color='orange', alpha=0.5)
                        plot_widget.axes.axvline(x=(center_wavelength - range_value), ymin=0, ymax=1, linewidth=1, linestyle="--", color='k', alpha=0.9)
                        plot_widget.axes.axvline(x=(center_wavelength + range_value), ymin=0, ymax=1, linewidth=1, linestyle="--", color='k', alpha=0.9)

                        legend = plot_widget.axes.legend(fontsize=12)
                        legend.get_frame().set_facecolor('white')
                        legend.get_frame().set_edgecolor('lightgray')
                        legend.get_frame().set_alpha(1)

                        plot_widget.set_subplot(2)
                        # plot_widget.axes.set_xlabel('Wavelength (cm^-1)', fontsize=12)
                        plot_widget.axes.set_ylabel('Intensity (a.u.)', fontsize=12)
                        plot_widget.axes.set_title('Linear Interpolation on Savitzky-Golay Filtered Data')

                        plot_widget.axes.tick_params(axis="both", direction="in", pad=10, labelsize=10)
                        plot_widget.axes.plot(wavelength, filtered_intensity, label='Savitzky-Golay Filtered', color='blue')
                        plot_widget.axes.plot(wavelength, interpolated_intensity, label='Linear Interpolation on Filtered Data', color='green')
                        plot_widget.axes.axvline(x=(center_wavelength - range_value), ymin=0, ymax=1, linewidth=1,
                                                 linestyle="--", color='k', alpha=0.9)
                        plot_widget.axes.axvline(x=(center_wavelength + range_value), ymin=0, ymax=1, linewidth=1,
                                                 linestyle="--", color='k', alpha=0.9)
                        plot_widget.axes.fill_between(
                            wavelength,
                            filtered_intensity,
                            interpolated_intensity,
                            where=(filtered_intensity > interpolated_intensity),  # 조건 추가 가능
                            hatch='///',
                            color='b',
                            alpha=0.3,
                            label='Difference Area'
                        )

                        legend = plot_widget.axes.legend(fontsize=12)
                        legend.get_frame().set_facecolor('white')
                        legend.get_frame().set_edgecolor('lightgray')
                        legend.get_frame().set_alpha(1)

                        plot_widget.set_subplot(3)
                        plot_widget.axes.set_xlabel('Wavelength (cm^-1)', fontsize=12)
                        plot_widget.axes.set_ylabel('Intensity (a.u.)', fontsize=12)
                        plot_widget.axes.set_title('Difference: Savitzky-Golay - Linear Interpolation')

                        difference_intensity_filtered = filtered_intensity - interpolated_intensity

                        plot_widget.axes.tick_params(axis="both", direction="in", pad=10, labelsize=10)
                        plot_widget.axes.plot(wavelength, difference_intensity_filtered, color='red')
                        plot_widget.axes.axvline(x=(center_wavelength - range_value), ymin=0, ymax=1, linewidth=1, linestyle="--",
                                                 color='k', alpha=0.9)
                        plot_widget.axes.axvline(x=(center_wavelength + range_value), ymin=0, ymax=1, linewidth=1, linestyle="--",
                                                 color='k', alpha=0.9)

                        plot_widget.figure.tight_layout()
                        plot_widget.draw()
            except Exception as e:
                QtWidgets.QMessageBox.information(self, "Information",
                                                  f"This cell value is empty.\n{e}")
                pass

    def matched_result(self):
        # 1. 데이터프레임 생성
        columns = ['Filename', 'peak [CNN Model]', 'Prediction Value', 'peak', 'Matching']
        df = pd.DataFrame(columns=columns)

        # 2. CSV 파일 읽기
        file1 = self.lineEdit_16.text()  # self.lineEdit_16 파일 경로
        file2 = self.lineEdit_17.text()  # self.lineEdit_17 파일 경로

        df1 = pd.read_csv(file1)
        df2 = pd.read_csv(file2)

        # 3. Filename 기준으로 정렬
        df1 = df1.sort_values(by=df1.columns[0]).reset_index(drop=True)
        df2 = df2.sort_values(by=df2.columns[0]).reset_index(drop=True)

        # 4. 데이터프레임에 값 넣기
        df['Filename'] = df1.iloc[:, 0]  # df1의 0열
        df['peak [CNN Model]'] = df1.iloc[:, 1]  # df1의 1열
        df['Prediction Value'] = df1.iloc[:, 2]  # df1의 2열
        df['peak'] = df2.iloc[:, 1]  # df2의 1열 (peak)

        # 5. Matching 열 값 설정
        df['Matching'] = df.apply(lambda row: 'o' if row['peak [CNN Model]'] == row['peak'] else 'x', axis=1)

        # 6. 일치 개수, 불일치 개수, 총 개수, 일치율 계산
        matching_count = (df['Matching'] == 'o').sum()
        non_matching_count = (df['Matching'] == 'x').sum()
        total_count = len(df)
        matching_rate = matching_count / total_count * 100

        # 7. peak [CNN Model]이 1이면서 불일치 개수, 0이면서 불일치 개수 계산
        non_matching_peak_1_count = df[(df['Matching'] == 'x') & (df['peak [CNN Model]'] == 1)].shape[0]
        non_matching_peak_0_count = df[(df['Matching'] == 'x') & (df['peak [CNN Model]'] == 0)].shape[0]

        # 결과 출력
        self.label_55.setText(f"{matching_count}") # 일치 개수
        self.label_56.setText(f"{non_matching_count}") # 불일치 개수
        self.label_62.setText(f"{non_matching_peak_1_count}") # 불일치 개수
        self.label_63.setText(f"{non_matching_peak_0_count}") # 불일치 개수
        self.label_57.setText(f"{total_count}") # 총 개수
        self.label_59.setText(f"{matching_rate:.2f}%") # 일치율

        # 결과 데이터프레임 확인
        self.tableWidget_4.clearContents()
        self.tableWidget_4.setRowCount(0)
        row_count = len(df['Filename'].to_list())
        self.tableWidget_4.setRowCount(row_count)
        self.tableWidget_4.setColumnCount(5)
        for row in range(row_count):
            self.tableWidget_4.setItem(row, 0,
                                       QtWidgets.QTableWidgetItem(str(df['Filename'].to_list()[row])))
            self.tableWidget_4.setItem(row, 1,
                                       QtWidgets.QTableWidgetItem(str(df['peak [CNN Model]'].to_list()[row])))
            self.tableWidget_4.setItem(row, 2, QtWidgets.QTableWidgetItem(str(df['Prediction Value'].to_list()[row])))
            self.tableWidget_4.setItem(row, 3, QtWidgets.QTableWidgetItem(str(df['peak'].to_list()[row])))
            self.tableWidget_4.setItem(row, 4, QtWidgets.QTableWidgetItem(str(df['Matching'].to_list()[row])))
        QtWidgets.QMessageBox.information(self, "Matching Complete",
                                          "The Report has completed successfully.")


    def start_detection(self):
        self.detection_thred = RunModelThread(model_file=self.lineEdit_9.text(),
                                              folder_path=self.lineEdit_10.text(),
                                              batch_size=int(self.lineEdit_14.text()))
        self.detection_thred.update_text.connect(self.update_textEdit_2)
        self.detection_thred.start()

    def update_textEdit_2(self, text):
        matches = re.search(r"Batch (\d+)/(\d+) processed\.", text)
        if matches:
            # print([int(matches.group(1)), int(matches.group(2))])
            p_val = (int(matches.group(1)) / int(matches.group(2))) * 100
            self.progressBar_2.setValue(int(p_val))
        if "Total samples" in text:
            self.label_39.setText(f"{text}")
        elif "Best accuracy from training" in text:
            self.label_45.setText(f"{text}")
        elif "Done" in text:
            self.tableWidget_2.clearContents()
            self.tableWidget_2.setRowCount(0)
            row_count = len(self.detection_thred.report_file)
            self.tableWidget_2.setRowCount(row_count)
            self.tableWidget_2.setColumnCount(3)
            for row in range(row_count):
                self.tableWidget_2.setItem(row, 0,
                                           QtWidgets.QTableWidgetItem(str(self.detection_thred.report_file[row])))
                self.tableWidget_2.setItem(row, 1,
                                           QtWidgets.QTableWidgetItem(str(self.detection_thred.report_peak[row])))
                self.tableWidget_2.setItem(row, 2, QtWidgets.QTableWidgetItem(str(self.detection_thred.report_cs[row])))
            # self.loading_popup.close()
            QtWidgets.QMessageBox.information(self, "Detection Complete",
                                              "The Detection has completed successfully.")

    def start_training(self):
        # Start training in separate thread
        self.textEdit.setPlainText("")
        self.train_thread = TrainModelThread(folder_path=self.lineEdit_6.text(), info_file=self.lineEdit_8.text(),
                                             test_size=float(self.lineEdit_11.text()),
                                             random_state=float(self.lineEdit_13.text()),
                                             epochs=float(self.lineEdit_12.text()),
                                             batch_size=float(self.lineEdit_14.text()),
                                             validation_split=float(self.lineEdit_15.text()),
                                             save_model=self.lineEdit_7.text())
        self.train_thread.update_text.connect(self.update_textEdit)
        self.Y_train_loss, self.Y_train_acc, self.Y_test_loss, self.Y_test_acc = [], [], [], []
        self.train_thread.start()

    def add_message(self, new_message):
        current_text = self.textEdit.toPlainText()
        updated_text = f"{current_text}{new_message}\n"
        self.textEdit.setPlainText(updated_text)
        self.textEdit.moveCursor(QtGui.QTextCursor.MoveOperation.End)

    def update_textEdit(self, text):
        self.add_message(text)
        if "Epoch" in text:
            numbers = re.findall(r"\d+\.?\d*", text)
            result = [int(numbers[0]), int(numbers[1])] + [float(n) for n in numbers[2:]]
            p_val = (float(result[0]) / float(result[1])) * 100
            self.progressBar.setValue(int(p_val))
            self.Y_train_loss.append(result[2])
            self.Y_train_acc.append(result[3])
            self.Y_test_loss.append(result[4])
            self.Y_test_acc.append(result[5])
            self.on_training_plot()

    def on_training_plot(self):
        self.plot_widget_3.axes.cla()
        self.plot_widget_4.axes.cla()

        self.plot_widget_3.axes.plot(self.Y_train_loss, label='Train Loss')
        self.plot_widget_3.axes.plot(self.Y_test_loss, label='Test Loss')
        self.plot_widget_3.axes.set_xlabel('Epoch', fontsize=12)
        self.plot_widget_3.axes.set_ylabel('Loss', fontsize=12)
        self.plot_widget_3.axes.tick_params(axis="both", direction="in", pad=10, labelsize=10)
        self.plot_widget_3.axes.set_title('Training and Test Loss')
        # self.plot_widget_3.axes.legend()
        legend = self.plot_widget_3.axes.legend(fontsize=12)
        legend.get_frame().set_facecolor('white')
        legend.get_frame().set_edgecolor('lightgray')
        legend.get_frame().set_alpha(1)
        self.plot_widget_3.figure.tight_layout()
        self.plot_widget_3.draw()

        self.plot_widget_4.axes.plot(self.Y_train_acc, label='Train Accuracy')
        self.plot_widget_4.axes.plot(self.Y_test_acc, label='Test Accuracy')
        self.plot_widget_4.axes.set_xlabel('Epoch', fontsize=12)
        self.plot_widget_4.axes.set_ylabel('Accuracy', fontsize=12)
        self.plot_widget_4.axes.tick_params(axis="both", direction="in", pad=10, labelsize=10)
        self.plot_widget_4.axes.set_title('Training and Test Accuracy')
        # self.plot_widget_4.axes.legend()
        legend2 = self.plot_widget_4.axes.legend(fontsize=12)
        legend2.get_frame().set_facecolor('white')
        legend2.get_frame().set_edgecolor('lightgray')
        legend2.get_frame().set_alpha(1)
        self.plot_widget_4.figure.tight_layout()
        self.plot_widget_4.draw()

    def classifier_run(self):
        first_list = []
        second_list = []

        for index in range(self.treeView.model().rowCount()):
            item = self.treeView.model().item(index)
            first_list.append(item.text())

        for row in range(self.tableWidget.rowCount()):
            item = self.tableWidget.item(row, 1)
            if item is not None:
                second_list.append(item.text())

        data = {'filename': first_list, 'peak': second_list}
        df = pd.DataFrame(data)

        file_path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Save File", "", "CSV Files (*.csv);;All Files (*)"
        )

        if file_path:
            df.to_csv(file_path, index=False, header=True)

    def run_detect_run(self):
        first_list = []
        second_list = []
        third_list = []

        for row in range(self.tableWidget_2.rowCount()):
            item = self.tableWidget_2.item(row, 0)
            if item is not None:
                first_list.append(item.text().replace(f"{self.lineEdit_10.text()}\\", ""))

        for row in range(self.tableWidget_2.rowCount()):
            item = self.tableWidget_2.item(row, 1)
            if item is not None:
                second_list.append(item.text())

        for row in range(self.tableWidget_2.rowCount()):
            item = self.tableWidget_2.item(row, 2)
            if item is not None:
                third_list.append(item.text())

        data = {'Filename': first_list, 'peak': second_list, 'Prediction Value': third_list}
        df = pd.DataFrame(data)

        file_path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Save File", "", "CSV Files (*.csv);;All Files (*)"
        )

        if file_path:
            df.to_csv(file_path, index=False, header=True)

    def save_file(self, value):
        # 현재 선택된 항목의 인덱스를 가져옴
        current_index = self.treeView.currentIndex()

        item = self.model.itemFromIndex(current_index)
        if item is not None:
            item_name = item.text()
            folder_path = self.lineEdit_3.text()

            # 파일 이름에서 확장자 변경
            new_file_name = item_name.replace('.csv', '.dat')
            new_file_path = os.path.join(folder_path, new_file_name)

            # 1 또는 0 저장
            with open(new_file_path, 'w') as f:
                f.write(str(value))

            row_index = current_index.row()
            self.tableWidget.setItem(row_index, 1, QtWidgets.QTableWidgetItem(f"{value}"))

            print(f"파일 저장됨: {new_file_path} - 값: {value}")
        else:
            print("선택된 파일이 없습니다.")

    def move_to_previous_table_item(self, table_widget):
        # 현재 활성화된 셀의 행, 열 인덱스를 가져옴
        current_row = table_widget.currentRow()
        current_column = table_widget.currentColumn()

        # 위로 이동, 첫 행에서 이동할 경우 마지막 행으로 이동 가능
        new_row = max(0, current_row - 1)
        table_widget.setCurrentCell(new_row, current_column)
        self.on_table_click()

    def move_to_next_table_item(self, table_widget):
        # 현재 활성화된 셀의 행, 열 인덱스를 가져옴
        current_row = table_widget.currentRow()
        current_column = table_widget.currentColumn()

        # 아래로 이동, 마지막 행에서 이동할 경우 첫 행으로 이동 가능
        new_row = min(table_widget.rowCount() - 1, current_row + 1)
        table_widget.setCurrentCell(new_row, current_column)
        self.on_table_click()

    def on_tab_changed(self, index):
        self.tab_index = index
        states = {
            3: {
                "shortcut_prev": False, "shortcut_next": False, "shortcut_o": False, "shortcut_x": False,
                "up_shortcut": True, "down_shortcut": True,
                "up_shortcut_2": False, "down_shortcut_2": False,
                "up_shortcut_3": False, "down_shortcut_3": False
            },
            1: {
                "shortcut_prev": True, "shortcut_next": True, "shortcut_o": True, "shortcut_x": True,
                "up_shortcut": False, "down_shortcut": False,
                "up_shortcut_2": False, "down_shortcut_2": False,
                "up_shortcut_3": False, "down_shortcut_3": False
            },
            4: {
                "shortcut_prev": False, "shortcut_next": False, "shortcut_o": False, "shortcut_x": False,
                "up_shortcut": False, "down_shortcut": False,
                "up_shortcut_2": True, "down_shortcut_2": True,
                "up_shortcut_3": False, "down_shortcut_3": False
            },
            5: {
                "shortcut_prev": False, "shortcut_next": False, "shortcut_o": False, "shortcut_x": False,
                "up_shortcut": False, "down_shortcut": False,
                "up_shortcut_2": False, "down_shortcut_2": False,
                "up_shortcut_3": True, "down_shortcut_3": True
            }
        }

        default_state = {
            "shortcut_prev": False, "shortcut_next": False, "shortcut_o": False, "shortcut_x": False,
            "up_shortcut": False, "down_shortcut": False,
            "up_shortcut_2": False, "down_shortcut_2": False,
            "up_shortcut_3": False, "down_shortcut_3": False
        }

        # 상태 설정
        self.set_shortcuts_state(states.get(index, default_state))

    def set_shortcuts_state(self, state):
        self.shortcut_prev.setEnabled(state["shortcut_prev"])
        self.shortcut_next.setEnabled(state["shortcut_next"])
        self.shortcut_o.setEnabled(state["shortcut_o"])
        self.shortcut_x.setEnabled(state["shortcut_x"])
        self.up_shortcut.setEnabled(state["up_shortcut"])
        self.down_shortcut.setEnabled(state["down_shortcut"])
        self.up_shortcut_2.setEnabled(state["up_shortcut_2"])
        self.down_shortcut_2.setEnabled(state["down_shortcut_2"])
        self.up_shortcut_3.setEnabled(state["up_shortcut_3"])
        self.down_shortcut_3.setEnabled(state["down_shortcut_3"])

    def move_to_previous_item(self):
        # 현재 선택된 항목의 인덱스를 가져옴
        current_index = self.treeView.currentIndex()

        # 이전 인덱스가 있는지 확인 후 이동
        previous_index = self.model.index(current_index.row() - 1, 0, current_index.parent())
        if previous_index.isValid():
            self.treeView.setCurrentIndex(previous_index)
            self.display_spectrum_from_selected_file_2()
            # print("이전 항목:", self.model.filePath(previous_index))
            self.tableWidget.setCurrentCell(current_index.row() - 1, 1)

    def move_to_next_item(self):
        # 현재 선택된 항목의 인덱스를 가져옴
        current_index = self.treeView.currentIndex()

        # 다음 인덱스가 있는지 확인 후 이동
        next_index = self.model.index(current_index.row() + 1, 0, current_index.parent())
        if next_index.isValid():
            self.treeView.setCurrentIndex(next_index)
            self.display_spectrum_from_selected_file_2()
            # print("다음 항목:", self.model.filePath(next_index))
            self.tableWidget.setCurrentCell(current_index.row() + 1, 1)

    def open_file_dialog(self, line_edit):
        current_text = line_edit.text()
        file_path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "파일 열기", "", "All Files (*);;Text Files (*.txt)")

        if file_path:
            line_edit.setText(file_path)
        else:
            line_edit.setText(current_text)

    def open_folder_dialog(self, line_edit):
        current_text = line_edit.text()
        folder_path = QtWidgets.QFileDialog.getExistingDirectory(self, "폴더 열기", "")

        if folder_path:
            line_edit.setText(folder_path)
        else:
            line_edit.setText(current_text)

    def open_folder_dialog_2(self):
        current_text = self.lineEdit_3.text()
        folder_path = QtWidgets.QFileDialog.getExistingDirectory(self, "폴더 열기", "")

        if folder_path:
            self.lineEdit_3.setText(folder_path)
        else:
            self.lineEdit_3.setText(current_text)

        folder_path = self.lineEdit_3.text()
        file_paths = glob.glob(os.path.join(folder_path, "*.csv"))
        file_paths = os_sorted(file_paths)
        file_paths = [item.replace(folder_path + "\\", "") for item in file_paths]

        # 파일명을 모델에 추가하여 트리뷰에 표시
        for file_name in file_paths:
            item = QtGui.QStandardItem(file_name)
            self.model.appendRow(item)

        self.tableWidget.clearContents()
        self.tableWidget.setRowCount(0)
        self.tableWidget.setRowCount(len(file_paths))  # 행 수 설정
        # file_paths = [item.replace(folder_path + "\\", "") for item in file_paths]
        file_paths_2 = [item.replace(file_paths[0].split('(')[0], "") for item in file_paths]
        for row, file_name in enumerate(file_paths_2):
            self.tableWidget.setItem(row, 0, QtWidgets.QTableWidgetItem(file_name))

        # 첫 번째 아이템의 인덱스를 가져와 선택 및 출력
        if self.model.rowCount() > 0:
            first_index = self.model.index(0, 0)
            self.treeView.setCurrentIndex(first_index)  # 첫 번째 항목 활성화
            self.on_tree_item_clicked(first_index)

        self.display_spectrum_from_selected_file_2()

        list_classifier = []

        for file_name in file_paths:
            file_name = file_name.replace('.csv', '.dat')
            dat_file_path = folder_path + '/' + file_name
            # print(dat_file_path)
            if os.path.isfile(dat_file_path):
                # print(f"'{file_name}' 파일을 읽습니다.")
                with open(dat_file_path, 'r') as file:
                    content = file.read()  # 파일 내용 읽기
                    # print(f"파일 내용:\n{content}\n")  # 파일 내용 출력
                    list_classifier.append(content)
            else:
                # print(f"'{file_name}' 파일이 존재하지 않습니다.")
                list_classifier.append('')

        for n, i in enumerate(list_classifier):
            self.tableWidget.setItem(n, 1, QtWidgets.QTableWidgetItem(f"{i}"))

    def on_tree_item_clicked(self, index):
        # 클릭된 항목의 데이터 가져오기
        item = self.model.itemFromIndex(index)
        file_name = item.text()
        # print(f"Selected item: {file_name}")  # 선택된 파일명을 출력
        return file_name

    def process_file(self):
        file_path_and_name = self.lineEdit_2.text()
        output_dir = self.lineEdit.text()
        file_name = os.path.splitext(os.path.basename(file_path_and_name))[0]

        # 파일이 존재하는지 확인
        if not os.path.isfile(file_path_and_name):
            print("지정한 파일이 존재하지 않습니다.")
            return

        # 파일 읽기 및 데이터 분리
        df = pd.read_csv(file_path_and_name, delimiter='\t')
        x_coords = df['Unnamed: 0']
        y_coords = df['Unnamed: 1']
        wavelengths = df.columns[2:].astype(float)
        intensities = df.iloc[:, 2:]

        # 파일 리스트 초기화
        self.file_list = []

        # 각 좌표에 대해 파일 저장
        for i in range(len(df)):
            x = x_coords[i]
            y = y_coords[i]
            # print(f'Curr. : {x}, {y}')

            intensity_at_point = intensities.iloc[i].to_numpy()
            output_df = pd.DataFrame({
                'Wavelength': wavelengths,
                'Intensity': intensity_at_point
            })

            # 파일 저장
            output_file_name = f"{file_name}_({x}_{y}).csv"
            output_path = os.path.join(output_dir, output_file_name)
            output_df.to_csv(output_path, index=False)

            # 파일 리스트에 파일 경로 추가
            self.file_list.append(output_path)

        # peaks_info.csv로 저장
        # peaks_info_file_name = 'peaks_info.csv'
        # output_path = os.path.join(output_dir, peaks_info_file_name)
        # pd.DataFrame([file for file in self.file_list], columns=['File_Name']).to_csv(peaks_info_file_name, index=False)

        # listView에 생성된 파일 리스트 표시
        self.list_model.clear()
        for file_path in self.file_list:
            item = QtGui.QStandardItem(os.path.basename(file_path))
            item.setData(file_path, QtCore.Qt.ItemDataRole.UserRole)  # 파일 경로 저장
            self.list_model.appendRow(item)

    def display_spectrum_from_selected_file(self, index):
        self.plot_widget.axes.cla()
        # 선택된 파일의 경로 가져오기
        selected_file_path = self.list_model.itemFromIndex(index).data(QtCore.Qt.ItemDataRole.UserRole)

        # 선택된 파일 읽기
        if os.path.isfile(selected_file_path):
            spectrum_df = pd.read_csv(selected_file_path)
            wavelengths = spectrum_df['Wavelength'].to_numpy()
            intensities = spectrum_df['Intensity'].to_numpy()

            # wavelength와 intensity 출력
            # print(f"Selected File: {selected_file_path}")
            # print("Wavelengths:", wavelengths)
            # print("Intensities:", intensities)
            self.plot_widget.axes.plot(wavelengths, intensities)
            self.plot_widget.axes.set_xlabel('Wavelength (cm^-1)', fontsize=12)
            self.plot_widget.axes.set_ylabel('Intensity (a.u.)', fontsize=12)
            self.plot_widget.axes.tick_params(axis="both", direction="in", pad=10, labelsize=10)

            # tight_layout 적용
            self.plot_widget.figure.tight_layout()

            self.plot_widget.draw()

    def display_spectrum_from_selected_file_2(self):
        self.plot_widget_2.axes.cla()

        selected_index = self.treeView.currentIndex()
        item = self.model.itemFromIndex(selected_index)
        if item is not None:
            selected_value = item.text()
            self.tableWidget.setCurrentCell(selected_index.row(), 1)
            # print(selected_value)
        else:
            print("No item selected.")

        selected_file_path = os.path.join(self.lineEdit_3.text() + '/', selected_value)

        # 선택된 파일 읽기
        if os.path.isfile(selected_file_path):
            spectrum_df = pd.read_csv(selected_file_path)
            wavelengths = spectrum_df['Wavelength'].to_numpy()
            intensities = spectrum_df['Intensity'].to_numpy()

            # wavelength와 intensity 출력
            # print(f"Selected File: {selected_file_path}")
            # print("Wavelengths:", wavelengths)
            # print("Intensities:", intensities)
            self.plot_widget_2.axes.plot(wavelengths, intensities)
            self.plot_widget_2.axes.set_xlabel('Wavelength (cm^-1)', fontsize=12)
            self.plot_widget_2.axes.set_ylabel('Intensity (a.u.)', fontsize=12)
            self.plot_widget_2.axes.tick_params(axis="both", direction="in", pad=10, labelsize=10)
            peak_start = float(self.lineEdit_4.text()) - float(self.lineEdit_5.text())
            peak_end = float(self.lineEdit_4.text()) + float(self.lineEdit_5.text())
            self.plot_widget_2.axes.axvline(x=peak_start, ymin=0, ymax=1, linewidth=1, linestyle="--", color='k')
            self.plot_widget_2.axes.axvline(x=peak_end, ymin=0, ymax=1, linewidth=1, linestyle="--", color='k')
            self.plot_widget_2.axes.set_title(f'{selected_file_path}')

            # tight_layout 적용
            self.plot_widget_2.figure.tight_layout()

            self.plot_widget_2.draw()

    def save_to_csv_index4(self):
        # CSV 저장 경로 선택 대화창
        file_path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Save CSV", "", "CSV Files (*.csv);;All Files (*)")
        if not file_path:
            return  # 파일 저장 경로를 선택하지 않으면 반환

        # TableWidget 데이터를 pandas DataFrame으로 변환
        try:
            data = []
            # 헤더 가져오기
            headers = [self.tableWidget_4.horizontalHeaderItem(col).text() for col in
                       range(self.tableWidget_4.columnCount())]

            # 데이터 가져오기
            for row in range(self.tableWidget_4.rowCount()):
                row_data = [
                    self.tableWidget_4.item(row, col).text() if self.tableWidget_4.item(row, col) else ""
                    for col in range(self.tableWidget_4.columnCount())
                ]
                data.append(row_data)

            # pandas DataFrame 생성
            df = pd.DataFrame(data, columns=headers)

            # DataFrame을 CSV로 저장
            df.to_csv(file_path, index=False, encoding='utf-8')
            QtWidgets.QMessageBox.information(self, "Save Report",
                                              f"File Path : {file_path}")
        except Exception as e:
            QtWidgets.QMessageBox.warning(self, "Warning", f"Err. :{e}")

    def process_file_interpolation(self, file_path, window_length, polyorder, center_wavelength, range_value):
        """
        Process a CSV file to filter intensity and perform linear interpolation.

        Parameters:
            file_path (str): Path to the CSV file.
            window_length (int): Window length for Savitzky-Golay filter.
            polyorder (int): Polynomial order for Savitzky-Golay filter.
            center_wavelength (float): Center wavelength for interpolation.
            range_value (float): Range for interpolation (+/- around center).

        Returns:
            wavelength (numpy.ndarray): Wavelength values.
            raw_intensity (numpy.ndarray): Raw intensity values.
            filtered_intensity (numpy.ndarray): Intensity after applying the filter.
            interpolated_intensity (numpy.ndarray): Intensity after interpolation.
        """
        # Read data from the file
        data = pd.read_csv(file_path)
        wavelength = data.iloc[:, 0].values  # 0th column: Wavelength
        raw_intensity = data.iloc[:, 1].values  # 1st column: Intensity

        # Apply Savitzky-Golay filter
        filtered_intensity = savgol_filter(raw_intensity, window_length=window_length, polyorder=polyorder)

        # Define the interpolation range
        start = center_wavelength - range_value
        end = center_wavelength + range_value
        mask = (wavelength >= start) & (wavelength <= end)

        linear_interp_filtered = np.interp(wavelength[mask], [start, end],
                                           [filtered_intensity[wavelength < start][-1],
                                            filtered_intensity[wavelength > end][0]])

        # Initialize interpolated intensity with filtered intensity
        interpolated_intensity_full = filtered_intensity.copy()
        interpolated_intensity_full[mask] = linear_interp_filtered

        return wavelength, raw_intensity, filtered_intensity, interpolated_intensity_full

    def process_data(self):
        try:
            # Read input data from line edits
            report_file = self.lineEdit_20.text()
            folder_path = self.lineEdit_21.text()
            window_length = int(self.lineEdit_22.text())
            polyorder = int(self.lineEdit_23.text())
            center_wavelength = float(self.lineEdit_24.text())
            range_value = float(self.lineEdit_25.text())

            # Read report CSV
            report_df = pd.read_csv(report_file)
            file_names = report_df.iloc[:, 0].tolist()  # File names from 0th column
            peaks = report_df.iloc[:, 1].tolist()  # Peaks from 1st column

            # Prepare table widget
            self.tableWidget_3.setRowCount(0)
            self.tableWidget_3.setColumnCount(3)
            self.tableWidget_3.setHorizontalHeaderLabels(["File Name", "Peak / No Peak", "Interg. val."])

            # Process each file
            for i, file_name in enumerate(file_names):
                file_path = os.path.join(folder_path, file_name)
                if not os.path.isfile(file_path):
                    print(f"File not found: {file_path}")
                    continue

                wavelength, intensity, filtered_intensity, interpolated_intensity = self.process_file_interpolation(
                    file_path=file_path,
                    window_length=window_length,
                    polyorder=polyorder,
                    center_wavelength=center_wavelength,
                    range_value=range_value
                )

                start = center_wavelength - range_value
                end = center_wavelength + range_value
                # interpolated_wavelength = np.linspace(start, end, 100)
                # interpolated_intensity = np.interp(interpolated_wavelength, wavelength[mask], filtered_intensity[mask])

                # Calculate integral value
                integral_value = np.trapezoid(interpolated_intensity, wavelength)

                # Update table widget with results
                row_position = self.tableWidget_3.rowCount()
                self.tableWidget_3.insertRow(row_position)
                self.tableWidget_3.setItem(row_position, 0, QtWidgets.QTableWidgetItem(file_name))
                self.tableWidget_3.setItem(row_position, 1, QtWidgets.QTableWidgetItem(str(peaks[i])))
                self.tableWidget_3.setItem(row_position, 2, QtWidgets.QTableWidgetItem(f"{integral_value:.2f}"))

            # print("Data processing completed successfully.")
        except Exception as e:
            print(f"An error occurred: {e}")

    def closeEvent(self, event):
        # 종료 시 Config 저장
        self.save_config()
        event.accept()


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    app.setStyle("Fusion")
    form = MainWindow()
    form.show()
    app.exec()
