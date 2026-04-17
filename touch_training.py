import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, MaxAbsScaler, RobustScaler
from torchsummary import summary


# 커스텀 데이터셋 클래스 정의 (헤더 처리 추가)
class GraphDataset(Dataset):
    def __init__(self, folder_path, info_file=None, transform=None):
        self.folder_path = folder_path
        self.transform = transform

        if info_file:
            # info.csv가 있는 경우 (학습용 데이터셋)
            self.info_df = pd.read_csv(info_file)
            self.file_names = self.info_df.iloc[:, 0].values
            self.labels = self.info_df.iloc[:, 1].values
        else:
            # info.csv가 없는 경우 (예측용 데이터셋)
            self.file_names = [f for f in os.listdir(folder_path) if f.endswith('.csv')]
            self.labels = np.zeros(len(self.file_names))  # 더미 라벨

        # 첫 번째 파일을 로드하여 특성 수 확인
        first_file = pd.read_csv(os.path.join(folder_path, self.file_names[0]))
        self.n_features = len(first_file.columns)
        # print(f"Number of features per sample: {self.n_features}")

        # 데이터 정규화를 위한 scaler 초기화
        # self.scaler = StandardScaler()
        self.scaler = StandardScaler()
        self._fit_scaler()

    def _fit_scaler(self):
        # 모든 데이터를 합쳐서 scaler 학습
        all_data = []
        for file_name in self.file_names:
            file_path = os.path.join(self.folder_path, file_name)
            data = pd.read_csv(file_path)
            all_data.append(data.values)
        all_data = np.vstack(all_data)
        self.scaler.fit(all_data)
        # print("Data scaler fitted successfully")

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, idx):
        # CSV 파일 로드 (헤더 포함)
        file_path = os.path.join(self.folder_path, self.file_names[idx])
        data = pd.read_csv(file_path)

        # 데이터를 numpy 배열로 변환
        features = data.values.astype(np.float32)

        # 데이터 정규화
        features = self.scaler.transform(features)

        if self.transform:
            features = self.transform(features)

        # 텐서로 변환 (1차원으로 평탄화)
        features = torch.FloatTensor(features.flatten())
        # print(self.labels[idx])
        label = torch.tensor(self.labels[idx], dtype=torch.long)

        return features, label

class GraphDataset2(Dataset):
    def __init__(self, folder_path, info_file=None, transform=None):
        self.folder_path = folder_path
        self.transform = transform

        if info_file:
            # info.csv가 있는 경우 (학습용 데이터셋)
            self.info_df = pd.read_csv(info_file)
            self.file_names = self.info_df.iloc[:, 0].values
            self.labels = self.info_df.iloc[:, 1].values
        else:
            # info.csv가 없는 경우 (예측용 데이터셋)
            self.file_names = [f for f in os.listdir(folder_path) if f.endswith('.csv')]
            self.labels = np.zeros(len(self.file_names))  # 더미 라벨

        # 첫 번째 파일을 로드하여 특성 수 확인
        first_file = pd.read_csv(os.path.join(folder_path, self.file_names[0]))
        self.n_features = len(first_file.columns)

        # 데이터 정규화를 위한 scaler 초기화
        # self.scaler = StandardScaler()
        self.scaler = StandardScaler()
        self._fit_scaler()

    def _fit_scaler(self):
        all_data = []
        for file_name in self.file_names:
            file_path = os.path.join(self.folder_path, file_name)
            data = pd.read_csv(file_path)
            all_data.append(data.values)
        all_data = np.vstack(all_data)
        self.scaler.fit(all_data)

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, idx):
        file_path = os.path.join(self.folder_path, self.file_names[idx])
        data = pd.read_csv(file_path)

        features = data.values.astype(np.float32)
        features = self.scaler.transform(features)

        if self.transform:
            features = self.transform(features)

        features = torch.FloatTensor(features.flatten())
        label = torch.tensor(self.labels[idx], dtype=torch.long)

        return features, label, self.file_names[idx]  # 파일명도 반환


# 신경망 모델 정의
class BinaryClassifier(nn.Module):
    def __init__(self, input_size):
        super(BinaryClassifier, self).__init__()
        self.layer1 = nn.Linear(input_size, 128)
        self.layer2 = nn.Linear(128, 64)
        self.layer3 = nn.Linear(64, 32)
        self.layer4 = nn.Linear(32, 1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.01)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.dropout(self.relu(self.layer1(x)))
        x = self.dropout(self.relu(self.layer2(x)))
        x = self.dropout(self.relu(self.layer3(x)))
        x = self.sigmoid(self.layer4(x))
        return x


# 학습 함수
def train_model(model, train_loader, criterion, optimizer, device):
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
def evaluate_model(model, test_loader, criterion, device):
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


def main():
    # 하이퍼파라미터 설정
    BATCH_SIZE = 64
    EPOCHS = 120
    LEARNING_RATE = 0.0001

    # 장치 설정
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 데이터 로드
    folder_path = r'D:/prog_test/20241118 serum 4 to10/big ng samp'  # CSV 파일이 있는 폴더 경로
    info_file = r'D:/prog_test/20241118 serum 4 to10/output file 4 to 10/output big ng samp.csv'  # 라벨 정보가 있는 파일 경로

    # 데이터셋 생성
    dataset = GraphDataset(folder_path, info_file)

    # 학습/테스트 데이터 분할
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

    # 데이터로더 생성
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True)

    # 입력 크기 계산
    sample_data, _ = dataset[0]
    input_size = sample_data.shape[0]
    print(f"Input size: {input_size}")

    # 모델 초기화
    model = BinaryClassifier(input_size).to(device)
    criterion = nn.BCELoss()  # Binary Cross Entropy Loss
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    summary(model, input_size=(input_size,))

    # 모델 학습
    best_acc = 0.0
    print("Starting training...")
    for epoch in range(EPOCHS):
        train_loss, train_acc = train_model(model, train_loader, criterion, optimizer, device)
        test_loss, test_acc = evaluate_model(model, test_loader, criterion, device)

        print(f'Epoch [{epoch + 1}/{EPOCHS}]')
        print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
        print(f'Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%')

        # 최고 성능 모델 저장
        if test_acc > best_acc:
            best_acc = test_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_acc': best_acc,
            }, 'best_model.pth')

    print(f'Best Test Accuracy: {best_acc:.2f}%')
    # summary(model, input_size=input_size)


# 저장된 모델을 사용한 예측
def predict(model, data_loader, device):
    model.eval()
    predictions = []

    with torch.no_grad():
        for inputs, _ in data_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            predicted = (outputs.squeeze() > 0.5).float()
            predictions.extend(predicted.cpu().numpy())

    return predictions


if __name__ == '__main__':
    main()

# folder_path = 'D:/prog_test/20241104_1600_slicing'  # CSV 파일이 있는 폴더 경로
# info_file = 'D:/prog_test/20241104_output.csv'  # 라벨 정보가 있는 파일 경로