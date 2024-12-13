import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import pandas as pd
from PIL import Image, ImageOps
import matplotlib.pyplot as plt
import os
import numpy as np
from tqdm import tqdm
from src import models_bam
from collections import OrderedDict
from utils import utils, custom_dataset
import datetime
import copy
time_now = f"{datetime.datetime.now()}"
time_now = time_now[:16]


# 랜덤 시드 고정    
utils.set_seed(18)

data_dir = "../dataset/train"
batch_size = 8

# 데이터 셋 제작 (train:val=8:2) (batch_size=4)
dataset = custom_dataset.CustomDataset(data_dir)
print(dataset.class_name_list)

train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size

train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# 정답 클래스 개수와 정답 클래스 이름
num_of_classes = len(os.listdir(data_dir))
class_names = dataset.class_name_list

# 모델 정의
model = models_bam.KNOWN_MODELS['BiT-M-R152x2'](head_size=4, bam=True)

# 가중치 로드
weights = torch.load("./weights/pretrain/best_hoya_BiT+BAM+Burn.pth")
weight_state_dict = weights.module.state_dict()
model.load_state_dict(weight_state_dict)

# 모델 head 지정
wf = 2
model.head = nn.Sequential(OrderedDict([
        ('gn', nn.GroupNorm(32, 2048*wf)),
        ('relu', nn.ReLU(inplace=True)),
        ('avg', nn.AdaptiveAvgPool2d(output_size=1)),
        ('conv', nn.Conv2d(2048*wf, num_of_classes, kernel_size=1, bias=True))
        ]))

#################### CHECK ####################

# 모델 head만 학습
for param in model.parameters():
    param.requires_grad = True
for param in model.head.parameters():
    param.requires_grad = True 

# device 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# 손실함수와 최적화 설정
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.head.parameters(), lr=0.001)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=600)

#################### CHECK ####################

# 손실값, 정확도값 리스트
train_losses = []
train_accuracies = []
val_losses = []
val_accuracies = []

best_val_acc = 0.0
best_model_weights = None

best_prediction_of_valid = []
num_epochs = 1
for epoch in range(num_epochs):

    # 학습 시작
    model.train()
    train_loss = 0.0
    train_corrects = 0

    for inputs, labels in tqdm(train_dataloader, desc=f'Training Epoch {epoch+1}'):
        inputs = inputs.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()

        # 학습 추론
        try:
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            scheduler.step()

        except Exception as e:
            print(f"Error during training: {e}")
            continue  # 오류가 발생하면 다음 배치로 넘어감

        train_loss += loss.item() * inputs.size(0)
        _, preds = torch.max(outputs, 1)
        train_corrects += torch.sum(preds == labels.data)
            
    epoch_loss = train_loss / train_size
    epoch_acc = train_corrects.double() / train_size

    train_losses.append(epoch_loss)
    train_accuracies.append(epoch_acc.item())

    # 평가 시작
    model.eval()
    val_loss = 0.0
    val_corrects = 0

    to_save = []
    with torch.no_grad():
        for inputs, labels in tqdm(val_dataloader, desc=f'Validation Epoch {epoch+1}'):
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            val_corrects += torch.sum(preds == labels.data)
            to_save.append((list(preds), list(labels.data)))
            

    val_loss = val_loss / val_size
    val_acc = val_corrects.double() / val_size

    val_losses.append(val_loss)
    val_accuracies.append(val_acc.item())

    # 최고 validation 정확도 업데이트
    if val_acc >= best_val_acc:
        best_val_acc = val_acc
        best_prediction_of_valid = copy.deepcopy(to_save)
        best_model_weights = model.state_dict().copy()

    print(f'Epoch {epoch+1}/{num_epochs}, '
          f'Train Loss: {epoch_loss:.4f}, Train Acc: {epoch_acc:.4f}, '
          f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}')


result_dir = f"./results/{time_now}"
os.mkdir(result_dir)

# 최고 validation 정확도의 모델 가중치 저장
if best_model_weights is not None:
    os.mkdir(f"{result_dir}/weight")
    torch.save(best_model_weights, f"{result_dir}/weight/model.pth")
    print(f'Saved model with best validation accuracy: {best_val_acc:.4f}')
    pd.DataFrame(best_prediction_of_valid).to_csv(f"{result_dir}/best_prediction.csv", index=False)

print('Finished Training')

# 학습 손실 곡선
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(train_losses, label='Train Loss', color='blue')
plt.plot(val_losses, label='Validation Loss', color='orange')
plt.title('Loss Curve')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid()

# 학습 정확도 곡선
plt.subplot(1, 2, 2)
plt.plot(train_accuracies, label='Train Accuracy', color='green')
plt.plot(val_accuracies, label='Validation Accuracy', color='red')
plt.title('Accuracy Curve')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.grid()

plt.tight_layout()

# 그래프를 지정한 경로에 저장
plt.savefig(f"{result_dir}/plot.png")

pd.DataFrame(train_accuracies).to_csv(f"{result_dir}/train_accuracy.csv", index=False)
pd.DataFrame(train_losses).to_csv(f"{result_dir}/train_loss.csv", index=False)
pd.DataFrame(val_accuracies).to_csv(f"{result_dir}/validation_accuracy.csv", index=False)
pd.DataFrame(val_losses).to_csv(f"{result_dir}/validation_loss.csv", index=False)

print(f"result in {time_now}")