import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# Параметры обучения
CSV_PATH = "drone_crops_with_bbox.csv"
MODEL_SAVE_PATH = "weights/resnet_regressor.pth"
BATCH_SIZE = 16
EPOCHS = 40
LEARNING_RATE = 1e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

class DroneDatasetWithBBox(Dataset):
    def __init__(self, df, transform=None):
        self.df = df.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image = Image.open(row['crop_path']).convert('RGB')
        if self.transform:
            image = self.transform(image)
        
        # Передаем координаты рамки как дополнительные признаки
        bbox = torch.tensor([row['bbox_x'], row['bbox_y'],
                             row['bbox_w'], row['bbox_h']], dtype=torch.float32)
        distance = torch.tensor(row['distance'], dtype=torch.float32)
        return image, bbox, distance

# Трансформации с аугментацией для компенсации перспективных искажений
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.RandomAffine(degrees=0, scale=(0.9, 1.1)), 
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

class DroneRegressorWithBBox(nn.Module):
    def __init__(self):
        super().__init__()
        backbone = models.resnet18(weights='IMAGENET1K_V1')
        # Извлекаем признаки из предобученной ResNet-18
        self.features = nn.Sequential(*list(backbone.children())[:-1])
        self.feature_dim = backbone.fc.in_features # 512

        # Объединяем 512 признаков изображения и 4 признака координат рамки
        self.fc = nn.Sequential(
            nn.Linear(self.feature_dim + 4, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, img, bbox):
        x = self.features(img)
        x = torch.flatten(x, 1)
        x = torch.cat([x, bbox], dim=1) # Конкатенация векторов
        return self.fc(x)

# Подготовка данных
df = pd.read_csv(CSV_PATH)
bins = np.linspace(df['distance'].min(), df['distance'].max(), 10)
df['dist_bin'] = np.digitize(df['distance'], bins)

train_df, val_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['dist_bin'])

train_loader = DataLoader(DroneDatasetWithBBox(train_df, train_transform), batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(DroneDatasetWithBBox(val_df, val_transform), batch_size=BATCH_SIZE)

# Инициализация модели и оптимизатора
model = DroneRegressorWithBBox().to(DEVICE)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5)

# Цикл обучения
best_val_loss = float('inf')
for epoch in range(EPOCHS):
    model.train()
    train_loss = 0.0
    for img, bbox, dist in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
        img, bbox, dist = img.to(DEVICE), bbox.to(DEVICE), dist.to(DEVICE).unsqueeze(1)
        
        optimizer.zero_grad()
        pred = model(img, bbox)
        loss = criterion(pred, dist)
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * img.size(0)

    # Валидация
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for img, bbox, dist in val_loader:
            img, bbox, dist = img.to(DEVICE), bbox.to(DEVICE), dist.to(DEVICE).unsqueeze(1)
            pred = model(img, bbox)
            val_loss += criterion(pred, dist).item() * img.size(0)
    
    val_loss /= len(val_loader.dataset)
    scheduler.step(val_loss)

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), MODEL_SAVE_PATH)
        print(f" -> Сохранена лучшая модель (MSE: {val_loss:.3f})")

# Итоговая оценка MAE
model.load_state_dict(torch.load(MODEL_SAVE_PATH))
model.eval()
all_preds, all_labels = [], []
with torch.no_grad():
    for img, bbox, dist in val_loader:
        img, bbox = img.to(DEVICE), bbox.to(DEVICE)
        pred = model(img, bbox).cpu().numpy()
        all_preds.extend(pred.flatten())
        all_labels.extend(dist.numpy())

print(f"\nИтоговая средняя ошибка (MAE): {np.mean(np.abs(np.array(all_preds) - np.array(all_labels))):.2f} м")