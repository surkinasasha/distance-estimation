import cv2
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms, models
from ultralytics import YOLO

class DroneRegressorWithBBox(nn.Module):
    def __init__(self):
        super().__init__()
        backbone = models.resnet18(weights=None)
        self.features = nn.Sequential(*list(backbone.children())[:-1])
        self.feature_dim = backbone.fc.in_features
        self.fc = nn.Sequential(
            nn.Linear(self.feature_dim + 4, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, img, bbox):
        x = self.features(img)
        x = torch.flatten(x, 1)
        x = torch.cat([x, bbox], dim=1)
        return self.fc(x)

class NeuralDistanceEstimator:
    def __init__(self, yolo_path, regressor_path):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.yolo = YOLO(yolo_path)
        self.regressor = DroneRegressorWithBBox()
        self.regressor.load_state_dict(torch.load(regressor_path, map_location=self.device))
        self.regressor.to(self.device).eval()

        self.preprocess = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def _get_distance(self, crop_np, bbox_coords, img_shape):
        h, w = img_shape[:2]
        x1, y1, x2, y2 = bbox_coords
        crop_pil = Image.fromarray(cv2.cvtColor(crop_np, cv2.COLOR_BGR2RGB))
        input_tensor = self.preprocess(crop_pil).unsqueeze(0).to(self.device)

        bbox_w, bbox_h = (x2 - x1) / w, (y2 - y1) / h
        bbox_x, bbox_y = (x1 + x2) / (2 * w), (y1 + y2) / (2 * h)
        bbox_tensor = torch.tensor([[bbox_x, bbox_y, bbox_w, bbox_h]], dtype=torch.float32).to(self.device)

        with torch.no_grad():
            dist = self.regressor(input_tensor, bbox_tensor).item()
        return dist

    def analyze_frame(self, frame, draw=True):
        results = self.yolo(frame, verbose=False, conf=0.4)
        detections = results[0].boxes
        results_data = []

        if len(detections) > 0:
            best_box = max(detections, key=lambda b: b.conf[0])
            x1, y1, x2, y2 = map(int, best_box.xyxy[0].tolist())
            conf = best_box.conf[0].item()

            crop = frame[y1:y2, x1:x2]
            if crop.size > 0:
                distance = self._get_distance(crop, (x1, y1, x2, y2), frame.shape)
                
                results_data.append({
                    "bbox": (x1, y1, x2, y2),
                    "distance": distance,
                    "confidence": conf
                })

                if draw:
                    label = f"Distance: {distance:.2f}m"
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    cv2.putText(frame, label, (x1, y1 - 10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        return frame, results_data