import cv2
import numpy as np
from ultralytics import YOLO

class GeometricDistanceEstimator:
    def __init__(self, model_path, k_matrix, real_size_m):
        self.model = YOLO(model_path)
        # Матрица параметров камеры
        self.K = k_matrix
        # Реальный физический размер аппарата в метрах
        self.real_size_m = real_size_m
        self.fx = k_matrix[0, 0]
        self.fy = k_matrix[1, 1]

    def _calculate_distance(self, bbox_w, bbox_h):
        dist_w = (self.real_size_m * self.fx) / bbox_w
        dist_h = (self.real_size_m * self.fy) / bbox_h
        return (dist_w + dist_h) / 2

    def analyze_frame(self, frame, draw=True):
        results = self.model(frame, verbose=False)
        detections = results[0].boxes
        results_data = []

        for box in detections:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            conf = box.conf[0].item()
            
            bbox_w = x2 - x1
            bbox_h = y2 - y1

            distance = self._calculate_distance(bbox_w, bbox_h)
            
            results_data.append({
                "bbox": (x1, y1, x2, y2),
                "distance": distance,
                "confidence": conf
            })

            if draw:
                label = f"Drone: {distance:.2f}m"
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, label, (x1, y1 - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        return frame, results_data