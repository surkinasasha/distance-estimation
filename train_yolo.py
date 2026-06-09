from ultralytics import YOLO

model = YOLO("yolo11s.pt") 

results = model.train(
    data="seraphim_drone_dataset/data.yaml", 
    epochs=100,                 
    imgsz=640,
    batch=16,
    device=0,
    project="drone_localization",
    name="yolov11s_seraphim",
    mosaic=1.0,              
    save=True,
    plots=True
)

metrics = model.val()
print(f"Точность mAP@0.5: {metrics.box.map50:.3f}")