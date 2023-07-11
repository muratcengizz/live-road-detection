from ultralytics import YOLO

model = YOLO("best.pt")

metrics = model.val(plots=True)
print(metrics)