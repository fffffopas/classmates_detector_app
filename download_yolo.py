import shutil
from ultralytics import YOLO

model = YOLO("yolo26n.pt")
exported_path = model.export(format="tflite", imgsz=416, nms=False)

to_path = "android/app/src/main/assets/yolo26n.tflite"
shutil.move(exported_path, to_path)