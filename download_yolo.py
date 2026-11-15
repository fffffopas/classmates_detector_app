import shutil
from ultralytics import YOLO

model = YOLO("yolo26n.pt")
exported_path = model.export(format="litert", imgsz=(640, 480),nms=False)

to_path = "android/app/src/main/assets/yolo26n.tflite"
shutil.move(exported_path, to_path)

from ai_edge_litert.interpreter import Interpreter

interp = Interpreter(model_path="android/app/src/main/assets/yolo26n.tflite")
d = interp.get_input_details()[0]
print(d["shape"], d["dtype"])
print(interp.get_output_details()[0]["shape"])