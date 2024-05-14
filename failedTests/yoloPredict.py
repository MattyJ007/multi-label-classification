from ultralytics import YOLO
import os
import csv
import json

model = YOLO("yolov8s.pt") # load the model

predictions = {}

for file in range(30000, 40000):
  filename = str(file) + ".jpg"
  results = model("./data/images/" + filename, stream = True)

  for result in results:
      prediction = []
      for box in result.boxes:
          class_id = int(box.data[0][-1])
          prediction.append(class_id)
      predictions[filename] = prediction

  with open('predictions.json',
  'w') as f:
      json.dump(predictions, f)