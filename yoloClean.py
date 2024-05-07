import json
import csv
from datetime import datetime

yolo_classes = {
    0: 'person',
    1: 'bicycle',
    2: 'car',
    3: 'motorcycle',
    4: 'airplane',
    5: 'bus',
    6: 'train',
    7: 'truck',
    8: 'boat',
    9: 'traffic light',
    10: 'fire hydrant',
    11: 'stop sign',
    12: 'parking meter',
    13: 'bench',
    14: 'bird',
    15: 'cat',
    16: 'dog',
    17: 'horse',
    18: 'sheep',
    19: 'cow',
    20: 'elephant',
    21: 'bear',
    22: 'zebra',
    23: 'giraffe',
    24: 'backpack',
    25: 'umbrella',
    26: 'handbag',
    27: 'tie',
    28: 'suitelif prediction ==',
    29: 'frisbee',
    30: 'skis',
    31: 'snowboard',
    32: 'sports ball',
    33: 'kite',
    34: 'baseball bat',
    35: 'baseball glove',
    36: 'skateboard',
    37: 'surfboard',
    38: 'tennis racket',
    39: 'bottle',
    40: 'wine glass',
    41: 'cup',
    42: 'fork',
    43: 'knife',
    44: 'spoon',
    45: 'bowl',
    46: 'banana',
    47: 'apple',
    48: 'sandwich',
    49: 'orange',
    50: 'broccoli',
    51: 'carrot',
    52: 'hot dog',
    53: 'pizza',
    54: 'donut',
    55: 'cake',
    56: 'chair',
    57: 'couch',
    58: 'potted plant',
    59: 'bed',
    60: 'dining table',
    61: 'toilet',
    62: 'tv',
    63: 'laptop',
    64: 'mouse',
    65: 'remote',
    66: 'keyboard',
    67: 'cell phone',
    68: 'microwave',
    69: 'oven',
    70: 'toaster',
    71: 'sink',
    72: 'refrigerator',
    73: 'book',
    74: 'clock',
    75: 'vase',
    76: 'scissors',
    77: 'teddy bear',
    78: 'hair drier',
    79: 'toothbrush'
}

def classMapping(prediction):
    if prediction == 0:
      return 1
    elif prediction == 1:
      return 2
    elif prediction == 2:
      return 3
    elif prediction == 3:
      return 4
    elif prediction == 4:
      return 5
    elif prediction == 5:
      return 6
    elif prediction == 6:
      return 7
    elif prediction == 7:
      return 8
    elif prediction == 8:
      return 9
    elif prediction == 9:
      return 10
    elif prediction == 10:
      return 11
    elif prediction == 11:
      return 13
    elif prediction == 12:
      return 14
    elif prediction == 13:
      return 15
    elif prediction == 14:
      return 16
    elif prediction == 15:
      return 17
    elif prediction == 16:
      return 18
    elif prediction == 17:
      return 19
    else:
      return 0
 
# Opening JSON file
f = open('predictions.json')

data = json.load(f)

now = datetime.now()

filepath = 'predictions/' + now.strftime("%d-%m-%Y-%H-%M") + '-510369965-490424191-490299418.csv'

with open(filepath, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['ImageID', 'Labels'])
    for key in data.keys():
        row = []
        for predictedClass in data[key]:
            mapped = classMapping(predictedClass)
            if mapped > 0:
              row.append(mapped)
        row = list(set(row))
        writer.writerow([key, " ".join(str(label) for label in row)])
 
f.close()
