import cv2
import pandas as pd
import dlib
import numpy as np
from Face_and_Landmark_Detection.ExtractionUtil import ExtractionUtil
import random

extractionUtil = ExtractionUtil()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

def extract_landmarks(pixels2D):
    rectangle = dlib.rectangle(0, 0, 48, 48)
    landmarks = predictor(np_image, rectangle)

    if len(landmarks.parts()) == 0:
        return None

    coordinates = np.zeros((len(landmarks.parts()), 2), dtype=int)
    for i, landmark in enumerate(landmarks.parts()):
        coordinates[i] = (landmark.x, landmark.y)

    return coordinates.tolist()

def checkIfDuplicateCoordinates(landmarks):
    length = len(landmarks)
    found = False

    for i in range(length):
        for j in range(i + 1, length):
            if (landmarks[i][0] == landmarks[j][0] and landmarks[i][1] == landmarks[j][1]):
                print(f"landmark number {i} and {j} are identical !")
                found = True
                break
        if (found):
            break
    
    return found

data = pd.read_csv('fer2013.csv')
total_rows = len(data)
processed_rows = 0

i = 10

label = data['emotion'][i]
pixels = data['pixels'][i]

pixel_list = [int(x) for x in pixels.split()]
np_image = np.array(pixel_list, dtype=np.uint8).reshape(48, 48)

landmarks = extract_landmarks(pixels)
print(np.shape(np_image))
scaled_np_image = cv2.resize(np_image,(600, 600))
scaled_np_image = cv2.cvtColor(scaled_np_image, cv2.COLOR_GRAY2BGR)

print("landmarks : ")
print(landmarks)

if (checkIfDuplicateCoordinates(landmarks)):
    print("Duplicate co ordinates found; not all unique")
else:
    print("All co ordinates are unique")

for i in range(len(landmarks)):
    cv2.circle(scaled_np_image, (int((landmarks[i][0] * 600) / 48), int((landmarks[i][1] * 600) / 48)), random.randint(3, 10), (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)), 1)

cv2.rectangle(scaled_np_image, (0, 0), (10, 10), (255, 0, 0), 2)
print(f"number of landmarks : {len(landmarks)}")

cv2.imshow("image", scaled_np_image)
cv2.waitKey(0)