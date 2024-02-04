import pandas as pd
import dlib
import numpy as np
from Emotion_Classification.DataSetWriter import DatasetWriter
from Face_and_Landmark_Detection.FeatureExtraction import extract_features

datasetWriter = DatasetWriter()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

def extract_landmarks(pixels):
    pixel_list = [int(x) for x in pixels.split()]
    np_image = np.array(pixel_list, dtype=np.uint8).reshape(48, 48)
    rectangle = dlib.rectangle(0, 0, 48, 48)
    landmarks = predictor(np_image, rectangle)

    if len(landmarks.parts()) == 0:
        return None

    coordinates = np.zeros((len(landmarks.parts()), 2), dtype=int)
    for i, landmark in enumerate(landmarks.parts()):
        coordinates[i] = (landmark.x, landmark.y)

    return coordinates.tolist()

data = pd.read_csv('fer2013.csv')
total_rows = len(data)
processed_rows = 0

for row in data.itertuples(index=False):
    label = row.emotion
    pixels = row.pixels

    landmarks = extract_landmarks(pixels)

    if landmarks is not None:
        # Example usage:
        # landmark_positions = [(18, 25, 29), (20, 23, 29), (29, 48, 54),
        #     (48, 51, 57), (51, 54, 57)]

        features = extract_features(landmarks)
        #print("Extracted Features:", features)
        datasetWriter.addNewData(features,label)

    processed_rows += 1
    remaining_percentage = ((total_rows - processed_rows) / total_rows) * 100
    print(f'Remaining data: {remaining_percentage:.2f}%')
