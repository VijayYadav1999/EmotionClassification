import cv2
import numpy as np
from Face_and_Landmark_Detection.ExtractionUtil import ExtractionUtil
import Emotion_Classification.Classification_Prediction as c
from sklearn.preprocessing import MinMaxScaler
#from Analyser.Suggestion import get_suggestion,update_emotion_record

# Initialize video capture
vid = cv2.VideoCapture(0)

# Initialize ExtractionUtil
extractionUtil = ExtractionUtil()

# Initialize scaler for parameter normalization
scaler = MinMaxScaler()

# Initialize emotion detection count
emotion_count = 0

while True:
    # Read frame from video capture
    ret, frame = vid.read()

    # Convert frame to grayscale
    frameBW = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Get face rectangle
    ret, rect = extractionUtil.getFaceRect(frameBW)

    if ret == 0:
        # Draw face rectangle on frame
        cv2.rectangle(frame, extractionUtil.getFlattenedRectangleFromDLibRectangle(rect), (255, 0, 0), 10)

        # Get landmarks
        landmarks = extractionUtil.get_landmarks(frameBW, rect)
        extractionUtil.draw_shape_points(frame, landmarks)

        landmarks = extractionUtil.normalizelandmarks(landmarks)

        flattened_list = np.reshape(landmarks, (136, 1))

        # Get emotion prediction
        emotion = c.predict_emotions(flattened_list)

        #update_emotion_record(emotion)

        # Display emotion
        print(emotion)

        # # Increment emotion detection count
        # emotion_count += 1

        # # Check if 20 emotions have been detected
        # if emotion_count == 20:
        #     break

    # Resize frame
    frame = cv2.resize(frame, (640, 480), interpolation=cv2.INTER_LINEAR)

    # Display frame
    cv2.imshow("window", frame)

    # Check for quit command
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release video capture and close windows
vid.release()
cv2.destroyAllWindows()

# # Get suggestion based on the most frequent emotion
# suggestion = get_suggestion()
# print(suggestion)

