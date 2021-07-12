from tensorflow.keras.models import load_model
import cv2
import mediapipe as mp
import numpy as np
import string

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
calibrated_hand = np.load("calibrated_hand.npy")

model = load_model('AlexNet-040.model')
categories = list(range(25))
categories.remove(9)
classes = len(categories)
letters = list(string.ascii_uppercase)
letters.remove("J")
letters.remove("Z")
IMG_SIZE = 28

w, h = 640, 480

hands = mp_hands.Hands(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5)

def normalize(landmark):
    # Get the normalized x,y coordinates. (Map function)
    return int(landmark.x * w), int(landmark.y * h)

def get_orientation(landmarks):
    # d(x) - d(y)
    if (abs(landmarks[0][0] - landmarks[12][0]) >= abs(landmarks[0][1] - landmarks[12][1])):
        # Difference in x is more than difference in y
        if (landmarks[12][0] > landmarks[0][0]):
            #print("Horizontal Right")
            return 0, 1 # Orientation, Direction
        else:
            #print("Horizontal Left")
            return 0, -1
    else:
        # Difference in y is more than difference in x
        if(landmarks[12][1] > landmarks[0][1]):
            #print("Vertical Down")
            return 1, 1
        else:
            #print("Vertical Up")
            return 1, -1
    return None

def open_close_fingers(landmarks):
    # Get vertical / horizontal orientation and direction
    orient, direct = get_orientation(landmarks)
    open_close_list = []
    for i in range(4, 24, 4):
        joint_1 = (landmarks[i][orient] - landmarks[i-3][orient]) * direct
        joint_2 = (landmarks[i][orient] - landmarks[i-1][orient]) * direct

        if (joint_1 < 0):
            # Fully closed
            open_close_list.append(1 * direct)
        elif (joint_2 < 0):
            # Half open - half closed
            open_close_list.append(2 * direct)
        else:
            # Fully open
            open_close_list.append(3 * direct)
    return open_close_list

def get_similarity_fast(predictions, hand_o_c):
    # Compares the calibrated hand and the hand from frame
    # Returns the letter (index) that's equal to the calibrated hand
    # (fast but less accurate)
    hand_o_c = np.array(hand_o_c)
    for p_index in predictions:
        comparison = calibrated_hand[p_index] == hand_o_c
        print(comparison.all())
        # If both the numpy arrays are equal then:
        if comparison.all():
            return p_index
    return 0

def get_similarity_slow(predictions, hand_o_c):
    # Compares the calibrated hand and the hand from frame
    # Returns the letter (index) that's similar to the calibrated hand
    # (slow but more accurate)
    hand_o_c = np.array(hand_o_c)
    max_sim, index = 0, 0
    for p_index in predictions:
        sim = 0
        for i, finger in enumerate(calibrated_hand[p_index]):
            if(finger == hand_o_c[i]):
                sim += 1
        if (sim > max_sim):
            max_sim = sim
            index = p_index
    return index

def cam():
    cap = cv2.VideoCapture(0)
    cap.set(3, w)
    cap.set(4, h)

    while cap.isOpened():
        success, image = cap.read()
        if not success:
          print("Ignoring empty camera frame.")
          continue

        # Flip the image horizontally for a later selfie-view display, and convert
        # the BGR image to RGB.
        image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
        # To improve performance, optionally mark the image as not writeable to
        # pass by reference.
        image.flags.writeable = False
        results = hands.process(image)

        # Draw the hand annotations on the image.
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        if results.multi_hand_landmarks: # Detected a hand
            hand_landmarks = results.multi_hand_landmarks[0]
            landmarks = results.multi_hand_landmarks[0].landmark
            landmarks_normal = list(map(normalize, landmarks))

            open_close_list = open_close_fingers(landmarks_normal)

            # Drawing boxes
            x_center, y_center = landmarks_normal[9]
            x, y = x_center - 150, y_center - 200
            width, height = 300, 350
            cv2.rectangle(image, (x, y), (x + int(width), y + int(height)), (255, 0, 0), 2)
            try:
                # Predict letter
                gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)[y:y + height, x:x + width]
                resized = cv2.resize(gray, (IMG_SIZE, IMG_SIZE))
                normalized = resized / 255.0
                reshaped = np.reshape(normalized, (1, IMG_SIZE, IMG_SIZE, 1))
                result = model.predict(reshaped)

                # Get top 5 predictions
                top_5 = np.argsort(result[0])[::-1][:5]
                label = get_similarity_slow(top_5, open_close_list)
                accuracy = "{:.2f}".format(np.amax(result) * 100)
                # Draw result
                cv2.rectangle(image, (0, 0), (150, 50), (255, 255, 255), -1)
                # cv2.rectangle(frame, (x, y - 40), (x + w + 50, y), (255, 0, 0), -1)
                cv2.putText(image, letters[label] + " " + accuracy + "%", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                            (0, 0, 0), 2)
            except Exception as e:
                print("Broken Image")
            # mp_drawing.draw_landmarks(
            #    image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
        cv2.imshow('MediaPipe Hands', image)
        if cv2.waitKey(5) & 0xFF == 27:
          break
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    cam()