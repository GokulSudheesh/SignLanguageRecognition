import cv2
import mediapipe as mp
import numpy as np
import os
import string

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

letters = list(string.ascii_uppercase)
letters.remove("J")
letters.remove("Z")
w, h = None, None

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


def web_cam():
    global w, h
    # Calibrate from web cam
    cap = cv2.VideoCapture(0)
    w, h = 640, 480
    cap.set(3, w)
    cap.set(4, h)
    count = 0
    labels = []
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            continue
        image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = hands.process(image)
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        if results.multi_hand_landmarks: # Detected a hand
            hand_landmarks = results.multi_hand_landmarks[0]
            landmarks = results.multi_hand_landmarks[0].landmark
            landmarks_normal = list(map(normalize, landmarks))
            # Get opened and closed fingers in a frame
            open_close = open_close_fingers(landmarks_normal)
            print(open_close)

            mp_drawing.draw_landmarks(
                    image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            if ((cv2.waitKey(1) & 0xFF == ord('q')) and count < len(letters)):
                labels.append(open_close)
                count += 1
        if count == len(letters):
            print(labels)
            np.save("calibrated_hand_cam.npy", np.array(labels))
            break
        # Annotation
        cv2.putText(image,"Press Q to save, esc to quit.", (250, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                     (0, 0, 0), 2)
        annotation_image = cv2.resize(cv2.imread("./SLR/annotations/"+letters[count]+".jpg"), (200, 200))

        image[0:200, 0:200] = annotation_image
        cv2.imshow('MediaPipe Hands', image)
        if (cv2.waitKey(1) & 0xFF == 27):
            break
    cv2.destroyAllWindows()
    cap.release()

if __name__ == '__main__':
    web_cam()