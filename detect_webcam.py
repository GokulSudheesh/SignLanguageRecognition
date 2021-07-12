from tensorflow.keras.models import load_model
import cv2
import numpy as np
import string
import urllib.request

# Load the trained SLR model
model = load_model('AlexNet-040.model')
categories = list(range(25))
categories.remove(9)
classes = len(categories)
letters = list(string.ascii_uppercase)
letters.remove("J")
letters.remove("Z")
IMG_SIZE = 28
w, h = 640, 480
#url = "http://192.168.2.2:8080/shot.jpg"
#cap = cv2.VideoCapture(url)
cap = cv2.VideoCapture(0)
cap.set(3, w)
cap.set(4, h)

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        print("Ignoring empty camera frame.")
        continue
    # Reading from phone cam
    '''
    imgResp = urllib.request.urlopen(url)
    imgNp = np.array(bytearray(imgResp.read()), dtype=np.uint8)
    frame = cv2.imdecode(imgNp, -1)
    '''
    frame = cv2.flip(frame, 1)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Draw Region of interest
    x, y, w, h = 100, 100, 400, 300
    cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

    hand_img = gray[y:y + w, x:x + w]
    resized = cv2.resize(hand_img, (IMG_SIZE, IMG_SIZE))
    normalized = resized / 255.0
    reshaped = np.reshape(normalized, (1, IMG_SIZE, IMG_SIZE, 1))
    result = model.predict(reshaped)

    #print(result)
    accuracy = "{:.2f}".format(np.amax(result)*100)

    label = np.argmax(result, axis=1)[0]
    #print(accuracy)
    cv2.rectangle(frame, (0, 0), (150, 50), (255, 255, 255), -1)
    cv2.putText(frame, letters[label] + " " + accuracy + "%", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                (0, 0, 0), 2)

    cv2.imshow('SLR', frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break
cv2.destroyAllWindows()
cap.release()