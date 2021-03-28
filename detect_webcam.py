from tensorflow.keras.models import load_model
import cv2
import numpy as np
import string
import urllib.request

model = load_model('AlexNet-040.model')
#model = load_model('model2-020.model')
categories = list(range(25))
categories.remove(9)
classes = len(categories)
letters = list(string.ascii_uppercase)
letters.remove("J")
letters.remove("Z")
IMG_SIZE = 28

url = "http://192.168.2.2:8080/shot.jpg"
#cap = cv2.VideoCapture(url)
cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    # Reading from phone cam
    '''
    imgResp = urllib.request.urlopen(url)
    imgNp = np.array(bytearray(imgResp.read()), dtype=np.uint8)
    frame = cv2.imdecode(imgNp, -1)
    '''
    frame = cv2.resize(frame, (640, 480), interpolation=cv2.INTER_LINEAR)
    #frame = cv2.flip(frame, 1)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    x,y,w,h = 100,100,400,300
    cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

    hand_img = gray[y:y + w, x:x + w]
    resized = cv2.resize(hand_img, (IMG_SIZE, IMG_SIZE))
    normalized = resized / 255.0
    reshaped = np.reshape(normalized, (1, IMG_SIZE, IMG_SIZE, 1))
    result = model.predict(reshaped)

    #print(result)
    accuracy = "{:.2f}".format(np.amax(result)*100)
    #accuracy = "%0.2f" % 3

    label = np.argmax(result, axis=1)[0]
    print(accuracy)
    cv2.rectangle(frame, (0, 0), (150, 50), (255, 255, 255), -1)
    #cv2.rectangle(frame, (x, y - 40), (x + w + 50, y), (255, 0, 0), -1)
    cv2.putText(frame, letters[label] + " " + accuracy + "%", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                (0, 0, 0), 2)

    cv2.imshow('Emotion Detector', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cv2.destroyAllWindows()
cap.release()
