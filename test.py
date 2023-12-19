#pip3 install tensorflow
#model_path  keras model เอามาจาก teachablemachine https://teachablemachine.withgoogle.com/train/image
import cv2
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
import numpy as np
import math

cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=2)
model_path = "/Users/pimuk/Work/6401811 HandSignDetection/Model/keras_model.h5"
labels_path = "/Users/pimuk/Work/6401811 HandSignDetection/Model/labels.txt"
classifier = Classifier(model_path, labels_path) # สร้างอ็อบเจกต์ Classifier โดยโหลดโมเดลจากไฟล์ keras_model.h5 และ labels.txt.

offset = 20
imgSize = 300

data_folder = "Data"
labels = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L" , "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X" ,"Y", "Z", "Nice", "Bad"]

while True:
    success, img = cap.read()
    imgOutput = img.copy()
    hands, img = detector.findHands(img)

    for hand in hands:
        x, y, w, h = hand['bbox']

        if x + w // 2 < img.shape[1] // 2:  
            continue

        if w > 0 and h > 0:
            imgCrop = img[y - offset : y + h + offset, x - offset : x + w + offset]

            if imgCrop.shape[0] > 0 and imgCrop.shape[1] > 0:
                imgCrop = cv2.resize(imgCrop, (imgSize, imgSize))

                imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255

                if h > w:
                    k = imgSize / h
                    wCal = math.ceil(k * w)
                    imgResize = cv2.resize(imgCrop, (wCal, imgSize))
                    wGap = math.ceil((imgSize - wCal) / 2)
                    imgWhite[:, wGap:wCal + wGap] = imgResize

                    prediction, index = classifier.getPrediction(imgWhite) 

                    print(prediction, index)


                else:
                    k = imgSize / w
                    hCal = math.ceil(k * h)
                    imgResize = cv2.resize(imgCrop, (imgSize, hCal))
                    hGap = math.ceil((imgSize - hCal) / 2)
                    imgWhite[hGap:hCal + hGap, :] = imgResize

                    prediction, index = classifier.getPrediction(imgWhite)

                cv2.putText(imgOutput, labels[index], (x, y - 20), cv2.FONT_HERSHEY_COMPLEX, 2, (255, 0, 255), 2) #print ออกมาตามชื่อ labels

                cv2.imshow("ImageCrop", imgCrop)
                cv2.imshow("ImageWhite", imgWhite)

    if img.shape[0] > 0 and img.shape[1] > 0:
        # img.shape[0]: หมายถึง ความสูง (จำนวนแถว) ของภาพ.
        # img.shape[1]: หมายถึง ความกว้าง (จำนวนคอลัมน์) ของภาพ.
        # img.shape[0] > 0: ตรวจสอบว่าความสูงมีค่ามากกว่า 0.
        # img.shape[1] > 0: ตรวจสอบว่าความกว้างมีค่ามากกว่า 0.
        # and: ตรวจสอบว่าทั้งความสูงและความกว้างมีค่ามากกว่า 0.
        # เป็นเงื่อนไขที่ใช้ตรวจสอบว่าขนาดของภาพที่ถูกดึงมาจากกล้องมีความกว้างและความสูงที่มีค่ามากกว่าศูนย์หรือไม่
        cv2.imshow("Image", imgOutput)

    key = cv2.waitKey(1)
    if key == ord('q'):
        break

cv2.destroyAllWindows()
