# pip3 install opencv-python
# pip3 install cvzone      
# pip3 install mediapipe  
import cv2 #คือ OpenCV เป็นไลบรารีที่มีความสามารถในการประมวลผลภาพและวิดีโอ, การตรวจจับวัตถุ, การทำ computer vision, และการทำงานที่เกี่ยวข้องกับการประมวลผลภาพดิจิทัล
from cvzone.HandTrackingModule import HandDetector #นำเข้าโมดูลที่เกี่ยวข้องกับการตรวจจับมือ (hand tracking) จากไลบรารี cvzone
import numpy as np # เป็นไลบรารีที่มีความสามารถในการทำงานกับ arrays และการดำเนินการทางตัวเลข
import math # ใช้ฟังก์ชันทางคณิตศาสตร์และค่าคงที่ทางคณิตศาสตร์ที่มีในโมดูลนี้ได้
import time # ใช้ฟังก์ชันที่เกี่ยวข้องกับการจับเวลาและการหน่วงเวลาใน Python ได้

cap = cv2.VideoCapture(0) #เปิดการใช้งานกล้องเว็บแคม
detector = HandDetector(maxHands=2) #สร้างอ็อบเจกต์ HandDetector ที่สามารถตรวจจับมือได้สูงสุด 2 มือ

offset = 20 #สร้างตัวแปร offset มีค่าเท่ากับ 20
imgSize = 300 #สร้างตัวแปร imgSize มีค่าเท่ากับ 300

folder = "Data/C" #สร้างตัวแปร folder กำหนดไปในที่ตั้ง Data/C
counter = 0 #สร้างตัวแปร counter มีค่าเท่ากับ 0

while True: #loop
    success, img = cap.read()
# ใช้อ่านภาพเป็น frames จากวิดีโอโดยใช้ OpenCV library
# cap คือ object ของคลาส cv2.VideoCapture ใช้เพื่อเปิดและจัดการกับวิดีโอหรือกล้อง
# cap.read() ใช้เพื่ออ่าน frame ถัดไปจากวิดีโอหรือกล้องที่ถูกเปิด ผลลัพธ์จะมีสองค่าที่ถูกนำเข้าไปในตัวแปร success และ img.
# success มีค่า True ถ้าการอ่าน frame สำเร็จ และ False ถ้าไม่สำเร็จ
# img: เป็นภาพ (frame) ที่ถูกอ่านจากวิดีโอหรือกล้อง
# คำสั่งนี้ถูกใช้ในการอ่าน frame จากวิดีโอหรือกล้องอย่างต่อเนื่องในลูป


    hands, img = detector.findHands(img) #เป็นส่วนหนึ่งของการใช้งานตัวตรวจจับมือ (hand detection) โดยใช้คลาสหรือฟังก์ชัน detector


# เป็นการประมวลผลภาพสำหรับแต่ละมือที่ตรวจจับได้จากการใช้ detector.findHands(img) 
# และทำการ crop และปรับขนาดของภาพในลูป for hand in hands:
    for hand in hands:
        x, y, w, h = hand['bbox'] # ดึงข้อมูล bounding box (bbox) ของมือ (hand) ที่ได้จากการตรวจจับมือ

        if w > 0 and h > 0: 
            imgCrop = img[y - offset: y + h + offset, x - offset: x + w + offset] # ทำการ crop ภาพตาม bounding box และเพิ่ม offset 20 เพื่อหลีกเลี่ยงการตัดขอบของมือ

            #ส่วนของการปรับขนาดภาพ
            if imgCrop.shape[0] > 0 and imgCrop.shape[1] > 0:
                imgCrop = cv2.resize(imgCrop, (imgSize, imgSize)) # ปรับขนาดของภาพให้มีขนาดเท่ากับ imgSize x imgSize (300*300)

                imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255 # สร้าง imgWhite ที่มีขนาด imgSize x imgSize (300*300) และทำให้พื้นหลังเป็นสีขาว

                # ตรวจสอบเงื่อนไขว่าความสูง h ของ bbox มีค่า มากกว่า w หรือไม่ ถ้าถูกก็นำมาปรับขนาดภาพ
                if h > w:
                    k = imgSize / h # คำนวณอัตราส่วนของความสูงเพื่อปรับขนาดภาพในทิศทางความกว้าง
                    
                    #การใช้ math.ceil ทำให้ค่าที่ได้เป็นจำนวนขนาดภาพที่ถูกปรับขนาดเป็นจำนวนเต็ม
                    wCal = math.ceil(k * w) # คำนวณความกว้างที่จะได้หลังจากที่ปรับขนาด
                    imgResize = cv2.resize(imgCrop, (wCal, imgSize)) # ปรับขนาดภาพใหม่ตามความกว้างที่ได้
                    wGap = math.ceil((imgSize - wCal) / 2) #คำนวณความกว้างของพื้นที่ที่เป็นสีขาวด้านซ้ายและด้านขวาของภาพที่ถูกปรับขนาด
                    imgWhite[:, wGap: wCal + wGap] = imgResize # นำภาพที่ถูกปรับขนาดมาวางตรงกลางของพื้นที่สีขาว

                else: # เมื่อความกว้าง (w) มีค่ามากกว่าความสูง h
                    k = imgSize / w # คำนวณอัตราส่วนของความกว้างเพื่อปรับขนาดภาพในทิศทางความสูง
                    hCal = math.ceil(k * h) #คำนวณความสูงที่จะได้หลังจากที่ปรับขนาด
                    imgResize = cv2.resize(imgCrop, (imgSize, hCal)) # ปรับขนาดภาพใหม่ตามความสูงที่ได้
                    hGap = math.ceil((imgSize - hCal) / 2) # คำนวณความสูงของพื้นที่ที่เป็นสีขาวด้านบนและด้านล่างของภาพที่ถูกปรับขนาด
                    imgWhite[hGap: hCal + hGap, :] = imgResize # นำภาพที่ถูกปรับขนาดมาวางตรงกลางของพื้นที่สีขาว

                cv2.imshow("ImageCrop", imgCrop) # แสดงภาพที่ถูก crop แล้วในหน้าต่างที่ชื่อ "ImageCrop"
                cv2.imshow("ImageWhite", imgWhite) # แสดงภาพที่ถูกปรับขนาดและวางตำแหน่งในพื้นที่สีขาวในหน้าต่างที่ชื่อ "ImageWhite"
                
# ตรวจสอบว่าภาพ (img) มีความสูงและความกว้างมากกว่าศูนย์หรือไม่ ก่อนที่จะใช้ cv2.imshow 
# เพื่อแสดงภาพทั้งหมดในหน้าต่างที่ชื่อ "Image"
    if img.shape[0] > 0 and img.shape[1] > 0:
        cv2.imshow("Image", img)

    key = cv2.waitKey(1) #ปิดโปรแกรม
    if key == ord('q'):
        break
    elif key == ord("s"): #บันทึกรูป
        counter += 1
        cv2.imwrite(f'{folder}/Image_{time.time()}.jpg', imgWhite)
        print(counter)

cv2.destroyAllWindows()
