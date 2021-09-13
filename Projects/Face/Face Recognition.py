# FACE RECOGNITION + ATTENDANCE PROJECT
# 1. Finding Faces by Hog Method at the backened which are histogram oreiented
# 2. Posing and Projecting Faces
# 3. Encoding Faces : 128 Measurements
# 4. Finding the name of that person by encoding

import face_recognition
import cv2
import numpy as np

# Uploading Images
# 1. At first Loading an image
# 2. Then testing an image with based on 1st Img

imgSteve = face_recognition.load_image_file("photos/sj2.jpg")
imgSteve = cv2.cvtColor(imgSteve, cv2.COLOR_BGR2RGB)
imgTest = face_recognition.load_image_file("photos/sj_Test.jpg")
imgTest = cv2.cvtColor(imgTest, cv2.COLOR_BGR2RGB)

# Finding faces and their encoding
faceLoc = face_recognition.face_locations(imgSteve)[0]
encodeSteve = face_recognition.face_encodings(imgSteve)[0]
cv2.rectangle(imgSteve, (faceLoc[3], faceLoc[0], faceLoc[2], faceLoc[2]), (255, 0, 255), 2)
# print(faceLoc)        #Print values of top bottom lef and right coordinates

faceLocTest = face_recognition.face_locations(imgTest)[0]
encodeTest = face_recognition.face_encodings(imgTest)[0]
cv2.rectangle(imgTest, (faceLocTest[3], faceLocTest[0], faceLocTest[2], faceLocTest[2]), (255, 0, 255), 2)

# Comparing our load Img and Test Img
results = face_recognition.compare_faces([encodeSteve],encodeTest)
faceDistance = face_recognition.face_distance([encodeSteve], encodeTest)
print(results, faceDistance)
cv2.putText(imgTest, f'{results} {round(faceDistance[0],2)}', (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)



cv2.imshow('Steve Job', imgSteve)
cv2.imshow('Steve Job Test Image', imgTest)
cv2.waitKey(0)
