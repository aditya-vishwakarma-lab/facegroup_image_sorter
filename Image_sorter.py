import cv2
import face_recognition
import os
import numpy as np
import shutil

path = 'solo_images'
images = []
classNames = []
mylist = os.listdir(path)

for cl in mylist:
    curImg = cv2.imread(f'{path}/{cl}')
    images.append(curImg)
    name = os.path.splitext(cl)[0]
    name_folder_path = f"./sorted_img/{name}/"  
    shutil.rmtree(f"./sorted_img/{name}", ignore_errors=True)
    os.mkdir(name_folder_path)

    classNames.append(name)


def find_encodings(pictures):
    encodeList = []
    for pic in pictures:
        pic = cv2.cvtColor(pic, cv2.COLOR_BGR2RGB)
        encoded_face = face_recognition.face_encodings(pic)[0]
        encodeList.append(encoded_face)
    return encodeList


encoded_face_train = find_encodings(images)
for cl in os.listdir("group_photos"):

    img = cv2.imread(f'group_photos/{cl}')
    imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25)
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)
    faces_in_frame = face_recognition.face_locations(imgS)
    encoded_faces = face_recognition.face_encodings(imgS, faces_in_frame)
    for face_encode, faceloc in zip(encoded_faces, faces_in_frame):
        matches = face_recognition.compare_faces(encoded_face_train, face_encode)
        faceDist = face_recognition.face_distance(encoded_face_train, face_encode)
        matchIndex = np.argmin(faceDist)
        if matches[matchIndex]:
            name = classNames[matchIndex]
            shutil.copy(f"./group_photos/{cl}", f"./sorted_img/{name}/")
