import cv2

detected_facedata = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
webcamera_cap = cv2.VideoCapture(0)
while True:
    sucessfull_frame,img = webcamera_cap.read()
    #must covert to gray scale
    grayscaled_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    #Detect face
    face_coordinates = detected_facedata.detectMultiScale(grayscaled_img)
    #Draw rectangle around the face
    for (x,y,w,h) in face_coordinates:
        cv2.rectangle(img,(x,y),(x+w , y+h),(0,255,0),2)
    cv2.imshow('camera face detect',img)
    cv2.waitKey(1)




