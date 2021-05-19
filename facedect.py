import cv2,glob

gimage = glob.glob('*.jpg')
print(gimage)

detect=cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
for timage in gimage:
    image = cv2.imread(timage)
    graying = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    face= detect.detectMultiScale(graying, 1.25, 3)
    for (x,y,w,h) in face:
        cv2.rectangle(image, (x,y), (x+w , y+h), (0,255,0),2)
    cv2.imshow("photo dectected image",image)
    cv2.waitKey(20000)
    cv2.destroyAllWindows()