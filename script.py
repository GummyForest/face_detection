import cv2

detect = cv2.CascadeClassifier('haarcascade_frontalface_default.xml') # importing classifier for face detection

imp_img = cv2.VideoCapture('mark1.jpg') # capturing frames from the image
                                       # you can use a webcan - change the name of the image to 0 or 1 depending on your webcan

res, img = imp_img.read() # return TRUE/FALSE whether there is a image (res)
                          # also return the resolution of that image (img)             --> two variables are necessary

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # coverting to gray scale
                                             # (haarcascate is trainned to anaylize images in gray scale)

face = detect.detectMultiScale(gray, 1.3, 5) # detecting faces of diferent sizes
                                               #(gray_image, Scale Factor, minNeighbor)

for (x, y, w, h) in face:
    cv2.rectangle(img, (x, y), (x+w, y+h), (50, 205, 50), 3)
# square dimentions of detection --> width (x, x + w) , height (y, y + h)
# (image_name, most left/low vertex of retagle, most right/high vertex of retangle, color (hexadecimal), thikness of the border)

cv2.imshow("Elon Image", img) #showing image (title of the window, image name)

cv2.waitKey(0) #how much time the image will be presented? - In miliseconds (zero means that the window won't close)
imp_img.release()
cv2.destroyAllWindows()
