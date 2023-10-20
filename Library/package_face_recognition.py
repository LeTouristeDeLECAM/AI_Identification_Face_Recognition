# Here are the developed libraries

import cv2


def face_detection(image):
    """
    This function detects the face in the image
    input : image
    returns : the cropped image and the coordinates of the face
    """
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # cv2.imshow('gray', image_gray)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    haar_classifier = cv2.CascadeClassifier('Library\haarcascade_frontalface_default.xml')
    
    face = haar_classifier.detectMultiScale(image_gray, scaleFactor=1.3, minNeighbors=7)
    # print (face[0])
    (x,y,w,h) = face[0]
    # print(x,y,w,h)
    
    return image_gray[y:y+w, x:x+h], face[0]

#------------------------------------------------------------------------------------

