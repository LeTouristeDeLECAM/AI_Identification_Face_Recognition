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

def boundary_Name_Box(image, name, face_coordinates):
    """
    This function draws a box around the face and writes the name of the person
    input : image, name of the person, coordinates of the face
    returns : image with the box and the name
    """
    (x,y,w,h) = face_coordinates
    cv2.rectangle(image, (x,y), (x+w, y+h), (0,255,0), 2)
    cv2.putText(image, name, (x,y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)


    return image 

#------------------------------------------------------------------------------------
# cut an image

def cut_image(image, coordinates):
    """
    This function cuts the image to the size of the coordinates
    input : image, coordinates of sub image
    returns : image with the size of the coordinates
    """
    (x,y,w,h) = coordinates
    return image[y:y+w, x:x+h]
   

#------------------------------------------------------------------------------------

def face_recognition(image, model_lbph):
    """
    This function recognizes the face in the image
    input : image, model
    returns : the label of the person and the coordinates of the face
    """
    img = image.copy()
    try :
        face, bounding_box = face_detection(img)
    except:
        print ("No face detected 1")
        
        
    label = model_lbph.predict(face)



    # label_text = database[label-1]
    # print (label[0])
    # print (label_text)
    # (x,y,w,h) = bounding_box

    # if label[0] == 1:  # refaire un tableau avec les noms
    #     name="Elise"
    # elif label[0] == 2:
    #     name="Matthias"
    # else:
    #     name="Unknown"

    # cv2.rectangle(img, (x,y), (x+w, y+h), (0,255,0), 5)
    # cv2.putText(img, name, (x,y), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 0), 2)



    return label[0],bounding_box

#------------------------------------------------------------------------------------



