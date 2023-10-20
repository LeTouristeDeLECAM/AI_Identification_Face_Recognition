from ultralytics import YOLO
import Library.package_face_recognition as fr
import cv2


# Load the model
model_lbph = YOLO("Library/yolov8l.pt")

# Load the image

image = cv2.imread("Images/face_recognition.jpg")

# Detect the objects
results = model_lbph.predict(img)

result = results[0]
box = result.boxes[0]


# if object is a person then crop the image and detect the face
for i in range(len(result.boxes)):
    box = result.boxes[i]
    cords = box.xyxy[0].tolist()
    cords = [round(x) for x in cords]
    class_id = result.names[box.cls[0].item()]
    conf = round(box.conf[0].item(), 2)
    if class_id == 'person':
        cropped_image = fr.cut_image(image, cords)

        # cropped_image = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB)
        cv2.imshow('cropped_image', cropped_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        try:
            label, coordinates = fr.face_recognition(cropped_image, model_lbph)
        except:
            print("No face detected")

        # Load the label dictionary
        label_dict = np.load('Library/label_dict.npy', allow_pickle=True).item()

        # find the corresponding name for the integer label
        for name, l in label_dict.items():
            if l == label:
                print(name)
                break
  

        
        image = fr.boundary_Name_Box(image, name, coordinates)




