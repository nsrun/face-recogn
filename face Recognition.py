import cv2
import numpy as np
from mtcnn import MTCNN
from keras_facenet import FaceNet
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt


detector = MTCNN()
embedder = FaceNet()

def preprocess_face(image_path):
    img = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = detector.detect_faces(img_rgb)
    if results:
        x, y, width, height = results[0]['box']
        face = img_rgb[y:y + height, x:x + width]
        face = cv2.resize(face, (160, 160))
        embedding = embedder.embeddings([face])[0]
        return embedding
    else:
        raise Exception("No face found in the image.")
        
known_embedding = preprocess_face("arunkumar.jpg")


cam = cv2.VideoCapture(0)
if not cam.isOpened():
    print("Camera not working")
    exit()

while True:
    ret, frame = cam.read()
    if not ret:
        print("Can't receive the frame")
        break

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = detector.detect_faces(frame_rgb)
    
    
    
    for res in results:
        x, y, w, h = res['box']
        face = frame_rgb[y:y + h, x:x + w]
        if face.shape[0] < 160 or face.shape[1] < 160:
            continue
        face_resized = cv2.resize(face, (160, 160))
        live_embedding = embedder.embeddings([face_resized])[0]
        
        
        similarity = cosine_similarity([known_embedding],[live_embedding])[0][0]
        
        match = similarity > 0.7
        
        frame = cv2.rectangle(frame, (x, y), (x + w, y + h), color=(0, 255, 0), thickness=2)
        
        label = 'arunkumar' if match else "Unknown"
        color = (0,255,0) if match else(0,0,255)
        cv2.putText(frame, label, ( x,y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
        
        
        if match:
            print("Arunkumar Enter,,,,")
            plt.imshow(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB))
            plt.title("Matched Face")
            plt.axis("off")
            plt.show()
            cam.release()
            cv2.destroyAllWindows()
            exit()
            
            
cam.release()
cv2.destroyAllWindows()


        
        