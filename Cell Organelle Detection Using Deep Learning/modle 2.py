import cv2
import tensorflow as tf
import numpy as np
import os


model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'cell_organelles_model.h5')


model = tf.keras.models.load_model(model_path)  


cap = cv2.VideoCapture(0)

frame_width = int(cap.get(3))
frame_height = int(cap.get(5))
out = cv2.VideoWriter('output.avi', cv2.VideoWriter_fourcc(*'XVID'), 20.0, (frame_width, frame_height))

clicked_points = []


def click_event(event, x, y, flags, param):
    global clicked_points, frame
    if event == cv2.EVENT_LBUTTONDOWN:
        clicked_points.append((x, y))
        print(f"Clicked at: {x}, {y}")
        
        cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)
        cv2.imshow("Live Video", frame)  


def detect_organelles(frame):
   
    input_image = cv2.resize(frame, (150, 150)) 
    input_image = input_image / 255.0            
    input_image = np.expand_dims(input_image, axis=0)  

    
    predictions = model.predict(input_image)

   
    predicted_class = np.argmax(predictions, axis=1)
    return predicted_class[0], predictions


def get_label_from_class(class_index):
    labels = ['Centrosome', 'Golgi', 'Lysosome', 'Mitochondria', 'Nucleus' ,'Rough endoplasm' ,'smooth endoplasm']
    return labels[class_index]


while True:
    ret, frame = cap.read()
    if not ret:
        break

   
    predicted_class, predictions = detect_organelles(frame)
    organelle_label = get_label_from_class(predicted_class)

  
    cv2.putText(frame, f"Detected: {organelle_label}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255, 0), 2)

   
    if clicked_points:
      
        x, y = clicked_points[-1]
        width, height = 50, 50  
        cv2.rectangle(frame, (x, y), (x + width, y + height), (255, 0, 0), 2)
        cv2.putText(frame, organelle_label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

    
    out.write(frame)
    
    cv2.imshow("Live Video", frame)

    
    cv2.setMouseCallback("Live Video", click_event)

    
    key = cv2.waitKey(10) & 0xFF
    if key == ord('q'):
         break



cap.release()  
out.release()
cv2.destroyAllWindows() 





#                                          --/ devloped by i-cyber-ch4 \--