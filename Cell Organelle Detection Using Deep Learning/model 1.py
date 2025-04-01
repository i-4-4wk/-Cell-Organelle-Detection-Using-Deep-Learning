import cv2
import tensorflow as tf
import numpy as np
import os 


model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'cell_organelles_model.h5')


model = tf.keras.models.load_model(model_path)

cap = cv2.VideoCapture(0)

def detect_organelles(frame):
    input_image = cv2.resize(frame, (150, 150))  
    input_image = input_image / 255.0   
    input_image = np.expand_dims(input_image, axis=0)  
    
    predictions = model.predict(input_image)  

    predicted_class = np.argmax(predictions, axis=1)  
    return predicted_class[0], predictions

def get_label_from_class(class_index):
    labels = ['Centrosome', 'Golgi', 'Lysosome', 'Mitochondria', 'Nucleus', 'Rough endoplasm', 'Smooth endoplasm']
    return labels[class_index]

tracker = None
tracking_rect = None

while True:
    ret, frame = cap.read()
    if not ret:
        break

    if tracker is None:
        predicted_class, predictions = detect_organelles(frame)
        organelle_label = get_label_from_class(predicted_class)

        height, width, _ = frame.shape
        rect_width, rect_height = 100, 100  

        
        center_x, center_y = width // 2, height // 2
        top_left = (center_x - rect_width // 2, center_y - rect_height // 2)
        bottom_right = (center_x + rect_width // 2, center_y + rect_height // 2)

        
        cv2.rectangle(frame, top_left, bottom_right, (255, 0, 0), 2)

        
        cv2.putText(frame, organelle_label, (top_left[0], top_left[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

      
        tracker = cv2.TrackerCSRT_create()
        tracking_rect = (top_left[0], top_left[1], rect_width, rect_height)
        tracker.init(frame, tracking_rect)

    else:
        success, tracking_rect = tracker.update(frame)

        if success:
            (x, y, w, h) = tuple(map(int, tracking_rect))
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

         
            cv2.putText(frame, organelle_label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        else:
            tracker = None

   
    cv2.imshow("Live Video", frame)

    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release() 
cv2.destroyAllWindows()

                #                                              --/ devloped by i-cyber-ch4 \--
