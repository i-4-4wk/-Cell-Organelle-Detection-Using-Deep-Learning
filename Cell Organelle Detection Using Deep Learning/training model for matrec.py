import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Flatten, Dense
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt


dataset_dir = 'dataset2'  


train_datagen = ImageDataGenerator(
    rescale=1./255,              
    rotation_range=40,            
    width_shift_range=0.2,       
    height_shift_range=0.2,      
    shear_range=0.2,              
    zoom_range=0.2,              
    horizontal_flip=True,         
    fill_mode='nearest'           
)


train_generator = train_datagen.flow_from_directory(
    dataset_dir,
    target_size=(150, 150),      
    batch_size=32,                
    class_mode='categorical'      
)


base_model = VGG16(weights='imagenet', include_top=False, input_shape=(150, 150, 3))
for layer in base_model.layers:
    layer.trainable = False


x = Flatten()(base_model.output) 
x = Dense(512, activation='relu')(x)  
output = Dense(len(train_generator.class_indices), activation='softmax')(x)  


model = Model(inputs=base_model.input, outputs=output)


model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])


history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // train_generator.batch_size,
    epochs=10  
)


model.save('cell_organelles_model.h5')


plt.plot(history.history['accuracy'])
plt.title('Model Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.show()


plt.plot(history.history['loss'])
plt.title('Model Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.show()
