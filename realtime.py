import cv2
import numpy as np
import pathlib
import time

import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam

data_dir = pathlib.Path('./data2')
img_height, img_width = 224, 224
batch_size=32

train_ds = tf.keras.preprocessing.image_dataset_from_directory(
                                                                data_dir,
                                                                validation_split=0.2,
                                                                subset="training",
                                                                seed=123,
                                                                image_size=(img_height, img_width),
                                                                batch_size=batch_size)


class_names = train_ds.class_names
print(class_names)

# resnet_model = Sequential()

# pretrained_model= tf.keras.applications.ResNet50(include_top=False,
#                    input_shape=(180,180,3),
#                    pooling='avg',classes=5,
#                    weights='imagenet')

# resnet_model.add(pretrained_model)
# resnet_model.add(Flatten())
# resnet_model.add(Dense(512, activation='relu'))
# resnet_model.add(Dense(5, activation='softmax'))

# resnet_model = tf.keras.models.load_model('./model.h5')

model = tf.keras.models.load_model('./NOE_MobileNet_V2.h5')
  
# define a video capture object
vid = cv2.VideoCapture(0)
vid.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
vid.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

image_to_analyze = None

txt = ''
while(True):
      
    # Capture the video frame
    # by frame
    ret, frame = vid.read()

    if cv2.waitKey(1) & 0xFF == ord('s'):
        image_to_analyze = frame.copy()
        
    if image_to_analyze is not None:
        t0 = round(time.time() * 1000)
        # Display the resulting frame
        image_resized = cv2.resize(image_to_analyze, (224, 224))
        image = np.expand_dims(image_resized, axis=0)
        pred = model.predict(image)
        t1 = round(time.time() * 1000)
        print('execution time: ', t1-t0)

        # tops = np.argsort(pred)

        output_class = class_names[np.argmax(pred)]
        if np.max(pred) >= 0.95:
            txt = output_class + '/ {:.2f}%'.format(np.max(pred)*100)
        else:
            txt = ''

        image_to_analyze = None

    # print(txt)
    # print(np.max(pred))
    
    cv2.putText(frame, txt, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(frame, 'Press "s" to capture and process image, "q" to quit.', (10, frame.shape[0]-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

    cv2.imshow('', frame)
    # the 'q' button is set as the
    # quitting button you may use any
    # desired button of your choice
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
  
# After the loop release the cap object
vid.release()
# Destroy all the windows
cv2.destroyAllWindows()
