import cv2
import numpy as np
import pathlib
import argparse

import tensorflow as tf


MODELS_SIZES = {'ResNet50': 180,
                'DenseNet121': 224,
                'MobileNet': 224,
                'MobileNet_V2': 224,
                'MobileNetV3Small': 224
                }


if __name__ == '__main__':
        CLASSES = ['Bouchon de liege', 'Lunettes', 'Paquet de chips', 'Pile', 'Pot de yaourt']
        CLASSES.sort()

        parser = argparse.ArgumentParser(description ='Run training of a classification model.')
        parser.add_argument('--model', type=str, required=True, help='Model path.')
        parser.add_argument('--image', type=str, required=True, help='Image to analyse.')

        args = parser.parse_args()

        model_path = args.model
        img_path = args.image

        model = tf.keras.models.load_model(model_path)

        n_classes = model.get_layer('dense_1').output_shape[1]

        image = cv2.imread(img_path)
        image_resized = cv2.resize(image, (180, 180))
        image = np.expand_dims(image_resized,axis=0)

        pred = model.predict(image)[0]

        top_values_index = np.argsort(-pred)[:3]

        for idx in top_values_index:
            print(CLASSES[idx], '{:.2f}'.format(pred[idx]*100) + '%')










'''
















data_dir = pathlib.Path('./data2')
img_height,img_width=180,180
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
resnet_model = tf.keras.models.load_model('./MobileNetV3Small.h5')


img_path = './testdata/4.jpg'

image = cv2.imread(img_path)
image_resized = cv2.resize(image, (180, 180))
image = np.expand_dims(image_resized,axis=0)
print(image.shape)

pred = resnet_model.predict(image)
print(pred)

output_class = class_names[np.argmax(pred)]
print("The predicted class is", output_class)

'''