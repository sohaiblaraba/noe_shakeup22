import matplotlib.pyplot as plt
import pathlib
import argparse

import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam


MODELS_SIZES = {'ResNet50': 180,
                'DenseNet121': 224,
                'MobileNet': 224,
                'MobileNet_V2': 224,
                'MobileNetV3Small': 224
                }

def get_model(modelname, img_size, n_classes):
        if model_name == 'ResNet50':
                pretrained_model= tf.keras.applications.ResNet50(include_top=False,
                                input_shape=(img_size, img_size, 3),
                                pooling='avg', classes=n_classes,
                                weights='imagenet')

        elif model_name == 'DenseNet121':
                pretrained_model= tf.keras.applications.densenet.DenseNet121(include_top=False,
                                input_shape=(img_size, img_size, 3),
                                pooling='avg', classes=n_classes,
                                weights='imagenet')
        elif model_name == 'MobileNet':
                pretrained_model= tf.keras.applications.mobilenet.MobileNet(include_top=False,
                                input_shape=(img_size, img_size, 3),
                                pooling='avg', classes=n_classes,
                                weights='imagenet')
        elif model_name == 'MobileNetV2':
                pretrained_model= tf.keras.applications.mobilenet_v2.MobileNetV2(include_top=False,
                                input_shape=(img_size, img_size, 3),
                                pooling='avg', classes=n_classes,
                                weights='imagenet')
        elif model_name == 'MobileNetV3Small':
                pretrained_model= tf.keras.applications.MobileNetV3Small(include_top=False,
                                input_shape=(img_size, img_size, 3),
                                pooling='avg', classes=n_classes,
                                weights='imagenet')
        else:
                return None

        model = Sequential()
        for layer in pretrained_model.layers:
                layer.trainable=False

        model.add(pretrained_model)
        model.add(Flatten())
        model.add(Dense(512, activation='relu'))
        model.add(Dense(n_classes, activation='softmax'))

        print(model.summary())

        return model


def load_dataset(data_path, batch_size, split):
        try:
                data_dir = pathlib.Path(data_path)
                img_height, img_width = MODELS_SIZES[model_name], MODELS_SIZES[model_name]
                train_ds = tf.keras.preprocessing.image_dataset_from_directory(
                                                                                data_dir,
                                                                                validation_split=split,
                                                                                subset="training",
                                                                                seed=123,
                                                                                image_size=(img_height, img_width),
                                                                                batch_size=batch_size)
                                                                                
                val_ds = tf.keras.preprocessing.image_dataset_from_directory(
                                                                                data_dir,
                                                                                validation_split=split,
                                                                                subset="validation",
                                                                                seed=123,
                                                                                image_size=(img_height, img_width),
                                                                                batch_size=batch_size)
                return train_ds, val_ds

        except Exception as e:
                print("Exception when getting item : ", e)
        

if __name__ == '__main__':
        # Initialize the Parser
        parser = argparse.ArgumentParser(description ='Run training of a classification model.')
        parser.add_argument('--data', type=str, default='./data', help='Path to the dataset: it should contain \
                a list of folders with names corresponding to classes, each contains correponding images')
        parser.add_argument('--split', type=float, default=0.2, help='Value to split data to train and val. \
                Ex: --split 0.2 means that 80% of data will be used in training and 20% for validation')
        parser.add_argument('--model', type=str, default='ResNet50', help='Select model to train.')
        parser.add_argument('--batch', type=int, default=32, help='Batch size.')
        parser.add_argument('--epoch', type=int, default=10, help='Number of epochs.')
        parser.add_argument('--save_model', default='./models/model.h5', help='Path to model to be saved.')
        parser.add_argument('--save_history', default=None, help='Path to accuracy and loss history during training.')

        args = parser.parse_args()

        data_path = args.data
        split = args.split
        model_name = args.model
        batch = args.batch
        epochs = args.epoch
        path_to_save_model = args.save_model
        history_path = args.save_history

        train_ds, val_ds = load_dataset(data_path, batch, split)
        n_classes = len(train_ds.class_names)

        model = get_model(model_name, img_size=MODELS_SIZES[model_name], n_classes=n_classes)
        
        if model is not None:
                print('[INFO] Preparing model:', model_name)
                model.compile(optimizer=Adam(lr=0.001),loss='sparse_categorical_crossentropy',metrics=['accuracy'])

                print('[INFO] Training...')
                history = model.fit(train_ds,
                                    validation_data=val_ds,
                                    epochs=epochs,
                                    verbose=1)

                model.save(path_to_save_model)

                plt.subplot(1, 2, 1)
                plt.plot(history.history['accuracy'])
                plt.plot(history.history['val_accuracy'])
                plt.axis(ymin=0.4,ymax=1)
                plt.grid()
                plt.title('Model Accuracy')
                plt.ylabel('Accuracy')
                plt.xlabel('Epochs')
                plt.legend(['train', 'validation'])

                plt.subplot(1, 2, 2)
                plt.plot(history.history['loss'])
                plt.plot(history.history['val_loss'])
                plt.grid()
                plt.title('Model Loss')
                plt.ylabel('Loss')
                plt.xlabel('Epochs')
                plt.legend(['train', 'validation'])

                if history_path is not None:
                        plt.savefig(history_path)
                        
                plt.show()


        else:
                print('[ERROR] Model name not available. Please choose one model from this list:')
                print('- ResNet50')
                print('- DenseNet121')
                print('- MobileNet')
                print('- MobileNetV2')
                print('- MobileNetV3Small')


