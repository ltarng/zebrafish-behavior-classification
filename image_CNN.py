import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix
from keras.layers import InputLayer, Lambda, Dense, Flatten, Dropout, Conv2D, MaxPooling2D
from keras.models import Sequential, Model
from keras.applications.vgg19 import VGG19
from keras.applications.vgg19 import preprocess_input
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import to_categorical
# Load the TensorBoard notebook extension
import keras.backend as KB
import tensorflow as tf
import datetime
import seaborn as sns

folder_path = "D:/Google Cloud (60747050S)/Research/Trajectory Analysis/"


def read_images_in_folder(resource_path, img_width, img_height):
    img_arrays = []
    img_labels = []
    for folder in os.listdir(resource_path):
        sub_path = resource_path + "/" + folder
        for img in os.listdir(sub_path):
            image_path = sub_path + "/" + img
            img_arr = cv2.imread(image_path)
            img_arr = cv2.resize(img_arr, (img_width, img_height))
            img_arrays.append(img_arr)
            img_labels.append(img.split('_')[0])  # get labels: bite, chase, display, normal
    return img_arrays, img_labels


def images_normalization(img_arrays_train):
    train_img_arrays = np.array(img_arrays_train)
    train_img_arrays = train_img_arrays / 255.0
    return train_img_arrays


def convert_labels_from_str_to_int(img_labels):
    behavior_dict = {'bite': 0, 'chase': 1, 'display': 2, 'normal': 3}  # be aware of setting numbers, it's related to the result of one-hot encode

    img_labels_digit = []
    for index in range(0, len(img_labels)):
        label_name = img_labels[index]
        if label_name in behavior_dict.keys():
            img_labels_digit.append(behavior_dict[label_name])

    return img_labels_digit


def plot_train_info_graph(history, info_type):
    plt.plot(history.history[info_type], label='train ' + info_type)
    plt.plot(history.history['val_' + info_type], label='val ' + info_type)
    plt.legend()
    plt.savefig('vgg-' + info_type + '-rps-1.png')
    plt.show()


def create_saving_folder(folder_path):
    # Setting the path for saving confusion matrix pictures
    save_folder = folder_path + "confusion_matrix_picture/"
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    return save_folder


def plot_confusion_matrix(actual_classes, predicted_classes, model_name, save_folder):
    sorted_labels = ['bite', 'chase', 'display', 'normal']
    cm = confusion_matrix(actual_classes, predicted_classes)
    sns.heatmap(cm, square=True, annot=True, cmap='Blues', xticklabels=sorted_labels, yticklabels=sorted_labels)
    # sns.despine(left=False, right=False, top=False, bottom=False)

    # Text part
    plt.title('Confusion Matrix of ' + model_name + ' Classifier')
    plt.xlabel("Predicted Value")
    plt.ylabel("True Value")
    plt.tight_layout()

    plt.savefig(save_folder + model_name + "_confusion_matrix.png")
    plt.show()


def cnn_model_1(img_width, img_height):
    # Be aware of this number. If set wrong classes amount, it may lead to ValueError when execute model.fit()
    num_outputs = 4

    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(img_width, img_height, 3)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(50, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), padding="same"))
    model.add(Flatten())
    model.add(Dense(units=100, activation='relu'))
    model.add(Dense(num_outputs, activation='softmax'))
    return model


def my_model_main():
    # Basic setting
    model_name = "imageCNN_self"
    img_width, img_height = 224, 224
    train_path = "D:/Trajectory(image)/train/"
    test_path = "D:/Trajectory(image)/test/"

    # Get trajectory image data from folder
    x_train, train_labels = read_images_in_folder(train_path, img_width, img_height)
    x_test, test_labels = read_images_in_folder(test_path, img_width, img_height)

    # Convert labels from string to integer
    y_train = convert_labels_from_str_to_int(train_labels)
    y_test = convert_labels_from_str_to_int(test_labels)
    y_train, y_test = to_categorical(y_train), to_categorical(y_test)  # One-hot encoding

    # Normalization dataset
    train_x = images_normalization(x_train)
    test_x = images_normalization(x_test)


    # Setting about tensorboard
    log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

    # Fit the model
    model = cnn_model_1(img_width, img_height)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    history = model.fit(train_x, y_train, validation_split=0.1, batch_size=16, epochs=5, verbose=2, callbacks=[tensorboard_callback])
    print(model.summary())
    print(KB.eval(model.optimizer.lr))


    # Plot image about the training result
    plot_train_info_graph(history, 'accuracy')
    plot_train_info_graph(history, 'loss')

    # Predict and get classification report
    y_pred, y_test = np.argmax(model.predict(test_x), axis=1), np.argmax(y_test, axis=1)
    print(classification_report(y_pred, y_test))
    print(confusion_matrix(y_pred, y_test))

    # Setting the path for saving folder and plot confusion matrix for the result
    save_folder = create_saving_folder(folder_path)
    plot_confusion_matrix(y_test, y_pred, model_name, save_folder)


def vgg19_main():
    # https://www.analyticsvidhya.com/blog/2021/07/step-by-step-guide-for-image-classification-on-custom-datasets/
    model_name = "imageCNN_vgg19"
    img_width, img_height = 224, 224
    num_outputs = 4

    train_path = "D:/Trajectory(image)/train/"
    test_path = "D:/Trajectory(image)/test/"
    # val_path =  "D:/Trajectory(image)_val/val/"


    # Get trajectory image data from folder
    x_train, train_labels = read_images_in_folder(train_path, img_width, img_height)
    x_test, test_labels = read_images_in_folder(test_path, img_width, img_height)
    # x_val = read_images_in_folder(val_path, img_width, img_height)

    # Normalization
    train_x = images_normalization(x_train)
    test_x = images_normalization(x_test)
    # val_x = images_normalization(x_val)


    # Compute the labels of the corresponding datasets using ImageDataGenerator
    train_datagen = ImageDataGenerator(rescale = 1./255, horizontal_flip=True)
    test_datagen = ImageDataGenerator(rescale = 1./255)
    # val_datagen = ImageDataGenerator(rescale = 1./255)

    training_set = train_datagen.flow_from_directory(train_path,
                                                    target_size = (img_width, img_height),
                                                    batch_size = 32,
                                                    class_mode = 'sparse')
    test_set = test_datagen.flow_from_directory(test_path,
                                                target_size = (img_width, img_height),
                                                batch_size = 32,
                                                class_mode = 'sparse')
    # val_set = val_datagen.flow_from_directory(val_path,
    #                                             target_size = (224, 224),
    #                                             batch_size = 32,
    #                                             class_mode = 'categorical')

    train_y = training_set.classes
    test_y = test_set.classes
    # val_y = val_set.classes

    training_set.class_indices
    train_y.shape, test_y.shape
    # train_y.shape, test_y.shape, val_y.shape


    # Model Training
    vgg = VGG19(input_shape = (img_width, img_height, 3), weights='imagenet', include_top=False)
    
    for layer in vgg.layers:  #do not train the pre-trained layers of VGG-19
        layer.trainable = False

    x = Flatten()(vgg.output)

    # Add output layer.Softmax classifier is used as it is multi-class classification
    prediction = Dense(num_outputs, activation='softmax')(x)  # have to set the dense number as the classes number
    model = Model(inputs=vgg.input, outputs=prediction)
    model.summary()  # view the structure of the model


    # Setting about tensorboard
    log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

    # Fit the model
    model.compile(loss='sparse_categorical_crossentropy', optimizer="adam", metrics=['accuracy'])
    # history = model.fit(train_x, train_y, validation_data=(val_x, val_y), validation_split=0.1, batch_size=32, epochs=10, callbacks=[tensorboard_callback])
    history = model.fit(train_x, train_y, validation_split=0.1, batch_size=32, epochs=5, callbacks=[tensorboard_callback])

    # Plot image about the training result
    plot_train_info_graph(history, 'accuracy')
    plot_train_info_graph(history, 'loss')


    # Eveluate Model
    model.evaluate(test_x, test_y, batch_size=32)

    # Predict and get classification report
    y_pred = np.argmax(model.predict(test_set), axis=1)
    print(classification_report(y_pred, test_y))
    print(confusion_matrix(y_pred, test_y))

    # Setting the path for saving folder plot confusion matrix for the result
    save_folder = create_saving_folder(folder_path)
    plot_confusion_matrix(test_y, y_pred, model_name, save_folder)


def vgg19_revised():
    model_name = "imageCNN_vgg19_revised"
    img_width, img_height = 224, 224
    num_outputs = 4

    train_path = "D:/Trajectory(image)/train/"
    test_path = "D:/Trajectory(image)/test/"


    # Get trajectory image data from folder
    x_train, train_labels = read_images_in_folder(train_path, img_width, img_height)
    x_test, test_labels = read_images_in_folder(test_path, img_width, img_height)

    # Convert labels from string to integer
    y_train = convert_labels_from_str_to_int(train_labels)
    y_test = convert_labels_from_str_to_int(test_labels)

    # One-hot encoding, convert class vectors to binary class matrices
    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)

    # Normalization
    train_x = images_normalization(x_train)
    test_x = images_normalization(x_test)


    # Model Training
    vgg = VGG19(input_shape = (img_width, img_height, 3), weights='imagenet', include_top=False)
    
    for layer in vgg.layers:  #do not train the pre-trained layers of VGG-19
        layer.trainable = False

    x = Flatten()(vgg.output)

    #adding output layer.Softmax classifier is used as it is multi-class classification
    prediction = Dense(num_outputs, activation='softmax')(x)  # have to set the dense number as the classes number
    model = Model(inputs=vgg.input, outputs=prediction)
    model.summary()  # view the structure of the model

    # Setting about tensorboard
    log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

    # Fit the model
    model.compile(loss='categorical_crossentropy', optimizer="adam", metrics=['accuracy'])
    history = model.fit(train_x, y_train, validation_split=0.1, batch_size=32, epochs=5, verbose=2, callbacks=[tensorboard_callback])


    # Plot image about the training result
    plot_train_info_graph(history, 'accuracy')
    plot_train_info_graph(history, 'loss')

    # Predict and get classification report
    y_pred = np.argmax(model.predict(test_x), axis=1)
    y_test = np.argmax(y_test, axis=1)   

    print(classification_report(y_pred, y_test))
    print(confusion_matrix(y_pred, y_test))
    
    # Setting the path for saving folder plot confusion matrix for the result
    save_folder = create_saving_folder(folder_path)
    plot_confusion_matrix(y_test, y_pred, model_name, save_folder)


if __name__ == '__main__':
    vgg19_main()
    # vgg19_revised()
    # my_model_main()
