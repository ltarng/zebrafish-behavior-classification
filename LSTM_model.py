import numpy as np
import pandas as pd
import ml_model
from keras.utils import to_categorical
from keras import Input
from keras.models import Sequential
from keras.layers import LSTM, Dense, Embedding, Dropout
import keras.backend as KB
# Load the TensorBoard notebook extension
import tensorflow as tf
import datetime


def lstm_model_1(X_train, y_train):
    num_timesteps, num_features, num_outputs = X_train.shape[1], X_train.shape[2], y_train.shape[1]

    MAX_WORDS = 500
    MAX_LENGTH = 75
    EMBEDDING_DIM = 1

    model = Sequential()
    model.add(Input(shape=(num_timesteps, num_features)))
    # model.add(Embedding(MAX_WORDS, EMBEDDING_DIM, input_length=MAX_LENGTH))
    model.add(LSTM(32))
    # model.add(Dropout(0.2))
    model.add(Dense(16, activation='softmax'))
    model.add(Dense(num_outputs, activation="softmax"))

    return model


def lstm_main(folder_path, video_name, filter_name, model_name, feature):
    # Read preprocessed trajectory data
    df = pd.read_csv(folder_path + video_name + '_' + filter_name + '_preprocessed_result.csv')

    # Split data into trainging data and testing data
    X = ml_model.getFeaturesData(feature, df)
    y = df['BehaviorType']
    X_train, X_test, y_train, y_test = ml_model.train_test_split(X, y, test_size=0.20, random_state=54, stratify=y)

    # Reshape data, X_train.shape[0]: number of train data, X_test.shape[0]: number of test data
    X_train = X_train.reshape(X_train.shape[0], 1, X_train.shape[1]).astype('float32')
    X_test = X_test.reshape(X_test.shape[0], 1, X_test.shape[1]).astype('float32')
    # One-hot encoding, convert class vectors to binary class matrices
    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)

    log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

    model = lstm_model_1(X_train, y_train)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.fit(X_train, y_train, validation_split=0.1, batch_size=32, epochs=250, verbose=2, callbacks=[tensorboard_callback])

    print(model.summary())
    print(KB.eval(model.optimizer.lr))

    # Predict
    predictions = np.argmax(model.predict(X_test), axis=1)
    y_test = np.argmax(y_test, axis=1)

    # Setting the path for saving confusion matrix pictures
    save_folder = ml_model.create_saving_folder(folder_path)
    
    # Plot confusion matrix for the result
    # print("y_test is: ",y_test, "\npred.  is: ", predictions)
    print(ml_model.classification_report(y_test, predictions))
    sorted_labels = ['bite', 'chase', 'display', 'normal']
    ml_model.plot_confusion_matrix(y_test, predictions, sorted_labels, model_name, feature, save_folder)


def lstm_main_3categories(folder_path, video_name, filter_name, model_name, feature):
    # Read preprocessed trajectory data
    df = pd.read_csv(folder_path + video_name + '_' + filter_name + '_preprocessed_result.csv')

    # Split data into trainging data and testing data
    X = ml_model.getFeaturesData(feature, df)
    y = df['BehaviorType']
    X_train, X_test, y_train, y_test = ml_model.train_test_split(X, y, test_size=0.20, random_state=54, stratify=y)

    # Combine bite and chase
    for index in range(0, len(df['BehaviorType'].index)):
        if df['BehaviorType'].iloc[index] == 1 or df['BehaviorType'].iloc[index] == 2:
            df['BehaviorType'].iloc[index] = 1
            

    # Reshape data, X_train.shape[0]: number of train data, X_test.shape[0]: number of test data
    X_train = X_train.reshape(X_train.shape[0], 1, X_train.shape[1]).astype('float32')
    X_test = X_test.reshape(X_test.shape[0], 1, X_test.shape[1]).astype('float32')
    # One-hot encoding, convert class vectors to binary class matrices
    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)

    log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

    model = lstm_model_1(X_train, y_train)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    start_time = ml_model.process_time()
    model.fit(X_train, y_train, validation_split=0.1, batch_size=32, epochs=250, verbose=2, callbacks=[tensorboard_callback])
    end_time = ml_model.process_time()

    print(model.summary())
    print(KB.eval(model.optimizer.lr))

    # Predict
    predictions = np.argmax(model.predict(X_test), axis=1)
    y_test = np.argmax(y_test, axis=1)

    # Setting the path for saving confusion matrix pictures
    save_folder = ml_model.create_saving_folder(folder_path)
    
    # Plot confusion matrix for the result
    # print("y_test is: ",y_test, "\npred.  is: ", predictions)
    print(ml_model.classification_report(y_test, predictions))
    sorted_labels = ['bite', 'chase', 'display', 'normal']
    ml_model.plot_confusion_matrix(y_test, predictions, sorted_labels, model_name, feature, save_folder)
    print("Execution time: ", end_time - start_time)
