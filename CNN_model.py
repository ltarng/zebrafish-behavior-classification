import numpy as np
import pandas as pd
import ml_model
import copy as cp
from keras.utils import to_categorical
from keras.utils.vis_utils import plot_model
from keras import Input
from keras.models import Sequential
from keras.layers import Conv1D, MaxPooling1D, AveragePooling1D, Flatten, Dense, BatchNormalization, Dropout
import keras.backend as KB

# Load the TensorBoard notebook extension
import tensorflow as tf
import datetime


def evaluate_model(X_train, y_train, X_test, y_test, n_filters):
    verbose, epochs, batch_size = 2, 100, 32
    num_timesteps, num_features, num_outputs = X_train.shape[1], X_train.shape[2], y_train.shape[1]
    # create model
    model = Sequential()
    model.add(Input(shape=(num_timesteps, num_features)))
    model.add(Conv1D(filters=16, kernel_size=1, activation='relu'))
    model.add(MaxPooling1D(pool_size=2, padding="same"))
    model.add(Flatten())
    model.add(Dense(num_outputs, activation='softmax'))

    #save an image of the network
    plot_model(model, show_shapes=True, to_file='sequential.png')
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    # fit model
    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=verbose)
    # evaluate model
    _, accuracy = model.evaluate(X_test, y_test, batch_size=batch_size, verbose=verbose)
    return accuracy


def cnn_model_1(X_train, y_train):  # better than model 2
    num_timesteps, num_features, num_outputs = X_train.shape[1], X_train.shape[2], y_train.shape[1]

    model = Sequential()
    model.add(Input(shape=(num_timesteps, num_features)))
    model.add(Conv1D(filters=16, kernel_size=1, activation='relu'))
    model.add(MaxPooling1D(pool_size=2, padding="same"))
    model.add(Flatten())
    model.add(Dense(num_outputs, activation='softmax'))
    return model


def cnn_model_2(X_train, y_train):
    num_timesteps, num_features, num_outputs = X_train.shape[1], X_train.shape[2], y_train.shape[1]

    model = Sequential()
    model.add(Input(shape=(num_timesteps, num_features)))
    model.add(Conv1D(filters=16, kernel_size=1, activation='relu'))
    model.add(AveragePooling1D(pool_size=2, padding="same"))
    model.add(Flatten())
    model.add(Dense(num_outputs, activation='softmax'))
    return model


def cnn_model_3(X_train, y_train):  # not the best?
    num_timesteps, num_features, num_outputs = X_train.shape[1], X_train.shape[2], y_train.shape[1]

    model = Sequential()
    model.add(Input(shape=(num_timesteps, num_features)))
    model.add(Conv1D(filters=16, kernel_size=1, activation='relu'))
    model.add(MaxPooling1D(pool_size=2, padding="same"))
    model.add(Flatten())
    model.add(Dense(num_outputs, activation='softmax'))
    return model


def cnn_model_4(X_train, y_train):  # validation test not good
    num_timesteps, num_features, num_outputs = X_train.shape[1], X_train.shape[2], y_train.shape[1]

    model = Sequential()
    model.add(Input(shape=(num_timesteps, num_features)))
    model.add(Conv1D(filters=16, kernel_size=1, activation='relu'))
    model.add(MaxPooling1D(pool_size=2, padding="same"))
    model.add(Conv1D(filters=16, kernel_size=1, activation='relu'))
    model.add(MaxPooling1D(pool_size=2, padding="same"))
    model.add(Conv1D(filters=16, kernel_size=1, activation='relu'))
    model.add(MaxPooling1D(pool_size=2, padding="same"))
    model.add(Flatten())
    model.add(Dense(num_outputs, activation='softmax'))
    return model


def deep_learning_main(folder_path, video_name, filter_name, model_name, feature):
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

    model_select = "cnn_4"
    # model
    if model_select == "cnn_1":
        model = cnn_model_1(X_train, y_train)
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        model.fit(X_train, y_train, validation_split=0.1, batch_size=32, epochs=250, verbose=2, callbacks=[tensorboard_callback])
        # batch_size = 32 is better than batch_size = 16
    elif model_select == "cnn_2":
        model = cnn_model_2(X_train, y_train)
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        model.fit(X_train, y_train, validation_split=0.1, batch_size=32, epochs=250, verbose=2, callbacks=[tensorboard_callback])
    elif model_select == 'cnn_3':
        model = cnn_model_3(X_train, y_train)
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        model.fit(X_train, y_train, validation_split=0.1, batch_size=32, epochs=250, verbose=2, callbacks=[tensorboard_callback])
    elif model_select == 'cnn_4':
        model = cnn_model_4(X_train, y_train)
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
    sorted_labels = ['bite', 'chase', 'display', 'normal']
    # print("y_test is: ",y_test, "\npred.  is: ", predictions)
    print(ml_model.classification_report(y_test, predictions))
    ml_model.plot_confusion_matrix(y_test, predictions, sorted_labels, model_name, feature, save_folder)


def getTrainDataInfo(X, y) -> ml_model.Tuple[np.array, np.array, np.array]:
    # Reshape data, X_train.shape[0]: number of train data, X_test.shape[0]: number of test data
    X = X.reshape(X.shape[0], 1, X.shape[1]).astype('float32')

    train_X, train_y = X[0], y[0]
    num_timesteps, num_features, num_outputs = train_X.shape[1], train_X.shape[2], train_y.shape[1]
    return num_timesteps, num_features, num_outputs


def cross_val_predict(model, skfold: ml_model.StratifiedKFold, X: np.array, y: np.array) -> ml_model.Tuple[np.array, np.array, np.array]:
    # Reference: https://towardsdatascience.com/how-to-plot-a-confusion-matrix-from-a-k-fold-cross-validation-b607317e9874
    model_ = cp.deepcopy(model)
    no_classes = len(np.unique(y))
    
    actual_classes = np.empty([0], dtype=int)
    predicted_classes = np.empty([0], dtype=int)
    predicted_proba = np.empty([0, no_classes]) 

    for train_ndx, test_ndx in skfold.split(X, y):

        train_X, train_y, test_X, test_y = X[train_ndx], y[train_ndx], X[test_ndx], y[test_ndx]

        # Reshape data, X_train.shape[0]: number of train data, X_test.shape[0]: number of test data
        train_X = train_X.reshape(train_X.shape[0], 1, train_X.shape[1]).astype('float32')
        test_X = test_X.reshape(test_X.shape[0], 1, test_X.shape[1]).astype('float32')

        # One-hot encoding, convert class vectors to binary class matrices
        train_y = to_categorical(train_y)
        test_y = to_categorical(test_y)

        actual_classes = np.append(actual_classes, test_y)

        model.fit(train_X, train_y, validation_split=0.1, batch_size=16, epochs=50, verbose=0)
        predicted_classes = np.append(predicted_classes, model_.predict(test_X))

        try:
            predicted_proba = np.append(predicted_proba, model_.predict_proba(test_X), axis=0)
        except:
            predicted_proba = np.append(predicted_proba, np.zeros((len(test_X), no_classes), dtype=float), axis=0)

    return actual_classes, predicted_classes, predicted_proba


# def deep_learning_main_cv_ver(folder_path, video_name, filter_name, model_name, feature):
#     # Read preprocessed trajectory data
#     df = pd.read_csv(folder_path + video_name + '_' + filter_name + '_preprocessed_result.csv')

#     # Split data into trainging data and testing data
#     X = ml_model.getFeaturesData(feature, df)
#     y = df['BehaviorType']

#     skfold = ml_model.StratifiedKFold(n_splits=10, random_state=99, shuffle=True)  # Split data evenly among different behavior data type

#     # Some parameter setting for model
#     num_timesteps, num_features, num_outputs = getTrainDataInfo(X, y)

#     # Create model
#     model = Sequential()
#     model.add(Input(shape=(num_timesteps, num_features)))
#     model.add(Conv1D(filters=16, kernel_size=1, activation='relu'))
#     model.add(MaxPooling1D(pool_size=2, padding="same"))
#     model.add(Flatten())
#     model.add(Dense(num_outputs, activation='softmax'))

#     # 10-fold  cross validation
#     start_time = ml_model.process_time()
#     actual_classes, predicted_classes, _ = cross_val_predict(model, skfold, X, y)
#     end_time = ml_model.process_time()

#     # Show the testing result with confusion matrix
#     print(model_name, feature)
#     print(ml_model.classification_report(actual_classes, predicted_classes))
#     print("Class number meaning - 1:bite, 2:chase, 3:circle, 4:display, 5:normal")

#     # Setting the path and create a folder to save confusion matrix pictures
#     save_folder = ml_model.create_saving_folder(folder_path)

#     # Plot the confusion matrix graph on screen, and save it in png format
#     sorted_labels = ['bite', 'chase', 'display', 'normal']
#     ml_model.plot_confusion_matrix(actual_classes, predicted_classes, sorted_labels, model_name, feature, save_folder)
#     print("Execution time: ", end_time - start_time)
