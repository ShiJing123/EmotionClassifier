from __future__ import print_function

from keras.applications import VGG16
from keras.preprocessing.image import ImageDataGenerator

import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display

from keras.models import Sequential, Model
from keras.optimizers import SGD, Nadam, RMSprop, Adam
from keras.layers import Input, Dense, TimeDistributed, LSTM, Dropout, Activation
from keras.layers import Conv2D, MaxPooling2D, Flatten
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import ELU, LeakyReLU, PReLU
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras import backend
from keras.utils import np_utils
import os
from os.path import isfile

from timeit import default_timer as timer

mono = True


def get_class_names(path="Preproc/"):  # class names are subdirectory names in Preproc/ directory
    class_names = os.listdir(path)
    return class_names


def get_total_files(path="Preproc/", train_percentage=0.8):
    sum_total = 0
    sum_train = 0
    sum_test = 0
    subdirs = os.listdir(path)
    for subdir in subdirs:
        files = os.listdir(path + subdir)
        n_files = len(files)
        sum_total += n_files
        n_train = int(train_percentage * n_files)
        n_test = n_files - n_train
        sum_train += n_train
        sum_test += n_test
    return sum_total, sum_train, sum_test


def get_sample_dimensions(path='Preproc/'):
    classname = os.listdir(path)[0]
    files = os.listdir(path + classname)
    infilename = files[0]
    audio_path = path + classname + '/' + infilename
    melgram = np.load(audio_path)
    print("   get_sample_dimensions: melgram.shape = ", melgram.shape)
    return melgram.shape


def encode_class(class_name, class_names):  # makes a "one-hot" vector for each class name called
    try:
        idx = class_names.index(class_name)
        vec = np.zeros(len(class_names))
        vec[idx] = 1
        return vec
    except ValueError:
        return None


def shuffle_XY_paths(X, Y, paths):  # generates a randomized order, keeping X&Y(&paths) together
    assert (X.shape[0] == Y.shape[0])
    idx = np.array(range(Y.shape[0]))
    np.random.shuffle(idx)
    newX = np.copy(X)
    newY = np.copy(Y)
    newpaths = paths
    for i in range(len(idx)):
        newX[i] = X[idx[i], :, :]
        newY[i] = Y[idx[i], :]
        newpaths[i] = paths[idx[i]]
    return newX, newY, newpaths


'''
So we make the training & testing datasets here, and we do it separately.
Why not just make one big dataset, shuffle, and then split into train & test?
because we want to make sure statistics in training & testing are as similar as possible
'''


def build_datasets(train_percentage=0.8, preproc=False):
    if (preproc):
        path = "Preproc/"
    else:
        path = "audio/"

    class_names = get_class_names(path=path)
    print("class_names = ", class_names)

    total_files, total_train, total_test = get_total_files(path=path, train_percentage=train_percentage)
    print("total files = ", total_files)

    nb_classes = len(class_names)

    # pre-allocate memory for speed (old method used np.concatenate, slow)
    mel_dims = get_sample_dimensions(path=path)  # Find out the 'shape' of each data file
    X_train = np.zeros((total_train,  mel_dims[2], mel_dims[3]))
    Y_train = np.zeros((total_train, nb_classes))
    X_test = np.zeros((total_test,  mel_dims[2], mel_dims[3]))
    Y_test = np.zeros((total_test, nb_classes))
    paths_train = []
    paths_test = []

    train_count = 0
    test_count = 0
    for idx, classname in enumerate(class_names):
        this_Y = np.array(encode_class(classname, class_names))
        this_Y = this_Y[np.newaxis, :]
        class_files = os.listdir(path + classname)
        n_files = len(class_files)
        n_load = n_files
        n_train = int(train_percentage * n_load)
        printevery = 100
        print("")
        for idx2, infilename in enumerate(class_files[0:n_load]):
            audio_path = path + classname + '/' + infilename
            if (0 == idx2 % printevery):
                print('\r Loading class: {:14s} ({:2d} of {:2d} classes)'.format(classname, idx + 1, nb_classes),
                      ", file ", idx2 + 1, " of ", n_load, ": ", audio_path, sep="")
            # start = timer()
            if (preproc):
                melgram = np.load(audio_path)
                sr = 44100
            else:
                aud, sr = librosa.load(audio_path, mono=mono, sr=None)
                melgram = librosa.amplitude_to_db(librosa.feature.melspectrogram(aud, sr=sr, n_mels=96), ref=1.0)[
                          np.newaxis, np.newaxis, :, :]

            #melgram = melgram[:, :, :, 0:mel_dims[3]]  # just in case files are differnt sizes: clip to first file size

            # end = timer()
            # print("time = ",end - start)
            if (idx2 < n_train):
                # concatenate is SLOW for big datasets; use pre-allocated instead
                # X_train = np.concatenate((X_train, melgram), axis=0)
                # Y_train = np.concatenate((Y_train, this_Y), axis=0)
                # (a,b,c,d)  = np.shape((X_train))
                # melgram_resized = np.copy(melgram).resize((b,c,d))
                # X_train[train_count, :, :] = melgram_resized
                X_train[train_count,  :] = melgram
                Y_train[train_count, :] = this_Y
                paths_train.append(audio_path)  # list-appending is still fast. (??)
                train_count += 1
            else:
                # (a,b,c,d)  = np.shape((X_test))
                # melgram_resized = np.copy(melgram).resize((b,c,d))
                # X_test[test_count, :, :] = melgram_resized
                X_test[test_count,  :] = melgram
                Y_test[test_count, :] = this_Y
                # X_test = np.concatenate((X_test, melgram), axis=0)
                # Y_test = np.concatenate((Y_test, this_Y), axis=0)
                paths_test.append(audio_path)
                test_count += 1
        print("")

    print("Shuffling order of data...")
    X_train, Y_train, paths_train = shuffle_XY_paths(X_train, Y_train, paths_train)
    X_test, Y_test, paths_test = shuffle_XY_paths(X_test, Y_test, paths_test)

    return X_train, Y_train, paths_train, X_test, Y_test, paths_test, class_names, sr


def build_model(X, nb_classes):
    input_shape = (X.shape[1], X.shape[2])
    model = Sequential()

    # returns a sequence of vectors of dimension 256
    model.add(LSTM(256, return_sequences=True, input_shape=input_shape))

    model.add(Dropout(0.2))

    # return a single vector of dimension 128
    model.add(LSTM(128))

    model.add(Dropout(0.2))

    # apply softmax to output
    model.add(Dense(nb_classes, activation='softmax'))
    return  model


if __name__ == '__main__':
    np.random.seed(1)

    # get the data
    x_train, y_train, paths_train, x_test, y_test, paths_test, class_names, sr = build_datasets(preproc=True)
    model = build_model(x_train, len(class_names))


    model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])

    # callback koji snima tezine modela
    mc = ModelCheckpoint('weights.h5', monitor='val_loss', verbose=1, save_best_only=True)
    # callback koji prekida obucavanje u slucaju overfitting-a
    es = EarlyStopping(monitor='val_loss', patience=10)

    batch_size = 32
    nb_epoch = 50

    model.fit(x_train, y_train, batch_size=batch_size, epochs=nb_epoch,
               verbose=0, validation_data=(x_test, y_test), callbacks=[mc,es])

    score = model.evaluate(x_test, y_test, verbose=0)
    print('Test score:', score[0])
    print('Test accuracy:', score[1])
