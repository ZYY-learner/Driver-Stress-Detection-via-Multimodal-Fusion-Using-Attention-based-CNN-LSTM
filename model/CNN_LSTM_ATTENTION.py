import numpy as np
import os
import datetime
import matplotlib.pyplot as plt
from matplotlib import pyplot

from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

from keras.models import Model
from keras.layers import Input, Lambda
from keras.layers import Dense, Flatten, Activation
from keras.layers import Conv1D, MaxPooling1D, Concatenate, AveragePooling1D
from keras.layers import LSTM, Bidirectional
from keras.callbacks import EarlyStopping
from keras.layers.normalization import BatchNormalization
from keras.layers import Dropout
from keras.layers import ELU
from keras.utils import to_categorical
from keras import backend as K
from attention import Attention_layer, HierarchicalAttentionNetwork
import scipy.io as scio
from sklearn.metrics import roc_curve

from sklearn.model_selection import KFold

path = '/1data/'


def main(save_confusion=False):
    model = define_model()
    print("Loaddata...")
    data1 = scio.loadmat(os.getcwd() + path + 'cont_5s' + ".mat")
    data2 = scio.loadmat(os.getcwd() + path + 'eye_5s' + ".mat")
    data3 = scio.loadmat(os.getcwd() + path + 'phy_5s' + ".mat")
    data4 = scio.loadmat(os.getcwd() + path + 'y_5s' + ".mat")
    x_cont = data1['cont']
    x_eye = data2['eye']  # pupil
    # x_eye = x_eye[..., np.newaxis]
    x_phy = data3['phy']
    y = data4['y']
    stress1 = 0
    stress2 = 0
    stress3 = 0
    # Divide label
    for i in range(y.shape[0]):
        if y[i] < 1:
            y[i] = 1
            stress1 += 1
        elif (y[i] >= 1 and y[i] < 2):
            y[i] = 2
            stress2 += 1
        else:
            y[i] = 3
            stress3 += 1

    X_train_cont1, X_test_cont1, y_train1, y_test1 = train_test_split(x_cont, y, test_size=0.2, random_state=25,
                                                                      stratify=y)
    X_train_eye1, X_test_eye1, y_train1, y_test1 = train_test_split(x_eye, y, test_size=0.2, random_state=25,
                                                                    stratify=y)
    X_train_phy1, X_test_phy1, y_train1, y_test1 = train_test_split(x_phy, y, test_size=0.2, random_state=25,
                                                                    stratify=y)

    kf = KFold(n_splits=10, shuffle=False, random_state=2020)
    acc = []
    for k, (train_index, test_index) in enumerate(kf.split(X_train_eye1)):
        model = define_model()
        print("TRAIN:", train_index, "TEST:", test_index)

        # X_train_cont, X_test_cont = X_train_cont1[train_index], X_train_cont1[test_index]
        # X_train_eye, X_test_eye = X_train_eye1[train_index], X_train_eye1[test_index]
        # X_train_phy, X_test_phy = X_train_phy1[train_index], X_train_phy1[test_index]
        # y_train, y_test = y_train1[train_index], y_train1[test_index]

        x_train = [X_train_cont1, X_train_eye1, X_train_phy1]
        x_test = [X_test_cont1, X_test_eye1, X_test_phy1]
        y_train = to_categorical(y_train1, 4)
        y_train = y_train[:, 1:4]
        y_test = to_categorical(y_test1, 4)
        y_test = y_test[:, 1:4]
        # run model training and evaluation
        es = EarlyStopping(monitor='val_acc', mode='max', patience=10, verbose=1, restore_best_weights=True)
        history = model.fit(x_train, y_train, batch_size=16, epochs=100, verbose=1, validation_split=0.1,
                            shuffle=True,
                            callbacks=[es])
        _, accuracy = model.evaluate(x_test, y_test, batch_size=16, verbose=0)
        # plot history
        pyplot.plot(history.history['acc'], label='train')
        pyplot.plot(history.history['val_acc'], label='val')
        pyplot.legend()
        # pyplot.savefig('./acc_valacc.jpg')
        pyplot.show()
        pyplot.plot(history.history['loss'], label='train')
        pyplot.plot(history.history['val_loss'], label='val')
        pyplot.legend()
        # pyplot.savefig('./acc_valacc.jpg')
        pyplot.show()
        # create test set and targets
        y_prediction = model.predict(x_test)
        # Save the superparameters and structure
        # save_model(model)

        evaluate_model(model, history, accuracy, y_test, y_prediction, save_confusion)

    return None


def define_model():
    # first CNN input model cont

    input1 = Input(shape=(300, 6))
    conv11 = Conv1D(20, 10, strides=1, padding='same')(input1)
    acti11 = Activation('elu')(conv11)
    pool11 = MaxPooling1D(pool_size=2, strides=2)(acti11)
    Drop11 = Dropout(0.3)(pool11)

    conv12 = Conv1D(40, 5, strides=1, padding='same')(Drop11)
    acti12 = Activation('elu')(conv12)
    pool12 = MaxPooling1D(pool_size=2, strides=2)(acti12)
    Drop12 = Dropout(0.3)(pool12)

    conv13 = Conv1D(80, 3, strides=1, padding='same')(Drop12)
    acti13 = Activation('elu')(conv13)
    pool13 = MaxPooling1D(pool_size=2, strides=2)(acti13)
    Drop13 = Dropout(0.3)(pool13)

    lstm_out11 = LSTM(64, return_sequences=True)(Drop13)
    lstm_out12 = LSTM(64, return_sequences=True)(lstm_out11)

    # Drop1 = Dropout(0.3)(lstm_out12)
    # second CNN input model eye
    input2 = Input(shape=(900, 4))
    conv21 = Conv1D(20, 10, strides=1, padding='same')(input2)
    acti21 = Activation('elu')(conv21)
    pool21 = MaxPooling1D(pool_size=2, strides=2)(acti21)
    Drop21 = Dropout(0.15)(pool21)

    conv22 = Conv1D(40, 5, strides=1, padding='same')(Drop21)
    acti22 = Activation('elu')(conv22)
    pool22 = MaxPooling1D(pool_size=2, strides=2)(acti22)
    Drop22 = Dropout(0.15)(pool22)

    conv23 = Conv1D(80, 3, strides=1, padding='same')(Drop22)
    acti23 = Activation('elu')(conv23)
    pool23 = MaxPooling1D(pool_size=2, strides=2)(acti23)
    Drop23 = Dropout(0.15)(pool23)
    '''
    conv24 = Conv1D(160, 3, strides=1, padding='same')(Drop23)
    acti24 = Activation('elu')(conv24)
    pool24 = MaxPooling1D(pool_size=2, strides=2)(acti24)
    Drop24 = Dropout(0.3)(pool24)
    '''
    lstm_out21 = LSTM(64, return_sequences=True)(Drop23)
    lstm_out22 = LSTM(64, return_sequences=True)(lstm_out21)
    # Drop2 = Dropout(0.3)(lstm_out22)
    # th3 CNN input model phy
    input3 = Input(shape=(300, 3))
    conv31 = Conv1D(20, 10, strides=1, padding='same')(input3)
    acti31 = Activation('elu')(conv31)
    pool31 = MaxPooling1D(pool_size=2, strides=2)(acti31)
    Drop31 = Dropout(0.15)(pool31)

    conv32 = Conv1D(40, 5, strides=1, padding='same')(Drop31)
    acti32 = Activation('elu')(conv32)
    pool32 = MaxPooling1D(pool_size=2, strides=2)(acti32)
    Drop32 = Dropout(0.15)(pool32)

    conv33 = Conv1D(80, 3, strides=1, padding='same')(Drop32)
    acti33 = Activation('elu')(conv33)
    pool33 = MaxPooling1D(pool_size=2, strides=2)(acti33)
    Drop33 = Dropout(0.15)(pool33)
    '''
    conv34 = Conv1D(160, 3, strides=1, padding='same')(Drop33)
    acti34 = Activation('elu')(conv34)
    pool34 = MaxPooling1D(pool_size=2, strides=2)(acti34)
    Drop34 = Dropout(0.3)(pool34)
    '''
    lstm_out31 = LSTM(64, return_sequences=True)(Drop33)
    lstm_out32 = LSTM(64, return_sequences=True)(lstm_out31)
    # Drop3 = Dropout(0.3)(lstm_out32)
    # merge input models
    merge = Concatenate(axis=1)([lstm_out12, lstm_out22, lstm_out32])#1
    # merge = Lambda(lambda x:K.expand_dims(x, -1))(merge)

    #atten = HierarchicalAttentionNetwork(256)(merge)
    atten = Attention_layer()(merge)
    # dense1 = Dense(15, activation='relu')(atten)
    # dense2 = Dense(8, activation='relu')(dense1)

    output = Dense(3, activation='softmax')(atten)
    model = Model(inputs=[input1, input2, input3], outputs=output)

    # print(model.summary())

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model


def save_model(model):
    """Saves the learned model architecture and hyperparameters to an HDF5 file which can be loaded with Keras's
    load_model function"""
    now = datetime.datetime.now()
    title = now.strftime("%Y-%m-%d_%H%M")
    model.save('model_' + title)
    print('Model saved to working directory')

    return None


def evaluate_model(model, history, accuracy, y_test, y_prediction, save_confusion):
    """Generates all performance metrics"""
    # print accuracy as a percentage
    percent_accuracy = accuracy * 100.0
    print('Accuracy =', percent_accuracy, "%")

    # print confusion matrix
    matrix = confusion_matrix(y_test.argmax(axis=1), y_prediction.argmax(axis=1))
    print('Confusion Matrix:')
    print(np.matrix(matrix))

    # save confusion matrix
    if save_confusion:
        squeezed_confusion_matrix = np.squeeze(np.asarray(matrix))
        now = datetime.datetime.now()
        title = now.strftime("%Y-%m-%d_%H%M-%S")
        np.savetxt(title + '.csv', squeezed_confusion_matrix, delimiter=',', fmt='%d')
        print("Confusion matrix saved to working directory")

    # print classification report
    target_names = ['1', '2', '3']
    print('Classification Report:')
    print(classification_report(y_test.argmax(axis=1), y_prediction.argmax(axis=1),
                                target_names=target_names, digits=5))
    m = np.matrix(matrix)
    print(m)
    fp1 = (m[1, 0] + m[2, 0]) / ((m[1, 0] + m[2, 0]) + m[1, 1] + m[2, 2] + m[1, 2] + m[2, 1])
    fp2 = (m[0, 1] + m[2, 1]) / ((m[0, 1] + m[2, 1]) + m[0, 0] + m[2, 2] + m[2, 0] + m[0, 2])
    fp3 = (m[0, 2] + m[1, 2]) / ((m[0, 2] + m[1, 2]) + m[0, 0] + m[1, 1] + m[1, 0] + m[0, 1])
    fp = (fp1 + fp2 + fp3) / 3
    print('False positive =', fp * 100.0, "%")
    acc = history.history['acc']
    loss = history.history['loss']

    epochs = range(1, len(acc) + 1)

    plt.plot(epochs, acc)
    plt.title('Training accuracy')
    plt.xlabel('Training Epochs')
    plt.ylabel('Accuracy')
    plt.savefig('./Accuracy.jpg')
    plt.figure()

    plt.plot(epochs, loss)
    plt.title('Training loss')
    plt.xlabel('Training Epochs')
    plt.ylabel('Loss')
    plt.savefig('./Loss.jpg')
    # plt.show()

    print(model.summary())

    return None


# starts main if file called as script (rather than imported)
if __name__ == "__main__":
    main()
