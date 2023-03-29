from tensorflow import keras
import numpy as np
from keras.datasets import fashion_mnist
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix
import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt
import visualkeras
from PIL import ImageFont
from keras import backend as K


# method used for fitting the model and for plotting the loss and accuracy curves
# remove validation split entirely when testing the final model, since we then want to train on the entire training set
def fit_model(model, X_train, Y_train):
    history = model.fit(X_train, Y_train, batch_size=128,
                        epochs=12, verbose=1, validation_split=0.2)

    # this code is used to display the ouputs of different layers of the network.
    """
    img = X_train[0].reshape(1, 28, 28, 1)
    plt.imshow(img[0, :, :, 0], cmap='gray')
    plt.show()
    
    for i in range(len(model.layers)):
        get_layer_output = K.function([model.layers[0].input], [model.layers[i].output])

        layer_output = get_layer_output([img])[0]

        if len(layer_output.shape) > 2:
            layer_output = np.sum(layer_output[0, :, :, :], axis=2)

        plt.imshow(layer_output, cmap='gray')
        plt.show()
    """
    # saving the model weights.
    model.save_weights('val_model_weights.h5')

    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.xlim(0, 12)
    plt.ylim(0, 1)
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show()

    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.xlim(0, 12)
    plt.ylim(0, 1)
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show()


# method used to run the different variants of the model
def conv():
    # load data set and preprocess data
    (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
    print(x_train.shape)
    print(x_test.shape)
    X_train = x_train.reshape(60000, 28, 28, 1)
    X_test = x_test.reshape(10000, 28, 28, 1)
    print(X_train.shape)
    print(X_test.shape)
    X_train = np.divide(X_train, 255)
    X_test = np.divide(X_test, 255)
    Y_train = keras.utils.to_categorical(y_train, 10)
    Y_test = keras.utils.to_categorical(y_test, 10)
    # change model name here to test different models
    model = sigmoid_conv()
    # model weights can be loaded with the line commented out below.
    #model.load_weights('val_model_weights.h5')
    fit_model(model, X_train, Y_train)
    loss, accuracy = model.evaluate(X_test, Y_test, verbose=0)
    print(loss, accuracy)
    # method for creating confusion matrix, only run for top performing model.
    #create_confusion_matrix(model, X_test, Y_test)


# our baseline CNN
def baseline_conv():
    model = keras.Sequential()
    model.add(keras.layers.Conv2D(filters=32, kernel_size=(3, 3), activation="relu", input_shape=(28, 28, 1)))
    model.add(keras.layers.Conv2D(filters=64, kernel_size=(3, 3),
                                  activation="relu"))
    model.add(keras.layers.MaxPool2D(pool_size=(2, 2)))
    model.add(keras.layers.Dropout(rate=0.25))
    model.add(keras.layers.Conv2D(filters=64, kernel_size=(3, 3),
                                  activation="relu"))
    model.add(keras.layers.MaxPool2D(pool_size=(3, 3)))
    model.add(keras.layers.Dropout(rate=0.25))
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(128, activation="relu"))
    model.add(keras.layers.Dense(10, activation="softmax"))
    model.summary()
    # font = ImageFont.load_default() # using comic sans is strictly prohibited!
    # visualkeras.layered_view(model, legend=True, font=font).show()
    model.compile(loss='categorical_crossentropy',
                  optimizer=keras.optimizers.Adadelta(learning_rate=1), metrics='accuracy')
    return model


# variant in which we changed the kernel_size of each convolutional layer to 5,5
def larger_kernel_size():
    model = keras.Sequential()
    model.add(keras.layers.Conv2D(filters=32, kernel_size=(5, 5), activation="relu", input_shape=(28, 28, 1)))
    model.add(keras.layers.Conv2D(filters=64, kernel_size=(5, 5),
                                  activation="relu"))
    model.add(keras.layers.MaxPool2D(pool_size=(2, 2)))
    model.add(keras.layers.Dropout(rate=0.25))
    model.add(keras.layers.Conv2D(filters=64, kernel_size=(5, 5),
                                  activation="relu"))
    model.add(keras.layers.MaxPool2D(pool_size=(2, 2)))
    model.add(keras.layers.Dropout(rate=0.25))
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(128, activation="relu"))
    model.add(keras.layers.Dense(10, activation="softmax"))
    model.summary()
    model.compile(loss='categorical_crossentropy',
                  optimizer=keras.optimizers.Adadelta(learning_rate=0.1), metrics='accuracy')
    return model


# in this variant, we added a convolutional layer
def added_conv_layer():
    model = keras.Sequential()
    model.add(keras.layers.Conv2D(filters=32, kernel_size=(3, 3), activation="relu", input_shape=(28, 28, 1)))
    model.add(keras.layers.Conv2D(filters=64, kernel_size=(3, 3),
                                  activation="relu"))
    model.add(keras.layers.MaxPool2D(pool_size=(2, 2)))
    model.add(keras.layers.Dropout(rate=0.25))
    model.add(keras.layers.Conv2D(filters=64, kernel_size=(3, 3),
                                  activation="relu"))
    model.add(keras.layers.Conv2D(filters=64, kernel_size=(3, 3),
                                  activation="relu"))
    model.add(keras.layers.MaxPool2D(pool_size=(3, 3)))
    model.add(keras.layers.Dropout(rate=0.25))
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(128, activation="relu"))
    model.add(keras.layers.Dense(10, activation="softmax"))
    model.summary()
    model.compile(loss='categorical_crossentropy',
                  optimizer=keras.optimizers.Adadelta(learning_rate=1), metrics='accuracy')
    return model


# here we added a dropout layer between the hidden layer and output layer of the network.
def added_dropout_conv():
    model = keras.Sequential()
    model.add(keras.layers.Conv2D(filters=32, kernel_size=(3, 3), activation="relu", input_shape=(28, 28, 1)))
    model.add(keras.layers.Conv2D(filters=64, kernel_size=(3, 3),
                                  activation="relu"))
    model.add(keras.layers.MaxPool2D(pool_size=(2, 2)))
    model.add(keras.layers.Dropout(rate=0.25))
    model.add(keras.layers.Conv2D(filters=64, kernel_size=(3, 3),
                                  activation="relu"))
    model.add(keras.layers.MaxPool2D(pool_size=(3, 3)))
    model.add(keras.layers.Dropout(rate=0.25))
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(128, activation="relu"))
    model.add(keras.layers.Dropout(rate=0.25))
    model.add(keras.layers.Dense(10, activation="softmax"))
    model.summary()
    model.compile(loss='categorical_crossentropy',
                  optimizer=keras.optimizers.Adadelta(learning_rate=1), metrics='accuracy')
    return model


# here we changed all of the convolutional layers' activation functions from RELU to Simgoid.
def sigmoid_conv():
    model = keras.Sequential()
    model.add(keras.layers.Conv2D(filters=32, kernel_size=(3, 3), activation="sigmoid", input_shape=(28, 28, 1)))
    model.add(keras.layers.Conv2D(filters=64, kernel_size=(3, 3),
                                  activation="sigmoid"))
    model.add(keras.layers.MaxPool2D(pool_size=(2, 2)))
    model.add(keras.layers.Dropout(rate=0.25))
    model.add(keras.layers.Conv2D(filters=64, kernel_size=(3, 3),
                                  activation="sigmoid"))
    model.add(keras.layers.MaxPool2D(pool_size=(3, 3)))
    model.add(keras.layers.Dropout(rate=0.25))
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(128, activation="sigmoid"))
    model.add(keras.layers.Dense(10, activation="softmax"))
    model.summary()
    model.compile(loss='categorical_crossentropy',
                  optimizer=keras.optimizers.Adadelta(learning_rate=1), metrics='accuracy')
    return model


# method for creating and properly displaying a confusion matrix.
def create_confusion_matrix(model, X_test, Y_test):
    Y_pred = model.predict(X_test)
    Y_pred = np.argmax(Y_pred, axis=1)
    Y_test = np.argmax(Y_test, axis=1)
    cm = confusion_matrix(Y_test, Y_pred)
    print(cm)
    classes = ["T=shirt", "Trouser", "Pullover", "Dress", "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Ankle Boot"]
    df_cm = pd.DataFrame(cm, index=[i for i in classes],
                         columns=[i for i in classes])
    plt.figure(figsize=(10, 7))
    sn.heatmap(df_cm, fmt=".0f", cmap="crest", annot=True)
    plt.show()


# Our method for creating KFolds in order to perform cross validation
# We use 5 folds, to mitigate overly long learning times
# We also use manual hyperparameter tuning in order to select the best learning rate
def Kfold():
    (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
    print(x_train.shape)
    print(x_test.shape)
    X_train = x_train.reshape(60000, 28, 28, 1)
    X_test = x_test.reshape(10000, 28, 28, 1)
    print(X_train.shape)
    print(X_test.shape)
    X_train = np.divide(X_train, 255)
    X_test = np.divide(X_test, 255)
    Y_train = keras.utils.to_categorical(y_train, 10)
    Y_test = keras.utils.to_categorical(y_test, 10)
    kFold = KFold(n_splits=5, shuffle=True)
    acc_per_fold = np.zeros(5)
    loss_per_fold = np.zeros(5)
    idx = 0
    # only use variable learning rates for hyperparameter tuning, for the two best models at the end
    learning_rates = [1, 0.1, 0.01, 0.001, 0.05]
    for ((train, validation), learning_rate) in zip(kFold.split(X_train, Y_train), learning_rates):
    #for train, validation in kFold.split(X_train, Y_train):
        model = baseline_conv()
        model.fit(X_train[train], Y_train[train], batch_size=128,
                  epochs=12, verbose=1)
        # print(learning_rate)
        loss_per_fold[idx], acc_per_fold[idx] = model.evaluate(X_train[validation], Y_train[validation])
        # loss, accuracy = model.evaluate(X[test], Y[test], verbose=0)
        idx += 1
        model.save_weights(f'model_weights_fold{idx}.h5')

    print(f"losses per fold{loss_per_fold}, accuracies per fold {acc_per_fold}")
    print(f" mean accuracy {acc_per_fold.mean()}")

#conv()
Kfold()
