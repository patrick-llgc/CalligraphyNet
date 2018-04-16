import time
from keras.layers import Conv2D, MaxPooling2D, Dropout, Activation, Flatten, Dense
from keras.layers.normalization import BatchNormalization
from keras.models import Sequential
from data_loader import DataSet
import keras

def build_model():
    """Model builder"""
    model = Sequential()

    model.add(Conv2D(128, (3, 3), input_shape=(64, 64, 1)))
    model.add(BatchNormalization(axis=-1))
    model.add(Activation('relu'))
    model.add(Conv2D(64, (3, 3)))
    model.add(BatchNormalization(axis=-1))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(64, (3, 3)))
    model.add(BatchNormalization(axis=-1))
    model.add(Activation('relu'))
    model.add(Conv2D(64, (3, 3)))
    model.add(BatchNormalization(axis=-1))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())

    # Fully connected layer
    model.add(Dense(1024))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.4))
    model.add(Dense(100))

    model.add(Activation('softmax'))
    return model


def training(X_train, y_train, X_valid, y_valid):
    model = build_model()
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.summary()
    model.fit(X_train, y_train, epochs=1, validation_data=(X_valid, y_valid)) # the more epoch the better
    model.save('model.h5')


def testing(X_test, y_test):
    # load model
    from keras.models import load_model
    model = load_model('model.h5')
    model.compile(optimizer='adam',loss='categorical_crossentropy', metrics=['accuracy'])
    loaded_model_score = model.evaluate(X_test, y_test)
    print('test accuracy: ',loaded_model_score[1]) # the 0-th element is loss, the 1st element is accuracy
    print(model.metrics)


def app(train_or_test):
    if train_or_test == 'train':
        start_time = time.time()
        X_train, y_train = train_data.load_all()
        y_train = keras.utils.to_categorical(y_train, num_classes=num_classes)
        print('load training used time:', time.time() - start_time)
        print (X_train.shape)
        print (y_train.shape)
        training(X_train, y_train)

    if train_or_test == 'test':
        X_test, y_test = chars.test.load_all()
        testing(X_test, y_test)


if __name__ == '__main__':
    train_data = DataSet(r'/Users/megatron/DL/train_preproc/**/*jpg')
    test_data = DataSet(r'/Users/megatron/DL/test/**/*jpg')
    test_data.use_rotation = False
    test_data.use_filter = False
    num_classes = 100
    train_set, valid_set = train_data.train_valid_split()
    X_train, y_train = train_data.load_all(train_set)
    y_train = keras.utils.to_categorical(y_train, num_classes=num_classes)
    X_valid, y_valid = train_data.load_all(valid_set)
    y_valid = keras.utils.to_categorical(y_valid, num_classes=num_classes)
    print(X_valid.shape)
    print(y_valid.shape)
    training(X_train, y_train, X_valid, y_valid)




