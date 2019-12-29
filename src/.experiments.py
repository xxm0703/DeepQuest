from keras.layers import Conv2D, MaxPooling2D, Flatten
from keras.layers import Dropout
from keras.models import Sequential

input_shape = [3000, 1]
n_classes = 5


def cnn3dilated(input_shape):
    model = Sequential(name='cnn3adam')
    model.add(Conv2D(kernel_size=5, filters=32, input_shape=input_shape, activation='relu'))
    model.add(MaxPooling2D(pool_size=2))
    print(model.output_shape)

    model.add(Conv2D(kernel_size=3, filters=32, activation='relu'))
    model.add(MaxPooling2D(pool_size=2))
    print(model.output_shape)

    model.add(Dropout(0.2))
    model.add(Flatten())
    print(model.output_shape)



cnn3dilated([256, 256, 1])
