from keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Flatten, Input
from keras.models import Model

def spec_grav_model(input_shape):
    ip = Input(shape=input_shape)
    x = Conv2D(32, (3, 3), activation='relu', strides=2)(ip)
    x = Conv2D(32, (3, 3), activation='relu', strides=2)(x)
    x = MaxPooling2D((2, 2))(x)
    x = Conv2D(64, (3, 3), activation='relu')(x)
    x = MaxPooling2D((2, 2))(x)
    x = Flatten()(x)
    x = Dense(128, activation='relu')(x)
    op = Dense(3, activation='softmax')(x)

    model = Model(ip, op)

    return model

# model = spec_grav_model(input_shape=(256, 64, 3))