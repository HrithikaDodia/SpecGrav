from keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Flatten, Input, BatchNormalization
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


def spec_grav_model2(input_shape):
    ip = Input(shape=input_shape)
    x = Conv2D(64, (3, 3), activation='relu')(ip)
    x = Conv2D(64, (3, 3), activation='relu')(x)
    x = MaxPooling2D((2, 2))(x)
    x = Conv2D(128, (3, 3), activation='relu')(x)
    x = Conv2D(128, (3, 3), activation='relu')(x)
    x = MaxPooling2D((2, 2))(x)
    x = Conv2D(256, (3, 3), activation='relu')(x)
    x = Conv2D(256, (3, 3), activation='relu')(x)
    x = MaxPooling2D((2, 2))(x)
    x = Flatten()(x)
    x = Dense(128, activation='relu')(x)
    x = Dense(64, activation='relu')(x)
    op = Dense(3, activation='softmax')(x)

    model = Model(ip, op)

    return model


# model = spec_grav_model2(input_shape=(256, 64, 3))
# print(model.summary())

def spec_grav_model3(input_shape):
    ip = Input(shape=input_shape)
    x = Conv2D(64, (3, 3), activation='relu', strides=(2, 2))(ip)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2, 2))(x)
    x = Conv2D(128, (3, 3), activation='relu')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2, 2))(x)
    x = Conv2D(256, (3, 3), activation='relu')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2, 2))(x)
    x = Flatten()(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(64, activation='relu')(x)
    x = Dropout(0.5)(x)
    op = Dense(3, activation='softmax')(x)

    model = Model(ip, op)

    return model

# model = spec_grav_model3(input_shape=(256, 64, 3))
# print(model.summary())


# The Final Model For Real LIGO Data
def spec_grav_model_newdata(input_shape):
    ip = Input(shape=input_shape)
    x = Conv2D(32, (3, 3), activation='relu')(ip)
    x = MaxPooling2D((2, 2), strides=(2, 2))(x)
    x = Conv2D(64, (3, 3), activation='relu')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2))(x)
    x = Conv2D(128, (3, 3), activation='relu')(x)
    x = MaxPooling2D((2, 2))(x)
    x = Conv2D(128, (3, 3), activation='relu')(x)
    x = MaxPooling2D((2, 2))(x)
    x = Flatten()(x)
    x = Dense(128, activation='relu')(x)
    x = Dense(64, activation='relu')(x)
    op = Dense(3, activation='softmax')(x)

    model = Model(ip, op)

    return model


# model = spec_grav_model_newdata(input_shape=(256, 64, 3))
# print(model.summary())
