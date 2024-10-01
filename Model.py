import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, concatenate, BatchNormalization, Dropout
from tensorflow.keras.models import Model
from data import x_train,y_train,x_test,y_test

def unetpp(input_size=(256, 256, 1)):
    inputs = Input(input_size)

    c1 = Conv2D(64, (3, 3), activation='relu', padding='same')(inputs)
    c1 = BatchNormalization()(c1)
    c1 = Conv2D(64, (3, 3), activation='relu', padding='same')(c1)
    c1 = BatchNormalization()(c1)
    p1 = MaxPooling2D((2, 2))(c1)

    c2 = Conv2D(128, (3, 3), activation='relu', padding='same')(p1)
    c2 = BatchNormalization()(c2)
    c2 = Conv2D(128, (3, 3), activation='relu', padding='same')(c2)
    c2 = BatchNormalization()(c2)
    p2 = MaxPooling2D((2, 2))(c2)

    c3 = Conv2D(256, (3, 3), activation='relu', padding='same')(p2)
    c3 = BatchNormalization()(c3)
    c3 = Conv2D(256, (3, 3), activation='relu', padding='same')(c3)
    c3 = BatchNormalization()(c3)
    p3 = MaxPooling2D((2, 2))(c3)

    u4 = UpSampling2D((2, 2))(c3)
    u4 = concatenate([u4, c2])
    c4 = Conv2D(128, (3, 3), activation='relu', padding='same')(u4)
    c4 = BatchNormalization()(c4)
    c4 = Conv2D(128, (3, 3), activation='relu', padding='same')(c4)
    c4 = BatchNormalization()(c4)

    u5 = UpSampling2D((2, 2))(c4)
    u5 = concatenate([u5, c1])
    c5 = Conv2D(64, (3, 3), activation='relu', padding='same')(u5)
    c5 = BatchNormalization()(c5)
    c5 = Conv2D(64, (3, 3), activation='relu', padding='same')(c5)
    c5 = BatchNormalization()(c5)

    outputs = Conv2D(1, (1, 1), activation='sigmoid')(c5)

    model = Model(inputs=[inputs], outputs=[outputs])
    return model

unetpp_model = unetpp()
unetpp_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
                     loss='binary_crossentropy', metrics=['accuracy', tf.keras.metrics.MeanIoU(num_classes=2)])

unetpp_history = unetpp_model.fit(x_train, y_train, epochs=25, batch_size=16, validation_data=(x_test, y_test))




def attention_block(x, g, inter_channel):
    theta_x = Conv2D(inter_channel, (1, 1), strides=(2, 2), padding='same')(x)
    phi_g = Conv2D(inter_channel, (1, 1), padding='same')(g)

    concat_xg = tf.keras.layers.add([theta_x, phi_g])
    act_xg = tf.keras.layers.Activation('relu')(concat_xg)

    psi = Conv2D(1, (1, 1), padding='same')(act_xg)
    sigmoid_xg = tf.keras.layers.Activation('sigmoid')(psi)

    shape_sigmoid = tf.keras.backend.int_shape(sigmoid_xg)
    upsample_psi = UpSampling2D(size=(shape_sigmoid[1], shape_sigmoid[2]))(sigmoid_xg)

    return tf.keras.layers.multiply([upsample_psi, x])


def attention_unet(input_size=(256, 256, 1)):
    inputs = Input(input_size)

    # Encoder (Downsampling)
    c1 = Conv2D(64, (3, 3), activation='relu', padding='same')(inputs)
    p1 = MaxPooling2D((2, 2))(c1)

    c2 = Conv2D(128, (3, 3), activation='relu', padding='same')(p1)
    p2 = MaxPooling2D((2, 2))(c2)

    c3 = Conv2D(256, (3, 3), activation='relu', padding='same')(p2)
    p3 = MaxPooling2D((2, 2))(c3)

    # Bridge
    c4 = Conv2D(512, (3, 3), activation='relu', padding='same')(p3)

    # Decoder (Upsampling with Attention Blocks)
    u5 = UpSampling2D((2, 2))(c4)
    u5 = attention_block(c3, u5, 256)
    u5 = concatenate([u5, c3])
    c5 = Conv2D(256, (3, 3), activation='relu', padding='same')(u5)

    u6 = UpSampling2D((2, 2))(c5)
    u6 = attention_block(c2, u6, 128)
    u6 = concatenate([u6, c2])
    c6 = Conv2D(128, (3, 3), activation='relu', padding='same')(u6)

    u7 = UpSampling2D((2, 2))(c6)
    u7 = attention_block(c1, u7, 64)
    u7 = concatenate([u7, c1])
    c7 = Conv2D(64, (3, 3), activation='relu', padding='same')(u7)

    outputs = Conv2D(1, (1, 1), activation='sigmoid')(c7)

    model = Model(inputs=[inputs], outputs=[outputs])
    return model


# Compile the Attention U-Net model
att_unet_model = attention_unet()
att_unet_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
                       loss='binary_crossentropy', metrics=['accuracy', tf.keras.metrics.MeanIoU(num_classes=2)])
# Train the U-Net++ model
unetpp_history = unetpp_model.fit(x_train, y_train, epochs=25, batch_size=16, validation_data=(x_test, y_test))

# Train the Attention U-Net model
att_unet_history = att_unet_model.fit(x_train, y_train, epochs=25, batch_size=16, validation_data=(x_test, y_test))
# Evaluate U-Net++
unetpp_eval = unetpp_model.evaluate(x_test, y_test)
print("U-Net++ Model Evaluation:", unetpp_eval)

# Evaluate Attention U-Net
att_unet_eval = att_unet_model.evaluate(x_test, y_test)
print("Attention U-Net Model Evaluation:", att_unet_eval)

