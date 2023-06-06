from typing import Tuple
import tensorflow as tf
import tensorflow.keras.layers as layers
from tensorflow.keras import Model
from tensorflow.keras.layers import (Activation, BatchNormalization, Conv3D,
                          Conv3DTranspose, ConvLSTM2D, Dropout,
                          Input, MaxPool3D)

def unet_convlstm_reg(input_shape: Tuple[int] = (12, 256, 256, 4),
         num_filters_base: int = 8,
         dropout_rate: float = 0.2):
    
    input_shape = (12,256,256,4)
    inputs = Input(shape=input_shape)
    x_init = BatchNormalization()(inputs)  # Try with normalizing the dataset
    #x0 = ZeroPadding3D(padding=(0, 0, 2))(x_init)
    x0 = x_init
        
    x_conv1_b1 = ConvLSTM2D(filters= num_filters_base, kernel_size=(3,3), padding='same', return_sequences=True)(x0)
    x_conv2_b1 = ConvLSTM2D(filters=num_filters_base, kernel_size=(3,3), padding='same', return_sequences=True)(x_conv1_b1)
    x_max_b1 = MaxPool3D([1, 2, 2],padding='same')(x_conv2_b1)
    x_bn_b1  = BatchNormalization()(x_max_b1)
    x_do_b1  = Dropout(dropout_rate)(x_bn_b1)


    x_conv1_b2 = ConvLSTM2D(filters=2*num_filters_base, kernel_size=(3,3), padding='same', return_sequences=True)(x_do_b1)
    x_conv2_b2 = ConvLSTM2D(filters=2*num_filters_base, kernel_size=(3,3), padding='same', return_sequences=True)(x_conv1_b2)
    x_max_b2 = MaxPool3D([1, 2, 2],padding='same')(x_conv2_b2)
    x_bn_b2 = BatchNormalization()(x_max_b2)
    x_do_b2 = Dropout(dropout_rate)(x_bn_b2)


    x_conv1_b3 = ConvLSTM2D(filters=4*num_filters_base, kernel_size=(3,3), padding='same', return_sequences=True)(x_do_b2)
    x_conv2_b3 = ConvLSTM2D(filters=4*num_filters_base, kernel_size=(3,3), padding='same', return_sequences=True)(x_conv1_b3)
    x_max_b3 = MaxPool3D([1, 2, 2],padding='same')(x_conv2_b3)
    x_bn_b3 = BatchNormalization()(x_max_b3)
    x_do_b3 = Dropout(dropout_rate)(x_bn_b3)

    x_conv1_b4 =  ConvLSTM2D(filters=8*num_filters_base, kernel_size=(3,3), padding='same', return_sequences=True)(x_do_b3)
    x_conv2_b4 =  ConvLSTM2D(filters=8*num_filters_base, kernel_size=(3,3), padding='same', return_sequences=True)(x_conv1_b4)
    x_max_b4 = MaxPool3D([1, 2, 2],padding='same')(x_conv2_b4)
    x_bn_b4 = BatchNormalization()(x_max_b4)
    x_do_b4 = Dropout(dropout_rate)(x_bn_b4)


    # ------- Head Residual Output (Residual Decoder)

    x_conv1_b5 = ConvLSTM2D(filters=8*num_filters_base, kernel_size=(3,3), padding='same', return_sequences=True)(x_do_b4)
    x_conv2_b5 = ConvLSTM2D(filters=8*num_filters_base, kernel_size=(3,3), padding='same', return_sequences=True)(x_conv1_b5)
    x_deconv_b5 = Conv3DTranspose(filters=8*num_filters_base, kernel_size=(1, 2, 2),strides=(1,2,2),padding='same', activation="relu")(x_conv2_b5)
    x_bn_b5 = BatchNormalization()(x_deconv_b5)
    x_do_b5 = Dropout(dropout_rate)(x_bn_b5)

    cropped_x_conv2_b4 = layers.Cropping3D(cropping=((11,0),(0,0),(0,0)),data_format="channels_last")(x_conv2_b4)
    cropped_x_conv2_b4 = layers.concatenate([cropped_x_conv2_b4]*12,axis=1)
    x_lstm_b6 = ConvLSTM2D(filters=4*num_filters_base, kernel_size=(3,3), padding='same', return_sequences=True)(layers.concatenate([cropped_x_conv2_b4,x_do_b5]))
    x_conv1_b6 = Conv3D(filters=4*num_filters_base, kernel_size=(2,1,1), activation="relu")(x_lstm_b6)
    x_conv2_b6 = Conv3D(filters=4*num_filters_base, kernel_size=(1, 1, 1),padding='same', activation="relu")(x_conv1_b6)
    x_deconv_b6 = Conv3DTranspose(filters=4*num_filters_base,  kernel_size=(1, 2, 2),strides = (1,2,2),padding='same', activation="relu")(x_conv2_b6)
    x_bn_b6 = BatchNormalization()(x_deconv_b6)
    x_do_b6 = Dropout(dropout_rate)(x_bn_b6)

    cropped_x_conv2_b3 = layers.Cropping3D(cropping=((11,0),(0,0),(0,0)),data_format="channels_last")(x_conv2_b3)
    cropped_x_conv2_b3 = layers.concatenate([cropped_x_conv2_b3]*11,axis=1)
    x_lstm_b7 = ConvLSTM2D(filters=2*num_filters_base, kernel_size=(3,3), padding='same', return_sequences=True)(layers.concatenate([cropped_x_conv2_b3,x_do_b6]))
    x_conv1_b7 =Conv3D(filters=2*num_filters_base, kernel_size=(2,1,1), activation="relu")(x_lstm_b7)
    x_conv2_b7 = Conv3D(filters=2*num_filters_base, kernel_size=(1, 1, 1),padding='same', activation="relu")(x_conv1_b7)
    x_deconv_b7 =  Conv3DTranspose(filters=2*num_filters_base,  kernel_size=(1, 2, 2),strides = (1,2,2),padding='same', activation="relu")(x_conv2_b7)
    x_bn_b7 = BatchNormalization()(x_deconv_b7)
    x_do_b7 = Dropout(dropout_rate)(x_bn_b7)


    cropped_x_conv2_b2 =  layers.Cropping3D(cropping=((11,0),(0,0),(0,0)),data_format="channels_last")(x_conv2_b2)
    cropped_x_conv2_b2 = layers.concatenate([cropped_x_conv2_b2]*10,axis=1)
    x_lstm_b8 = ConvLSTM2D(filters=1*num_filters_base, kernel_size=(3,3), padding='same', return_sequences=True)(layers.concatenate([cropped_x_conv2_b2,x_do_b7]))
    x_conv1_b8 = Conv3D(filters=1*num_filters_base, kernel_size=(2,1,1), activation="relu")(x_lstm_b8)
    x_conv2_b8 = Conv3D(filters=1*num_filters_base, kernel_size=(1, 1, 1),padding='same', activation="relu")(x_conv1_b8)
    x_deconv_b8 = Conv3DTranspose(filters=1*num_filters_base,  kernel_size=(1, 2, 2),strides = (1,2,2),padding='same', activation="relu")(x_conv2_b8)
    x_bn_b8 = BatchNormalization()(x_deconv_b8)
    x_do_b8 = Dropout(dropout_rate)(x_bn_b8)


    cropped_x_conv2_b1 = layers.Cropping3D(cropping=((11,0),(0,0),(0,0)),data_format="channels_last")(x_conv2_b1)
    cropped_x_conv2_b1 = layers.concatenate([cropped_x_conv2_b1]*9,axis=1)
    x_lstm_b9 = ConvLSTM2D(filters=8*num_filters_base, kernel_size=(3,3), padding='same', return_sequences=True)(layers.concatenate([cropped_x_conv2_b1,x_do_b8]))
    x_conv1_b9 =Conv3D(filters=0.5*num_filters_base, kernel_size=(2,1,1), activation="relu")(x_lstm_b9)
    x_conv2_b9 = Conv3D(filters=0.5*num_filters_base, kernel_size=(1, 1, 1),padding='same', activation="relu")(x_conv1_b9)
    x_bn_b9 = BatchNormalization()(x_conv2_b9)
    x_do_b9 = Dropout(dropout_rate)(x_bn_b9)


    residual_output = Conv3DTranspose(1, kernel_size=(1, 1, 1), padding="same")(x_do_b9)
    output = Activation("linear", dtype="float32")(residual_output)
    #output = Cropping3D(cropping=(0, 0, 2))(output)

    output = tf.squeeze(output, axis=4)

    model=Model(inputs, output)
    return model

def unet_convlstm_class(input_shape: Tuple[int] = (12, 256, 256, 4),
         num_filters_base: int = 8,
         dropout_rate: float = 0.2,
         bins_num: int = 9):
    
    input_shape = (12,256,256,4)
    inputs = Input(shape=input_shape)
    x_init = BatchNormalization()(inputs)  # Try with normalizing the dataset
    #x0 = ZeroPadding3D(padding=(0, 0, 2))(x_init)
    x0 = x_init
        
    x_conv1_b1 = ConvLSTM2D(filters= num_filters_base, kernel_size=(3,3), padding='same', return_sequences=True)(x0)
    x_conv2_b1 = ConvLSTM2D(filters=num_filters_base, kernel_size=(3,3), padding='same', return_sequences=True)(x_conv1_b1)
    x_max_b1 = MaxPool3D([1, 2, 2],padding='same')(x_conv2_b1)
    x_bn_b1 = BatchNormalization()(x_max_b1)
    x_do_b1 = Dropout(dropout_rate)(x_bn_b1)


    x_conv1_b2 = ConvLSTM2D(filters=2*num_filters_base, kernel_size=(3,3), padding='same', return_sequences=True)(x_do_b1)
    x_conv2_b2 = ConvLSTM2D(filters=2*num_filters_base, kernel_size=(3,3), padding='same', return_sequences=True)(x_conv1_b2)
    x_max_b2 = MaxPool3D([1, 2, 2],padding='same')(x_conv2_b2)
    x_bn_b2 = BatchNormalization()(x_max_b2)
    x_do_b2 = Dropout(dropout_rate)(x_bn_b2)


    x_conv1_b3 = ConvLSTM2D(filters=4*num_filters_base, kernel_size=(3,3), padding='same', return_sequences=True)(x_do_b2)
    x_conv2_b3 = ConvLSTM2D(filters=4*num_filters_base, kernel_size=(3,3), padding='same', return_sequences=True)(x_conv1_b3)
    x_max_b3 = MaxPool3D([1, 2, 2],padding='same')(x_conv2_b3)
    x_bn_b3 = BatchNormalization()(x_max_b3)
    x_do_b3 = Dropout(dropout_rate)(x_bn_b3)

    x_conv1_b4 =  ConvLSTM2D(filters=8*num_filters_base, kernel_size=(3,3), padding='same', return_sequences=True)(x_do_b3)
    x_conv2_b4 =  ConvLSTM2D(filters=8*num_filters_base, kernel_size=(3,3), padding='same', return_sequences=True)(x_conv1_b4)
    x_max_b4 = MaxPool3D([1, 2, 2],padding='same')(x_conv2_b4)
    x_bn_b4 = BatchNormalization()(x_max_b4)
    x_do_b4 = Dropout(dropout_rate)(x_bn_b4)


    # ------- Head Residual Output (Residual Decoder)

    x_conv1_b5 = ConvLSTM2D(filters=8*num_filters_base, kernel_size=(3,3), padding='same', return_sequences=True)(x_do_b4)
    x_conv2_b5 = ConvLSTM2D(filters=8*num_filters_base, kernel_size=(3,3), padding='same', return_sequences=True)(x_conv1_b5)
    x_deconv_b5 = Conv3DTranspose(filters=8*num_filters_base, kernel_size=(1, 2, 2),strides=(1,2,2),padding='same', activation="relu")(x_conv2_b5)
    x_bn_b5 = BatchNormalization()(x_deconv_b5)
    x_do_b5 = Dropout(dropout_rate)(x_bn_b5)

    cropped_x_conv2_b4 = layers.Cropping3D(cropping=((11,0),(0,0),(0,0)),data_format="channels_last")(x_conv2_b4)
    cropped_x_conv2_b4 = layers.concatenate([cropped_x_conv2_b4]*12,axis=1)
    x_lstm_b6 = ConvLSTM2D(filters=4*num_filters_base, kernel_size=(3,3), padding='same', return_sequences=True)(layers.concatenate([cropped_x_conv2_b4,x_do_b5]))
    x_conv1_b6 = Conv3D(filters=4*num_filters_base, kernel_size=(2,1,1), activation="relu")(x_lstm_b6)
    x_conv2_b6 = Conv3D(filters=4*num_filters_base, kernel_size=(1, 1, 1),padding='same', activation="relu")(x_conv1_b6)
    x_deconv_b6 = Conv3DTranspose(filters=4*num_filters_base,  kernel_size=(1, 2, 2),strides = (1,2,2),padding='same', activation="relu")(x_conv2_b6)
    x_bn_b6 = BatchNormalization()(x_deconv_b6)
    x_do_b6 = Dropout(dropout_rate)(x_bn_b6)

    cropped_x_conv2_b3 = layers.Cropping3D(cropping=((11,0),(0,0),(0,0)),data_format="channels_last")(x_conv2_b3)
    cropped_x_conv2_b3 = layers.concatenate([cropped_x_conv2_b3]*11,axis=1)
    x_lstm_b7 = ConvLSTM2D(filters=2*num_filters_base, kernel_size=(3,3), padding='same', return_sequences=True)(layers.concatenate([cropped_x_conv2_b3,x_do_b6]))
    x_conv1_b7 =Conv3D(filters=2*num_filters_base, kernel_size=(2,1,1), activation="relu")(x_lstm_b7)
    x_conv2_b7 = Conv3D(filters=2*num_filters_base, kernel_size=(1, 1, 1),padding='same', activation="relu")(x_conv1_b7)
    x_deconv_b7 =  Conv3DTranspose(filters=2*num_filters_base,  kernel_size=(1, 2, 2),strides = (1,2,2),padding='same', activation="relu")(x_conv2_b7)
    x_bn_b7 = BatchNormalization()(x_deconv_b7)
    x_do_b7 = Dropout(dropout_rate)(x_bn_b7)


    cropped_x_conv2_b2 =  layers.Cropping3D(cropping=((11,0),(0,0),(0,0)),data_format="channels_last")(x_conv2_b2)
    cropped_x_conv2_b2 = layers.concatenate([cropped_x_conv2_b2]*10,axis=1)
    x_lstm_b8 = ConvLSTM2D(filters=1*num_filters_base, kernel_size=(3,3), padding='same', return_sequences=True)(layers.concatenate([cropped_x_conv2_b2,x_do_b7]))
    x_conv1_b8 = Conv3D(filters=1*num_filters_base, kernel_size=(2,1,1), activation="relu")(x_lstm_b8)
    x_conv2_b8 = Conv3D(filters=1*num_filters_base, kernel_size=(1, 1, 1),padding='same', activation="relu")(x_conv1_b8)
    x_deconv_b8 = Conv3DTranspose(filters=1*num_filters_base,  kernel_size=(1, 2, 2),strides = (1,2,2),padding='same', activation="relu")(x_conv2_b8)
    x_bn_b8 = BatchNormalization()(x_deconv_b8)
    x_do_b8 = Dropout(dropout_rate)(x_bn_b8)


    cropped_x_conv2_b1 = layers.Cropping3D(cropping=((11,0),(0,0),(0,0)),data_format="channels_last")(x_conv2_b1)
    cropped_x_conv2_b1 = layers.concatenate([cropped_x_conv2_b1]*9,axis=1)
    x_lstm_b9 = ConvLSTM2D(filters=8*num_filters_base, kernel_size=(3,3), padding='same', return_sequences=True)(layers.concatenate([cropped_x_conv2_b1,x_do_b8]))
    x_conv1_b9 =Conv3D(filters=0.5*num_filters_base, kernel_size=(2,1,1), activation="relu")(x_lstm_b9)
    x_conv2_b9 = Conv3D(filters=0.5*num_filters_base, kernel_size=(1, 1, 1),padding='same', activation="relu")(x_conv1_b9)
    x_bn_b9 = BatchNormalization()(x_conv2_b9)
    x_do_b9 = Dropout(dropout_rate)(x_bn_b9)


    residual_output = Conv3DTranspose(bins_num+1, kernel_size=(1, 1, 1), padding="same")(x_do_b9)
    output = Activation("softmax", dtype="float32")(residual_output)
    #output = Cropping3D(cropping=(0, 0, 2))(output)

    #output = tf.squeeze(output, axis=4)

    model=Model(inputs, output)
    return model

