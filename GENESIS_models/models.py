from typing import Tuple
import tensorflow as tf
import tensorflow.keras.layers as layers
from tensorflow.keras import Model
from tensorflow.keras.layers import (Activation, BatchNormalization, Conv3D,
                          Conv3DTranspose, ConvLSTM2D, Dropout, Input, MaxPool3D)

def convlstm_recurrent(tensor, num_filters_base, seq_len):
    # Initial states
    state_h = tf.zeros((tf.shape(tensor)[0], tf.shape(tensor)[2], tf.shape(tensor)[3], num_filters_base))
    state_c = tf.zeros((tf.shape(tensor)[0], tf.shape(tensor)[2], tf.shape(tensor)[3], num_filters_base))
    input = tf.zeros((tf.shape(tensor)[0],1,tf.shape(tensor)[2],tf.shape(tensor)[3],num_filters_base))
    
    # ConvLSTM layer
    convlstm_layer = ConvLSTM2D(filters=num_filters_base, kernel_size=(3, 3), padding='same', return_sequences=True, return_state=True)
    out_fl = []
    
    # Loop over time steps
    for step in range(seq_len):
        # Apply ConvLSTM layer
        output_fl, state_h, state_c = convlstm_layer(input + tensor, initial_state=[state_h, state_c])
        out_fl.append(tf.squeeze(output_fl,axis = 1))
        input = output_fl


    # Reshape the output
    out_fl = tf.convert_to_tensor(out_fl)
    out_fl = tf.reshape(out_fl,(tf.shape(out_fl)[1],tf.shape(out_fl)[0],tf.shape(out_fl)[2],tf.shape(out_fl)[3],tf.shape(out_fl)[4]))
    return out_fl


def unet_convlstm_reg(input_shape: Tuple[int] = (12, 256, 256, 4),
         num_filters_base: int = 4,
         dropout_rate: float = 0.2,
         seq_len: int = 8):
    
    input_shape = (12,256,256,4)
    inputs = Input(shape=input_shape)
    x_init = BatchNormalization()(inputs)  # Try with normalizing the dataset
    x0 = x_init

    x_conv1_b1 = ConvLSTM2D(filters= num_filters_base, kernel_size=(3,3), padding='same', return_sequences = 'True')(x0)
    contex_b1 =  layers.Cropping3D(cropping=((11,0),(0,0),(0,0)),data_format="channels_last")(x_conv1_b1)
    out_fl_b1 = convlstm_recurrent(contex_b1,num_filters_base,seq_len)
    x_max_b1 = MaxPool3D([1, 2, 2],padding='same')(x_conv1_b1)
    x_bn_b1 = BatchNormalization()(x_max_b1)
    x_do_b1 = Dropout(dropout_rate)(x_bn_b1)

    x_conv1_b2 = ConvLSTM2D(filters=2*num_filters_base, kernel_size=(3,3), padding='same', return_sequences = 'True')(x_do_b1)
    contex_b2 =  layers.Cropping3D(cropping=((11,0),(0,0),(0,0)),data_format="channels_last")(x_conv1_b2)
    out_fl_b2 = convlstm_recurrent(contex_b2,2*num_filters_base,seq_len)
    x_max_b2 = MaxPool3D([1, 2, 2],padding='same')(x_conv1_b2)
    x_bn_b2 = BatchNormalization()(x_max_b2)
    x_do_b2 = Dropout(dropout_rate)(x_bn_b2)


    x_conv1_b3 = ConvLSTM2D(filters=4*num_filters_base, kernel_size=(3,3), padding='same', return_sequences = 'True')(x_do_b2)
    contex_b3 =  layers.Cropping3D(cropping=((11,0),(0,0),(0,0)),data_format="channels_last")(x_conv1_b3)
    out_fl_b3  = convlstm_recurrent(contex_b3,4*num_filters_base,seq_len)
    x_max_b3 = MaxPool3D([1, 2, 2],padding='same')(x_conv1_b3)
    x_bn_b3 = BatchNormalization()(x_max_b3)
    x_do_b3 = Dropout(dropout_rate)(x_bn_b3)

    x_conv1_b4_ =   ConvLSTM2D(filters=8*num_filters_base, kernel_size=(3,3), padding='same', return_sequences = 'True')(x_do_b3)
    contex_b4 =  layers.Cropping3D(cropping=((11,0),(0,0),(0,0)),data_format="channels_last")(x_conv1_b4_)
    out_fl_b4  = convlstm_recurrent(contex_b4,8*num_filters_base,seq_len)
    x_max_b4 = MaxPool3D([1, 2, 2],padding='same')(x_conv1_b4_)
    x_bn_b4 = BatchNormalization()(x_max_b4)
    x_do_b4 = Dropout(dropout_rate)(x_bn_b4)

    x_conv1_b5_ =   ConvLSTM2D(filters=8*num_filters_base, kernel_size=(3,3), padding='same')(x_do_b4)
    x_conv2_b5 = tf.expand_dims(x_conv1_b5_, axis = 1)
    out_fl_b5 = convlstm_recurrent(x_conv2_b5,8*num_filters_base,seq_len)
    # ------- Head Residual Output (Residual Decoder)

    #x_conv1_b5 =  Conv3D(filters=8*num_filters_base, kernel_size=(1,3,3), padding='same')(out_fl_b5)
    x_deconv_b5 = Conv3DTranspose(filters=8*num_filters_base, kernel_size=(1, 2, 2),strides=(1,2,2),padding='same', activation="relu")(out_fl_b5)
    x_bn_b5 = BatchNormalization()(x_deconv_b5)
    x_do_b5 = Dropout(dropout_rate)(x_bn_b5)

    x_conv1_b6 =  Conv3D(filters=4*num_filters_base, kernel_size=(1,3,3), padding='same', activation="relu")(layers.concatenate([x_do_b5, out_fl_b4]))
    x_deconv_b6 = Conv3DTranspose(filters=4*num_filters_base,  kernel_size=(1, 2, 2),strides = (1,2,2),padding='same', activation="relu")(x_conv1_b6)
    x_bn_b6 = BatchNormalization()(x_deconv_b6)
    x_do_b6 = Dropout(dropout_rate)(x_bn_b6)


    x_conv1_b7 =  Conv3D(filters=2*num_filters_base, kernel_size=(1,3,3), padding='same', activation="relu")(layers.concatenate([x_do_b6, out_fl_b3]))
    x_deconv_b7 =  Conv3DTranspose(filters=2*num_filters_base,  kernel_size=(1, 2, 2),strides = (1,2,2),padding='same', activation="relu")(x_conv1_b7)
    x_bn_b7 = BatchNormalization()(x_deconv_b7)
    x_do_b7 = Dropout(dropout_rate)(x_bn_b7)

    x_conv1_b8 =  Conv3D(filters=1*num_filters_base, kernel_size=(1,3,3), padding='same', activation="relu")(layers.concatenate([x_do_b7, out_fl_b2]))
    x_deconv_b8 =  Conv3DTranspose(filters=1*num_filters_base,  kernel_size=(1, 2, 2),strides = (1,2,2),padding='same', activation="relu")(x_conv1_b8)
    x_bn_b8 = BatchNormalization()(x_deconv_b8)
    x_do_b8 = Dropout(dropout_rate)(x_bn_b8)

    x_conv1_b9 =  Conv3D(filters=1*num_filters_base, kernel_size=(1,3,3), padding='same', activation="relu")(layers.concatenate([out_fl_b1,x_do_b8]))
    residual_output = Conv3D(1, kernel_size=(1, 1, 1), padding="same")(x_conv1_b9)
    output = Activation("linear", dtype="float32")(residual_output)

    output = tf.squeeze(output, axis=4)
    model=Model(inputs, output)
    return model

def unet_convlstm_class(input_shape: Tuple[int] = (12, 256, 256, 4),
         num_filters_base: int = 8,
         dropout_rate: float = 0.2,
         seq_len : int =  8,
         bins_num: int = 9):
    
    input_shape = (12,256,256,4)
    inputs = Input(shape=input_shape)
    x_init = BatchNormalization()(inputs)  # Try with normalizing the dataset
    x0 = x_init

    x_conv1_b1 = ConvLSTM2D(filters= num_filters_base, kernel_size=(3,3), padding='same', return_sequences = 'True')(x0)
    contex_b1 =  layers.Cropping3D(cropping=((11,0),(0,0),(0,0)),data_format="channels_last")(x_conv1_b1)
    out_fl_b1 = convlstm_recurrent(contex_b1,num_filters_base,seq_len)
    x_max_b1 = MaxPool3D([1, 2, 2],padding='same')(x_conv1_b1)
    x_bn_b1 = BatchNormalization()(x_max_b1)
    x_do_b1 = Dropout(dropout_rate)(x_bn_b1)

    x_conv1_b2 = ConvLSTM2D(filters=2*num_filters_base, kernel_size=(3,3), padding='same', return_sequences = 'True')(x_do_b1)
    contex_b2 =  layers.Cropping3D(cropping=((11,0),(0,0),(0,0)),data_format="channels_last")(x_conv1_b2)
    out_fl_b2 = convlstm_recurrent(contex_b2,2*num_filters_base,seq_len)
    x_max_b2 = MaxPool3D([1, 2, 2],padding='same')(x_conv1_b2)
    x_bn_b2 = BatchNormalization()(x_max_b2)
    x_do_b2 = Dropout(dropout_rate)(x_bn_b2)


    x_conv1_b3 = ConvLSTM2D(filters=4*num_filters_base, kernel_size=(3,3), padding='same', return_sequences = 'True')(x_do_b2)
    contex_b3 =  layers.Cropping3D(cropping=((11,0),(0,0),(0,0)),data_format="channels_last")(x_conv1_b3)
    out_fl_b3  = convlstm_recurrent(contex_b3,4*num_filters_base,seq_len)
    x_max_b3 = MaxPool3D([1, 2, 2],padding='same')(x_conv1_b3)
    x_bn_b3 = BatchNormalization()(x_max_b3)
    x_do_b3 = Dropout(dropout_rate)(x_bn_b3)

    x_conv1_b4_ =   ConvLSTM2D(filters=8*num_filters_base, kernel_size=(3,3), padding='same', return_sequences = 'True')(x_do_b3)
    contex_b4 =  layers.Cropping3D(cropping=((11,0),(0,0),(0,0)),data_format="channels_last")(x_conv1_b4_)
    out_fl_b4  = convlstm_recurrent(contex_b4,8*num_filters_base,seq_len)
    x_max_b4 = MaxPool3D([1, 2, 2],padding='same')(x_conv1_b4_)
    x_bn_b4 = BatchNormalization()(x_max_b4)
    x_do_b4 = Dropout(dropout_rate)(x_bn_b4)

    x_conv1_b5_ =   ConvLSTM2D(filters=8*num_filters_base, kernel_size=(3,3), padding='same')(x_do_b4)
    x_conv2_b5 = tf.expand_dims(x_conv1_b5_, axis = 1)
    out_fl_b5 = convlstm_recurrent(x_conv2_b5,8*num_filters_base,seq_len)
    # ------- Head Residual Output (Residual Decoder)

    #x_conv1_b5 =  Conv3D(filters=8*num_filters_base, kernel_size=(1,3,3), padding='same')(out_fl_b5)
    x_deconv_b5 = Conv3DTranspose(filters=8*num_filters_base, kernel_size=(1, 2, 2),strides=(1,2,2),padding='same', activation="relu")(out_fl_b5)
    x_bn_b5 = BatchNormalization()(x_deconv_b5)
    x_do_b5 = Dropout(dropout_rate)(x_bn_b5)

    x_conv1_b6 =  Conv3D(filters=4*num_filters_base, kernel_size=(1,3,3), padding='same', activation="relu")(layers.concatenate([x_do_b5, out_fl_b4]))
    x_deconv_b6 = Conv3DTranspose(filters=4*num_filters_base,  kernel_size=(1, 2, 2),strides = (1,2,2),padding='same', activation="relu")(x_conv1_b6)
    x_bn_b6 = BatchNormalization()(x_deconv_b6)
    x_do_b6 = Dropout(dropout_rate)(x_bn_b6)


    x_conv1_b7 =  Conv3D(filters=2*num_filters_base, kernel_size=(1,3,3), padding='same', activation="relu")(layers.concatenate([x_do_b6, out_fl_b3]))
    x_deconv_b7 =  Conv3DTranspose(filters=2*num_filters_base,  kernel_size=(1, 2, 2),strides = (1,2,2),padding='same', activation="relu")(x_conv1_b7)
    x_bn_b7 = BatchNormalization()(x_deconv_b7)
    x_do_b7 = Dropout(dropout_rate)(x_bn_b7)

    x_conv1_b8 =  Conv3D(filters=1*num_filters_base, kernel_size=(1,3,3), padding='same', activation="relu")(layers.concatenate([x_do_b7, out_fl_b2]))
    x_deconv_b8 =  Conv3DTranspose(filters=1*num_filters_base,  kernel_size=(1, 2, 2),strides = (1,2,2),padding='same', activation="relu")(x_conv1_b8)
    x_bn_b8 = BatchNormalization()(x_deconv_b8)
    x_do_b8 = Dropout(dropout_rate)(x_bn_b8)

    x_conv1_b9 =  Conv3D(filters=1*num_filters_base, kernel_size=(1,3,3), padding='same', activation="relu")(layers.concatenate([out_fl_b1,x_do_b8]))
    residual_output = Conv3D(bins_num +1, kernel_size=(1, 1, 1), padding="same")(x_conv1_b9)
    output = Activation("softmax", dtype="float32")(residual_output)

    model=Model(inputs, output)
    return model
