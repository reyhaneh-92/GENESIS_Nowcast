import numpy as np
import tensorflow as tf
import random
from typing import List, Tuple
from tensorflow import keras
import tensorflow.keras.layers as layers
from tensorflow.keras import Model
from tensorflow.keras.layers import Input, ConvLSTM2D, BatchNormalization, Activation
from tensorflow.keras.layers import Conv3D, MaxPool3D, Conv3DTranspose
from tensorflow.keras.layers import Dropout
from tensorflow.keras.utils import to_categorical
from keras import backend as K
import os
from scipy import io
#%%
class CustomGenerator_reg(keras.utils.Sequence):
    """
    CustomGenerator Data loader/generator

    Custom data loader/generator used to load inputs from disk into RAM and GPU VRAM during training

    Parameters
    ----------
    keras : keras.utils.Sequence
        Inherited keras Sequence class
    """

    def __init__(self,
                 input_paths: List[str],
                 mu: list(),
                 std: list(),
                 batch_size: int,
                 shuffle: bool = True):
        """
        __init__ Class constructor

        Parameters
        ----------
        input_paths : List[str]
            List of file paths to each input (files should contain a single sample)
        batch_size : int
            Batch size to use when retrieving input
        shuffle : bool, optional
            Option to shuffle input samples, by default True
        """
        self.input_paths = input_paths
        self.batch_size  = batch_size
        self.mu = mu
        self.std = std

        if shuffle:
            random.shuffle(self.input_paths)

    def __len__(self) -> int:
        """
        __len__ Get number of batches based on batch size

        Returns
        -------
        int
            Total number of batches
        """
        return len(self.input_paths) // int(self.batch_size)

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        __getitem__ Get item

        Returns a batch based on index argument

        Parameters
        ----------
        idx : int
            Index of batch to return

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            (Input, label) pair
        """
        batch_x = self.input_paths[idx * self.batch_size:(idx + 1) *
                                   self.batch_size]
        

        X = []
        yy = []
        for i in range(self.batch_size):
            arr = np.load(batch_x[i], allow_pickle=True)
            X.append(arr[0])
            yy.append(arr[1])
         
        X = np.array(X)
        y = np.array(yy)
        
        for i in range(4):
            X[:,:,:,:,i] = (X[:,:,:,:,i]-self.mu[i])/self.std[i]

        return X, y
#%%
class CustomGenerator_class(keras.utils.Sequence):
    """
    CustomGenerator Data loader/generator

    Custom data loader/generator used to load inputs from disk into RAM and GPU VRAM during training

    Parameters
    ----------
    keras : keras.utils.Sequence
        Inherited keras Sequence class
    """

    def __init__(self,
                 input_paths: List[str],
                 mu: list(),
                 std: list(),
                 batch_size: int,
                 bins_num: int,
                 shuffle: bool = True):
        """
        __init__ Class constructor

        Parameters
        ----------
        input_paths : List[str]
            List of file paths to each input (files should contain a single sample)
        batch_size : int
            Batch size to use when retrieving input
        shuffle : bool, optional
            Option to shuffle input samples, by default True
        """
        self.input_paths = input_paths
        self.mu = mu
        self.std = std
        self.batch_size = batch_size
        self.bins_num = bins_num

        if shuffle:
            random.shuffle(self.input_paths)

    def __len__(self) -> int:
        """
        __len__ Get number of batches based on batch size

        Returns
        -------
        int
            Total number of batches
        """
        return len(self.input_paths) // int(self.batch_size)

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        __getitem__ Get item

        Returns a batch based on index argument

        Parameters
        ----------
        idx : int
            Index of batch to return

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            (Input, label) pair
        """
        batch_x = self.input_paths[idx * self.batch_size:(idx + 1) *
                                   self.batch_size]
        X  = []
        yy = []
        for i in range(self.batch_size):
            arr = np.load(batch_x[i], allow_pickle=True)
            X.append(arr[0])
            yy.append(arr[1])
                     
        X = np.array(X) 
        yy = np.array(yy)
                 
        for i in range(4):
            X[:,:,:,:,i] = (X[:,:,:,:,i]-self.mu[i])/self.std[i]            
                     
        bins = pow(10,np.linspace(np.log10(0.10), np.log10(32), self.bins_num))

        yy = np.digitize(yy, bins)   # classes ranges from 0 to bin_num
        y = to_categorical(yy, num_classes = self.bins_num+1)

        return X, y
    
class CustomGenerator_BM(keras.utils.Sequence):
    """
    CustomGenerator Data loader/generator

    Custom data loader/generator used to load inputs from disk into RAM and GPU VRAM during training

    Parameters
    ----------
    keras : keras.utils.Sequence
        Inherited keras Sequence class
    """

    def __init__(self,
                 input_paths: List[str],
                 mu: list(),
                 std: list(),
                 batch_size: int,
                 bins_num: int,
                 shuffle: bool = True):
        """
        __init__ Class constructor

        Parameters
        ----------
        input_paths : List[str]
            List of file paths to each input (files should contain a single sample)
        batch_size : int
            Batch size to use when retrieving input
        shuffle : bool, optional
            Option to shuffle input samples, by default True
        """
        self.input_paths = input_paths
        self.mu = mu
        self.std = std
        self.batch_size = batch_size
        self.bins_num = bins_num

        if shuffle:
            random.shuffle(self.input_paths)

    def __len__(self) -> int:
        """
        __len__ Get number of batches based on batch size

        Returns
        -------
        int
            Total number of batches
        """
        return len(self.input_paths) // int(self.batch_size)

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        __getitem__ Get item

        Returns a batch based on index argument

        Parameters
        ----------
        idx : int
            Index of batch to return

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            (Input, label) pair
        """
        batch_x = self.input_paths[idx * self.batch_size:(idx + 1) *
                                   self.batch_size]
        X  = []
        for i in range(self.batch_size):
            arr = np.load(batch_x[i], allow_pickle=True)
            X.append(arr[0])
                     
        X = np.array(X)
        X = X[:,:,:,:,0]
 
        bins = pow(10,np.linspace(np.log10(0.10), np.log10(32), self.bins_num))
        X_c = np.digitize(X, bins)   # classes ranges from 0 to bin_num
        
        return X, X_c
#%%
class CategoricalFocalLoss(tf.keras.losses.Loss):
        def __init__(self, alpha, gamma):
            super().__init__()
            self.alpha = alpha
            self.gamma = gamma
        
        def call(self, y_true, y_pred):

            epsilon = K.epsilon()
            y_pred = K.clip(y_pred, epsilon, 1. - epsilon)
            cross_entropy = -y_true * K.log(y_pred)
            loss = self.alpha * K.pow(1 - y_pred, self.gamma) * cross_entropy
            return K.mean(K.sum(loss, axis=-1))
        
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
    
def unet_convlstm_cls_class_v3_2(input_shape: Tuple[int] = (12, 256, 256, 4),
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

def unet_convlstm_reg_v3_2(input_shape: Tuple[int] = (12, 256, 256, 4),
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

def merge_patches(patches, patch_size, overlap, border, center, original_shape):
    """
    patches: list of patches
    patch_size: size of the patches (tuple of height and width)
    overlap: overlapping fraction (value between 0 and 1)
    original_shape: shape of the original image (height, width)
    """
    # Calculate the step size as patch_size * (1 - overlap)
    step_size = (int(patch_size[0] * (1 - overlap)), int(patch_size[1] * (1 - overlap)))

    # Calculate the number of patches in each direction
    n_rows = int((original_shape[1] - patch_size[0]) / step_size[0] + 1)
    n_cols = int((original_shape[2] - patch_size[1]) / step_size[1] + 1)


    # Create an empty image of the original shape
    original = np.zeros(original_shape, dtype=patches[0].dtype)

    # Iterate over the patches and place them on the original image
    idx = 0
    for i in range(n_rows):
        for j in range(n_cols):
            x_start = border  + i * center[0]
            x_end   = x_start + center[0]
            y_start = border  + j * center[1]
            y_end   = y_start + center[1]
            
            

            #print("y_start:", y_start, "   y_end:",y_end)
            patch = patches[idx,:]
            original[:,x_start:x_end, y_start:y_end] = patch[:,border:border+center[0], border:border+center[1]]
            
            idx += 1

    return original

def merge_patches_v2(patches, patch_size, overlap, border, center, original_shape_1, original_shape_2):
    """
    patches: list of patches
    patch_size: size of the patches (tuple of height and width)
    overlap: overlapping fraction (value between 0 and 1)
    original_shape: shape of the original image (height, width)
    """
    # Calculate the step size as patch_size * (1 - overlap)
    step_size = (int(patch_size[0] * (1 - overlap)), int(patch_size[1] * (1 - overlap)))

    # Calculate the number of patches in each direction
    n_rows = int((original_shape_1[1] - patch_size[0]) / step_size[0] + 1)
    n_cols = int((original_shape_1[2] - patch_size[1]) / step_size[1] + 1)


    # Create an empty image of the original shape
    original = np.zeros(original_shape_2, dtype=patches[0].dtype)

    # Iterate over the patches and place them on the original image
    idx = 0
    for i in range(n_rows):
        for j in range(n_cols):
            if i==0:
                if j==0:
                    x_start = i * center[0]
                    x_end   = x_start + center[0] + border
                    y_start = j * center[1]
                    y_end   = y_start + center[1] + border
                    
                    patch = patches[idx,:]
                    original[:,x_start:x_end, y_start:y_end] = patch[:,0:border+center[0], 0:border+center[1]]
                    idx += 1
                    
                elif j==n_cols-1:
                    x_start = i * center[0]
                    x_end   = x_start + center[0] + border
                    y_start = border + j * center[1]
                    y_end   = y_start + 80
                    
                    patch = patches[idx,:]
                    original[:,x_start:x_end, y_start:y_end] = patch[:,0:border+center[0], border:border+80]
                    idx += 1
                    
                else:
                    x_start = i * center[0]
                    x_end   = x_start + center[0] + border
                    y_start = border + j * center[1]
                    y_end   = y_start + center[1] 
                    
                    patch = patches[idx,:]
                    original[:,x_start:x_end, y_start:y_end] = patch[:,0:border+center[0], border:border+center[1]]
                    idx += 1
                    
                    
            elif i==n_rows-1:
                if j==0:
                    x_start = border+ i * center[0]
                    x_end   = x_start + 112
                    y_start = j * center[1]
                    y_end   = y_start + center[1] + border
                    patch = patches[idx,:]
                    original[:,x_start:x_end, y_start:y_end] = patch[:,border:border+112, 0:border+center[1]]
                    idx += 1
                    
                elif j==n_cols-1:
                    x_start = border+ i * center[0]
                    x_end   = x_start + 112
                    y_start = border + j * center[1]
                    y_end   = y_start + 80
                    patch = patches[idx,:]
                    original[:,x_start:x_end, y_start:y_end] = patch[:,border:border+112, border:border+80]
                    idx += 1
                    
                else:
                    x_start = border + i * center[0]
                    x_end   = x_start + 112
                    y_start = border + j * center[1]
                    y_end   = y_start + center[1] 
                    patch = patches[idx,:]
                    original[:,x_start:x_end, y_start:y_end] = patch[:,border:border+112, border:border+center[1]]
                    idx += 1
                    
            elif j==0 and i!=0 and i!=n_rows-1:
                x_start = border+ i * center[0]
                x_end   = x_start + center[0]
                y_start = j * center[1]
                y_end   = y_start + center[1] + border
                patch = patches[idx,:]
                original[:,x_start:x_end, y_start:y_end] = patch[:,border:border+center[0], 0:border+center[1]]
                idx += 1
                
            elif j==n_cols-1 and i!=0 and i!=n_rows-1:
                x_start = border+ i * center[0]
                x_end   = x_start + center[0]
                y_start = border+ j * center[1]
                y_end   = y_start + 80
                patch = patches[idx,:]
                original[:,x_start:x_end, y_start:y_end] = patch[:,border:border+center[0], border:border+80]
                idx += 1
            
           
            else:
            
                x_start = border  + i * center[0]
                x_end   = x_start + center[0]
                y_start = border  + j * center[1]
                y_end   = y_start + center[1]

                #print("y_start:", y_start, "   y_end:",y_end)
                patch = patches[idx,:]
                original[:,x_start:x_end, y_start:y_end] = patch[:,border:border+center[0], border:border+center[1]]

                idx += 1

    return original

def patchify_seq(mat_files, patch_size, overlap, parent_dir):
    """
    Function to convert an image into overlapping patches

    Args:
        img (ndarray): list of files directory
        patch_size (int): Size of the patches
        overlap (int): Overlap between consecutive patches

    Returns:
        list: List of patches
    """
    for k in range(len(mat_files)):
        directory = 'sample_'+str(k)
        path = os.path.join(parent_dir, directory) 
        os.makedirs(path) 
        
        mat_file = io.loadmat(mat_files[k])
        img_x = mat_file["X"]
        img_y = mat_file["y"]
        img_gfs = mat_file["y_gfs"]
        
        new_x1 = img_x[:,1200-80:1200,:,:]
        new_xx1 = np.concatenate((img_x,new_x1), axis=1)
        new_x2 = new_xx1[:,:,3600-112:3600,:]
        img_x = np.concatenate((new_xx1,new_x2), axis=2)
        
        new_x1 = img_y[:,1200-80:1200,:]
        new_xx1 = np.concatenate((img_y,new_x1), axis=1)
        new_x2 = new_xx1[:,:,3600-112:3600]
        img_y = np.concatenate((new_xx1,new_x2), axis=2)
        
        new_x1 = img_gfs[:,1200-80:1200,:]
        new_xx1 = np.concatenate((img_gfs,new_x1), axis=1)
        new_x2 = new_xx1[:,:,3600-112:3600]
        img_gfs = np.concatenate((new_xx1,new_x2), axis=2)

        patches = []
        x, y, _ = img_x.shape[1:4]
        x_stride = patch_size - overlap
        y_stride = patch_size - overlap
        row = 0
        col = 0
        for i in range(0, x-patch_size+1, x_stride):
            for j in range(0, y-patch_size+1, y_stride):
                patch_x = img_x[:,i:i+patch_size, j:j+patch_size,:]
                patch_y = img_y[:,i:i+patch_size, j:j+patch_size]
                patch_gfs = img_gfs[:,i:i+patch_size, j:j+patch_size]
                arr = np.array([patch_x, patch_y, patch_gfs], dtype=object)
                #put the directory that you want to save patches in here
                np.save(path+'/img_' +str(k)+'_patch_' +str(row)+'_'+str(col) +'.npy', arr)
                #patches.append(arr)
                col = col+1
            row = row+1
            col = 0
    return patches
