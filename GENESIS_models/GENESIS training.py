import numpy as np
from scipy import io
import random
from typing import List, Tuple
from tensorflow import keras
import tensorflow as tf
import tensorflow.keras.layers as layers
from tensorflow.keras.optimizers import Adam, SGD, RMSprop
from tensorflow.keras import Model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras import mixed_precision
from tensorflow.keras.losses import mse
from tensorflow.keras.layers import (Activation, BatchNormalization, Conv3D,
                          Conv3DTranspose, ConvLSTM2D, Dropout,
                          Input, MaxPool3D)
import os
from utils import *

tf.config.optimizer.set_experimental_options({"layout_optimizer": False})

os.environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_async"

policy = mixed_precision.Policy("mixed_float16")
mixed_precision.set_global_policy(policy)

gpus = tf.config.experimental.list_physical_devices(device_type="GPU")
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

print(tf.config.list_physical_devices("GPU"))
#%%
class CustomGenerator(keras.utils.Sequence):
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
        yy = np.array(yy)

        # Making classes on the X based on logspace
        X_rain = X[:,:,:,:,0]
        X_rain[X_rain<0.1] = 0.1
        X_rain_log   = np.log10(X_rain)
        X[:,:,:,:,0] = X_rain_log
        
        for i in range(4):
           X[:,:,:,:,i] = (X[:,:,:,:,i]-self.mu[i])/self.std[i]

        
        # Making classes on the y based on logspace
        yy[yy<0.1] = 0.1
        yy = np.log10(yy)
        y = yy

        return X, y
#%%
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
#%%
def unet_conv3d_reg(input_shape: Tuple[int] = (12, 256, 256, 4),
         num_filters_base: int = 8,
         dropout_rate: float = 0.1):
    
    input_shape = (12,256,256,4)
    inputs = Input(shape=input_shape)
    x_init = BatchNormalization()(inputs)  # Try with normalizing the dataset
    x0 = x_init
        
    x_conv1_b1 = Conv3D(filters= num_filters_base, kernel_size=(1,3,3), padding='same', activation="relu")(x0)
    x_conv2_b1 = Conv3D(filters=num_filters_base, kernel_size=(1,3,3), padding='same', activation="relu")(x_conv1_b1)
    x_max_b1 = MaxPool3D([1, 2, 2],padding='same')(x_conv2_b1)
    x_bn_b1 = BatchNormalization()(x_max_b1)
    x_do_b1 = Dropout(dropout_rate)(x_bn_b1)
    
    x_conv1_b2 = Conv3D(filters=2*num_filters_base, kernel_size=(1,3,3), padding='same', activation="relu")(x_do_b1)
    x_conv2_b2 = Conv3D(filters=2*num_filters_base, kernel_size=(1,3,3), padding='same', activation="relu")(x_conv1_b2)
    x_max_b2 = MaxPool3D([1, 2, 2],padding='same')(x_conv2_b2)
    x_bn_b2 = BatchNormalization()(x_max_b2)
    x_do_b2 = Dropout(dropout_rate)(x_bn_b2)
    
    x_conv1_b3 = Conv3D(filters=4*num_filters_base, kernel_size=(1,3,3), padding='same', activation="relu")(x_do_b2)
    x_conv2_b3 = Conv3D(filters=4*num_filters_base, kernel_size=(1,3,3), padding='same', activation="relu")(x_conv1_b3)
    x_max_b3 = MaxPool3D([1, 2, 2],padding='same')(x_conv2_b3)
    x_bn_b3 = BatchNormalization()(x_max_b3)
    x_do_b3 = Dropout(dropout_rate)(x_bn_b3)
    
    x_conv1_b4 = Conv3D(filters=8*num_filters_base, kernel_size=(1,3,3), padding='same', activation="relu")(x_do_b3)
    x_conv2_b4 =Conv3D(filters=8*num_filters_base, kernel_size=(1,3,3), padding='same', activation="relu")(x_conv1_b4)
    x_max_b4 = MaxPool3D([1, 2, 2],padding='same')(x_conv2_b4)
    x_bn_b4 = BatchNormalization()(x_max_b4)
    x_do_b4 = Dropout(dropout_rate)(x_bn_b4)
    
    # ------- Head Residual Output (Residual Decoder)
    
    x_conv1_b5 = Conv3D(filters=8*num_filters_base, kernel_size=(1,3,3), padding='same', activation="relu")(x_do_b4)
    x_conv2_b5 =Conv3D(filters=8*num_filters_base, kernel_size=(1,3,3), padding='same', activation="relu")(x_conv1_b5)
    x_deconv_b5 = Conv3DTranspose(filters=8*num_filters_base, kernel_size=(1, 2, 2),strides=(1,2,2),padding='same', activation="relu")(x_conv2_b5)
    x_bn_b5 = BatchNormalization()(x_deconv_b5)
    x_do_b5 = Dropout(dropout_rate)(x_bn_b5)
    
    
    cropped_x_conv2_b4 = layers.Cropping3D(cropping=((11,0),(0,0),(0,0)),data_format="channels_last")(x_conv2_b4)
    cropped_x_conv2_b4 = layers.concatenate([cropped_x_conv2_b4]*12,axis=1)
    x_conv1_b6 = Conv3D(filters=4*num_filters_base, kernel_size=(2,1,1), activation="relu")(layers.concatenate([cropped_x_conv2_b4,x_do_b5]))
    x_conv2_b6 = Conv3D(filters=4*num_filters_base, kernel_size=(1, 1, 1),padding='same', activation="relu")(x_conv1_b6)
    x_deconv_b6 = Conv3DTranspose(filters=4*num_filters_base,  kernel_size=(1, 2, 2),strides = (1,2,2),padding='same', activation="relu")(x_conv2_b6)
    x_bn_b6 = BatchNormalization()(x_deconv_b6)
    x_do_b6 = Dropout(dropout_rate)(x_bn_b6)
    
    
    cropped_x_conv2_b3 = layers.Cropping3D(cropping=((11,0),(0,0),(0,0)),data_format="channels_last")(x_conv2_b3)
    cropped_x_conv2_b3 = layers.concatenate([cropped_x_conv2_b3]*11,axis=1)
    x_conv1_b7 =Conv3D(filters=2*num_filters_base, kernel_size=(2,1,1), activation="relu")(layers.concatenate([cropped_x_conv2_b3,x_do_b6]))
    x_conv2_b7 = Conv3D(filters=2*num_filters_base, kernel_size=(1, 1, 1),padding='same', activation="relu")(x_conv1_b7)
    x_deconv_b7 =  Conv3DTranspose(filters=2*num_filters_base,  kernel_size=(1, 2, 2),strides = (1,2,2),padding='same', activation="relu")(x_conv2_b7)
    x_bn_b7 = BatchNormalization()(x_deconv_b7)
    x_do_b7 = Dropout(dropout_rate)(x_bn_b7)
    
    
    cropped_x_conv2_b2 =  layers.Cropping3D(cropping=((11,0),(0,0),(0,0)),data_format="channels_last")(x_conv2_b2)
    cropped_x_conv2_b2 = layers.concatenate([cropped_x_conv2_b2]*10,axis=1)
    x_conv1_b8 = Conv3D(filters=1*num_filters_base, kernel_size=(2,1,1), activation="relu")(layers.concatenate([cropped_x_conv2_b2,x_do_b7]))
    x_conv2_b8 = Conv3D(filters=1*num_filters_base, kernel_size=(1, 1, 1),padding='same', activation="relu")(x_conv1_b8)
    x_deconv_b8 = Conv3DTranspose(filters=2*num_filters_base,  kernel_size=(1, 2, 2),strides = (1,2,2),padding='same', activation="relu")(x_conv2_b8)
    x_bn_b8 = BatchNormalization()(x_deconv_b8)
    x_do_b8 = Dropout(dropout_rate)(x_bn_b8)
    
    
    cropped_x_conv2_b1 = layers.Cropping3D(cropping=((11,0),(0,0),(0,0)),data_format="channels_last")(x_conv2_b1)
    cropped_x_conv2_b1 = layers.concatenate([cropped_x_conv2_b1]*9,axis=1)
    x_conv1_b9 =Conv3D(filters=0.5*num_filters_base, kernel_size=(2,1,1), activation="relu")(layers.concatenate([cropped_x_conv2_b1,x_do_b8]))
    x_conv2_b9 = Conv3D(filters=0.5*num_filters_base, kernel_size=(1, 1, 1),padding='same', activation="relu")(x_conv1_b9)
    x_bn_b9 = BatchNormalization()(x_conv2_b9)
    x_do_b9 = Dropout(dropout_rate)(x_bn_b9)
    
    
    residual_output = Conv3DTranspose(1, kernel_size=(1, 1, 1), padding="same")(x_do_b9)
    output = Activation("linear", dtype="float32")(residual_output)
    #output = Cropping3D(cropping=(0, 0, 2))(output)
    
    output = tf.squeeze(output, axis=4)

    model=Model(inputs, output)
    return model
#%%
num_filters_base = 8
dropout_rate     = 0.2
learning_rate    = 0.0001
batch_size       = 8
model = unet_convlstm_reg((12, 256, 256, 4),
                            num_filters_base=num_filters_base,
                            dropout_rate=dropout_rate)
model.summary()
#%%
# loading mean and standard deviation
path = "/panfs/jay/groups/0/ebtehaj/rahim035/paper_2/V2/Models/full_models/dataset/"

#Define save_path and filename for the trained model here
save_path = "/panfs/jay/groups/0/ebtehaj/rahim035/paper_2/V2/Models/full_models/regression/MSE/physics/lstm/"
save_name = f"mse_lstm_phys_{batch_size}_{num_filters_base}"

stat = io.loadmat(path + "stat_full_models.mat")
mean_x = stat["mean_x"]
std_x  = stat["std_x"]
std    = np.transpose(std_x)
mu     = np.transpose(mean_x)
#%%
batch_size = 8
train_paths = np.load(path +'train_list_50000.npy', allow_pickle = True)
val_paths   = np.load(path +'val_list_12000.npy'  , allow_pickle = True)
test_paths  = np.load(path +'test_list_1000.npy'  , allow_pickle = True)

train_paths = train_paths.tolist()
val_paths   = val_paths.tolist()
test_paths  = test_paths.tolist()

train_dataset = CustomGenerator(train_paths, mu, std, batch_size)
val_dataset   = CustomGenerator(val_paths,   mu, std, batch_size)
test_dataset  = CustomGenerator(test_paths,  mu, std, 1, shuffle=False)
#%%
loss = "mean_squared_error"

model.compile(
    loss=loss,
    optimizer=keras.optimizers.Adam(learning_rate=0.0001),
    metrics=["mae"])

logger_filename = f"{save_path}/{save_name}.csv"
checkpoint_filepath = f"{save_path}/{save_name}.h5"
callback_2 = tf.keras.callbacks.CSVLogger(logger_filename, separator=",", append=True)

callbacks = [
    EarlyStopping(patience=5, verbose=1),
    ReduceLROnPlateau(factor=0.1, patience=2, min_lr=1e-16,
                      verbose=1),
    ModelCheckpoint(filepath=checkpoint_filepath,
                    verbose=1,
                    save_best_only=True)
]
#%%
del model
load_path = f"{save_path}/{save_name}_final.h5"
model = tf.keras.models.load_model(load_path, compile=False)
model.compile(
    loss=loss,
    optimizer=keras.optimizers.Adam(learning_rate=0.0001),
    metrics=["mae"])

#%%
print("Starting fit")
results = model.fit(train_dataset,
                                batch_size=batch_size,
                                epochs=10,
                                callbacks=[callbacks, callback_2],
                                verbose=1,
                                validation_data = val_dataset)

model.save(f"{save_path}/{save_name}_final.h5")
