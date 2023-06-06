import numpy as np
import random
from typing import List, Tuple
import tensorflow as tf
from tensorflow import keras
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
        gfs_yy = []
        for i in range(self.batch_size):
            arr = np.load(batch_x[i], allow_pickle=True)
            X.append(arr[0])
            yy.append(arr[1])
            gfs_yy.append(arr[2])
         
        X      = np.array(X)
        yy     = np.array(yy)
        gfs_yy = np.array(gfs_yy)

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
        
        gfs_yy[gfs_yy<0.001] = 0.001
        gfs_yy = np.log10(gfs_yy)
        gfs = gfs_yy

        return X, y, gfs
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
        gfs_yy = []
        for i in range(self.batch_size):
            arr = np.load(batch_x[i], allow_pickle=True)
            X.append(arr[0])
            yy.append(arr[1])
            gfs_yy.append(arr[2])
                     
        X = np.array(X) 
        yy = np.array(yy)
        gfs_yy = np.array(gfs_yy)
                     
        X_rain = X[:,:,:,:,0]
        X_rain[X_rain<0.1] = 0.1
        X_rain_log   = np.log10(X_rain)
        X[:,:,:,:,0] = X_rain_log
                     
        for i in range(4):
           X[:,:,:,:,i] = (X[:,:,:,:,i]-self.mu[i])/self.std[i]            
                     
        bins = np.linspace(np.log10(0.101), np.log10(32), self.bins_num)
        # Making classes on the y based on logspace
        yy[yy<0.1] = 0.1
        yy = np.log10(yy)
        yy = np.digitize(yy, bins)   # classes ranges from 0 to bin_num
        y = to_categorical(yy, num_classes = self.bins_num+1)
        
        gfs_yy[gfs_yy<0.001] = 0.001
        gfs_yy = np.log10(gfs_yy)
        gfs_yy = np.digitize(gfs_yy, bins)   # classes ranges from 0 to bin_num
        gfs = to_categorical(gfs_yy, num_classes = self.bins_num+1)

        return X, y, gfs
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
