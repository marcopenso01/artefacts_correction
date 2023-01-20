import os
import numpy as np
import h5py
import cv2

def makefolder(folder):
    '''
    Helper function to make a new folder if doesn't exist
    :param folder: path to new folder
    :return: True if folder created, False if folder already exists
    '''
    if not os.path.exists(folder):
        os.makedirs(folder)
        return True
    return False

def resize(array, size):
    new_arr = []
    for i in range(len(array)):
        if array.shape[-1] < size:
            new_arr.append(cv2.resize(array[i], (size, size), interpolation=cv2.INTER_CUBIC))
        elif array.shape[-1] > size:
            new_arr.append(cv2.resize(array[i], (size, size), interpolation=cv2.INTER_AREA))
        else:
            new_arr.append(array[i])
    return new_arr
    

def concatenate(input_path, out_fold, out_file, size):
    dt = h5py.special_dtype(vlen=str)
    if not os.path.exists(out_fold):
        makefolder(out_fold)
    c=1
    for paz in os.listdir(input_path):
        paz_path = os.path.join(input_path, paz)
        if os.path.exists(os.path.join(paz_path, 'pre_proc_ventr')):            
            print(paz)
            data = h5py.File(os.path.join(paz_path, 'pre_proc_ventr', 'pre_proc.hdf5'), 'r')
            d1 = np.asarray(resize(data['img_raw'][()], size))  
            d2 = np.asarray(resize(data['img_seg'][()], size))
            d3 = np.asarray(resize(data['mask'][()], size))
            d4 = data['paz'][()]
            #print("img_raw:", d1.shape, d1.dtype)
            #print("mask:", d3.shape, d3.dtype)   
            if c==1:
                img_raw = d1
                img_seg = d2
                mask = d3
                pat = d4
                c += 1
            else:
                img_raw = np.concatenate((img_raw, d1), axis=0)
                img_seg = np.concatenate((img_seg, d2), axis=0)
                mask = np.concatenate((mask, d3), axis=0)
                pat = np.concatenate((pat, d4), axis=0)
                
            print("img_raw after conc:", img_raw.shape)
            data.close()
            
    data_file_path = os.path.join(out_fold, out_file+'.hdf5')
    hdf5_file = h5py.File(data_file_path, "w")
    hdf5_file.create_dataset('paz', pat.shape, dtype=dt)
    hdf5_file.create_dataset('mask', mask.shape, mask.dtype)
    hdf5_file.create_dataset('img_seg', img_seg.shape, img_seg.dtype)
    hdf5_file.create_dataset('img_raw', img_raw.shape, img_raw.dtype)
  
    hdf5_file['paz'][()] = pat
    hdf5_file['mask'][()] = mask
    hdf5_file['img_seg'][()] = img_seg
    hdf5_file['img_raw'][()] = img_raw
    
    hdf5_file.close()
    

def concatenate2(input_path, out_fold, out_file, size):
    dt = h5py.special_dtype(vlen=str)
    d1 = []
    d2 = []
    for paz in os.listdir(input_path):
        paz_path = os.path.join(input_path, paz)
        if os.path.exists(os.path.join(paz_path, 'pre_proc_ventr')) and os.path.exists(os.path.join(paz_path, 'selected')):
            print(paz)
            data = h5py.File(os.path.join(paz_path, 'pre_proc_ventr', 'pre_proc.hdf5'), 'r')
            arr = np.asarray(resize(data['img_sel'][()], size))
            paz_name = data['paz'][0]
            for ii in range(len(arr)):
                d1.append(arr[ii])
                d2.append(paz_name)
            print("img_sel after conc:", len(d1))
            data.close()
    d1 = np.asarray(d1)
    d2 = np.asarray(d2)
    
    data_file_path = os.path.join(out_fold, out_file+'.hdf5')
    hdf5_file = h5py.File(data_file_path, "w")
    hdf5_file.create_dataset('paz', d2.shape, dtype=dt)
    hdf5_file.create_dataset('img_sel', d1.shape, d1.dtype)
  
    hdf5_file['paz'][()] = d2
    hdf5_file['img_sel'][()] = d1
    
    hdf5_file.close()
    
    
input_path = r'F:\ARTEFACTS_SEG\data\ART'
out_fold = 'F:\ARTEFACTS_SEG\data\output'
out_file = 'sel_art'
size = 224
concatenate2(input_path, out_fold, out_file, size)
