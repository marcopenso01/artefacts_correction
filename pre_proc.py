"""
Created on Wed Dec 14 12:13:04 2022

@author: Marco Penso
"""

import os
import numpy as np
import h5py
import cv2
import pydicom # for reading dicom files
import matplotlib.pyplot as plt
import shutil
from skimage import measure
X = []
Y = []

def keep_largest_connected_components(mask):
    '''
    Keeps only the largest connected components of each label for a segmentation mask.
    '''
    img = (mask == 1)
    out_img = np.zeros(mask.shape, dtype=np.uint8)
    temp_img = np.zeros(img.shape, dtype=np.uint8)

    blobs = measure.label(img, connectivity=1)  # find regions
    props = measure.regionprops(blobs)
    area = [ele.area for ele in props]  # area of each region
    largest_blob_ind = np.argmax(area)
    largest_blob_label = props[largest_blob_ind].label
    temp_img[blobs == largest_blob_label] = 255
    out_img[temp_img != 0] = 1
    return out_img


def rot(img, angle, interp=cv2.INTER_LINEAR):
    rows, cols = img.shape[:2]
    rotation_matrix = cv2.getRotationMatrix2D((cols / 2, rows / 2), angle, 1)
    return cv2.warpAffine(img, rotation_matrix, (cols, rows), flags=interp)


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


def click_event(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        # print(x,",",y)
        X.append(y)
        Y.append(x)
        cv2.destroyAllWindows()
        

def imfill(img):
    im_floodfill = img.copy()
    h, w = im_floodfill.shape[:2]
    mask = np.zeros((h+2, w+2), np.uint8)
    cv2.floodFill(im_floodfill, mask, (0,0), 255);
    return img | cv2.bitwise_not(im_floodfill)


def crop_or_pad_slice_to_size(slice, nx, ny):
    
    if len(slice.shape) == 3:
        stack = [slice[:,:,0], slice[:,:,1], slice[:,:,2]]
        RGB = []
    else:
        stack = [slice]
    
    for i in range(len(stack)):
        
        img = stack[i]
            
        x, y = img.shape
        
        x_s = (x - nx) // 2
        y_s = (y - ny) // 2
        x_c = (nx - x) // 2
        y_c = (ny - y) // 2
    
        if x > nx and y > ny:
            slice_cropped = img[x_s:x_s + nx, y_s:y_s + ny]
        else:
            slice_cropped = np.zeros((nx, ny))
            if x <= nx and y > ny:
                slice_cropped[x_c:x_c + x, :] = img[:, y_s:y_s + ny]
            elif x > nx and y <= ny:
                slice_cropped[:, y_c:y_c + y] = img[x_s:x_s + nx, :]
            else:
                slice_cropped[x_c:x_c + x, y_c:y_c + y] = img[:, :]
        if len(stack)>1:
            RGB.append(slice_cropped)
    
    if len(stack)>1:
        return np.dstack((RGB[0], RGB[1], RGB[2]))
    else:
        return slice_cropped
    
    
def crop_or_pad_slice_to_size_specific_point(slice, nx, ny, cx, cy):
    
    if len(slice.shape) == 3:
        stack = [slice[:,:,0], slice[:,:,1], slice[:,:,2]]
        RGB = []
    else:
        stack = [slice]
        
    for i in range(len(stack)):
        img = stack[i]
        x, y = img.shape
        y1 = (cy - (ny // 2))
        y2 = (cy + (ny // 2))
        x1 = (cx - (nx // 2))
        x2 = (cx + (nx // 2))
    
        if y1 < 0:
            img = np.append(np.zeros((x, abs(y1))), img, axis=1)
            x, y = img.shape
            y1 = 0
        if x1 < 0:
            img = np.append(np.zeros((abs(x1), y)), img, axis=0)
            x, y = img.shape
            x1 = 0
        if y2 > 512:
            img = np.append(img, np.zeros((x, y2 - 512)), axis=1)
            x, y = img.shape
        if x2 > 512:
            img = np.append(img, np.zeros((x2 - 512, y)), axis=0)
    
        slice_cropped = img[x1:x1 + nx, y1:y1 + ny]
        if len(stack)>1:
            RGB.append(slice_cropped)
        
    if len(stack)>1:
        return np.dstack((RGB[0], RGB[1], RGB[2]))
    else:
        return slice_cropped



'''
if __name__ == '__main__':
    
    # Paths settings
    path = r'F:/ARTEFACTS_SEG/data/ART'
    nx = 192
    ny = 192
    force_overwrite = True
    crop = 280
    paz = 'paz24'
    angle = -18
    # 
    
    output_folder = os.path.join(path, paz, 'pre_proc_ventr')
    if not os.path.exists(output_folder) or force_overwrite:
        if os.path.exists(output_folder):
            shutil.rmtree(output_folder)
        makefolder(output_folder)

    if not os.path.isfile(os.path.join(output_folder, 'pre_proc.hdf5')) or force_overwrite:
        
        print('This configuration of mode has not yet been preprocessed')
        print('Preprocessing now!')

        # ciclo su pazienti train
        IMG_SEG = []  # img in uint8 con segmentazione
        IMG_RAW = []  # img in float senza segmentazione
        IMG_SEL = []
        PAZ = []
        MASK = []
        
        paz_path = os.path.join(path, paz)
        
        print('---------------------------------------------------------------------------------')
        print('processing paz %s' % paz)
        n_img = 0

        path_seg = os.path.join(paz_path, 'SEG_VENTR')
        if not os.path.exists(path_seg):
            raise Exception('path %s not found' % path_seg)
        path_seg = os.path.join(path_seg, os.listdir(path_seg)[0])

        path_raw = os.path.join(paz_path, 'RAW')
        if not os.path.exists(path_raw):
            raise Exception('path %s not found' % path_raw)
        path_raw = os.path.join(path_raw, os.listdir(path_raw)[0])
        
        flag_sel = False
        path_sel = os.path.join(paz_path, 'selected')
        if os.path.exists(path_sel):
            flag_sel = True
            path_sel = os.path.join(path_sel, os.listdir(path_sel)[0])

        # select center image
        print('selec center ROI')
        X = []
        Y = []
        data_row_img = pydicom.dcmread(os.path.join(path_seg, os.listdir(path_seg)[0]))
        while True:
            img = data_row_img.pixel_array
            if angle != 0:
                img = rot(img, angle)
            cv2.imshow("image", img.astype('uint8'))
            cv2.namedWindow('image')
            cv2.setMouseCallback("image", click_event)
            k = cv2.waitKey(0)
            plt.figure()
            plt.imshow(crop_or_pad_slice_to_size_specific_point(img, crop, crop, X[-1], Y[-1]))
            plt.show()
            # press 'q' to exit
            if k == ord('q') or k == 27:
                break
            else:
                cv2.destroyAllWindows()          
        cv2.destroyAllWindows()
            
        print('center coordinate:', X[-1], Y[-1])
        
        for file in os.listdir(path_seg):
            
            data_row_img = pydicom.dcmread(os.path.join(path_seg, file))
            img = data_row_img.pixel_array
            if angle != 0:
                img = rot(img, angle)
            img = crop_or_pad_slice_to_size_specific_point(img, crop, crop, X[-1], Y[-1])
            
            temp_img = img.copy()
            for r in range(0, img.shape[0]):
                for c in range(0, img.shape[1]):
                    if img[r,c,0] == img[r,c,1] == img[r,c,2]:
                        temp_img[r,c,:]=0
                    else:
                        temp_img[r,c,:]=255
                        
            temp_img = temp_img[:,:,0]
            mask = imfill(temp_img)
            img = cv2.resize(img, (nx, ny), interpolation=cv2.INTER_AREA)
            mask = cv2.resize(mask, (nx, ny), interpolation=cv2.INTER_LINEAR)
            IMG_SEG.append(img)
            
            fig = plt.figure()
            ax1 = fig.add_subplot(121)
            ax1.imshow(img)
            ax2 = fig.add_subplot(122)
            ax2.imshow(mask)
            plt.title('img %s' % file);
            plt.show()
            
            mask[mask > 0] = 1
            MASK.append(mask)

            # save data raw
            data_row_img = pydicom.dcmread(os.path.join(path_raw, file))
            img = data_row_img.pixel_array
            if angle != 0:
                img = rot(img, angle)
            img = crop_or_pad_slice_to_size_specific_point(img, crop, crop, X[-1], Y[-1])
            img = cv2.resize(img, (nx, ny), interpolation=cv2.INTER_AREA)
            IMG_RAW.append(img)
                    
            PAZ.append(paz)
                
        if flag_sel:
            # save data selected
            for file in os.listdir(path_sel):
                dcmPath = os.path.join(path_sel, file)
                data_row_img = pydicom.dcmread(dcmPath)
                img = data_row_img.pixel_array
                if angle != 0:
                    img = rot(img, angle)
                img = crop_or_pad_slice_to_size_specific_point(img, crop, crop, X[-1], Y[-1])
                img = cv2.resize(img, (nx, ny), interpolation=cv2.INTER_AREA)
                IMG_SEL.append(img)
            

        hdf5_file = h5py.File(os.path.join(output_folder, 'pre_proc.hdf5'), "w")

        dt = h5py.special_dtype(vlen=str)
        hdf5_file.create_dataset('paz', (len(PAZ),), dtype=dt)
        hdf5_file.create_dataset('mask', [len(MASK)] + [nx, ny], dtype=np.uint8)
        hdf5_file.create_dataset('img_seg', [len(IMG_SEG)] + [nx, ny, 3], dtype=np.uint8)
        hdf5_file.create_dataset('img_raw', [len(IMG_RAW)] + [nx, ny], dtype=np.float32)

        for i in range(len(PAZ)):
            hdf5_file['paz'][i, ...] = PAZ[i]
            hdf5_file['mask'][i, ...] = MASK[i]
            hdf5_file['img_seg'][i, ...] = IMG_SEG[i]
            hdf5_file['img_raw'][i, ...] = IMG_RAW[i]
        
        if flag_sel:
            hdf5_file.create_dataset('img_sel', [len(IMG_SEL)] + [nx, ny], dtype=np.float32)
            for i in range(len(IMG_SEL)):
                hdf5_file['img_sel'][i, ...] = IMG_SEL[i]

        # After loop:
        hdf5_file.close()

    else:
        print('Already preprocessed this configuration!')
'''

if __name__ == '__main__':
    
    # Paths settings
    path = r'F:/ARTEFACTS_SEG/data/SANI'
    nx = 224
    ny = 224
    force_overwrite = True
    d = 30
    paz = 'paz125'
    angle = -21
    # 
    
    output_folder = os.path.join(path, paz, 'pre_proc_ventr')
    if not os.path.exists(output_folder) or force_overwrite:
        if os.path.exists(output_folder):
            shutil.rmtree(output_folder)
        makefolder(output_folder)

    if not os.path.isfile(os.path.join(output_folder, 'pre_proc.hdf5')) or force_overwrite:
        
        print('This configuration of mode has not yet been preprocessed')
        print('Preprocessing now!')

        # ciclo su pazienti train
        IMG_SEG = []  # img in uint8 con segmentazione
        IMG_RAW = []  # img in float senza segmentazione
        IMG_SEL = []
        PAZ = []
        MASK = []
        
        paz_path = os.path.join(path, paz)
        
        print('---------------------------------------------------------------------------------')
        print('processing paz %s' % paz)
        n_img = 0

        path_seg = os.path.join(paz_path, 'SEG_VENTR')
        if not os.path.exists(path_seg):
            raise Exception('path %s not found' % path_seg)
        path_seg = os.path.join(path_seg, os.listdir(path_seg)[0])

        path_raw = os.path.join(paz_path, 'RAW')
        if not os.path.exists(path_raw):
            raise Exception('path %s not found' % path_raw)
        path_raw = os.path.join(path_raw, os.listdir(path_raw)[0])
        
        flag_sel = False
        path_sel = os.path.join(paz_path, 'selected')
        if os.path.exists(path_sel):
            flag_sel = True
            path_sel = os.path.join(path_sel, os.listdir(path_sel)[0])

        CX = []
        CY = []
        LEN_X = []
        LEN_Y = []
        for ii in range(3):
            data_row_img = pydicom.dcmread(os.path.join(path_seg, os.listdir(path_seg)[ii]))
            img = data_row_img.pixel_array
            if angle != 0:
                img = rot(img, angle)
            temp_img = img.copy()
            for r in range(0, img.shape[0]):
                for c in range(0, img.shape[1]):
                    if img[r,c,0] == img[r,c,1] == img[r,c,2]:
                        temp_img[r,c,:]=0
                    else:
                        temp_img[r,c,:]=255
            
            temp_img = temp_img[:,:,0]
            mask = imfill(temp_img)
            mask[mask > 0] = 1
            mask = keep_largest_connected_components(mask)
                     
            a = mask.copy()
            #plt.imshow(a)
            contours, hier = cv2.findContours(a, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            top_left_x = 1000
            top_left_y = 1000
            bottom_right_x = 0
            bottom_right_y = 0
            for cntr in contours:
                x,y,w,h = cv2.boundingRect(cntr)
                if x < top_left_x:
                    top_left_x = x
                if y < top_left_y:
                    top_left_y= y
                if x+w-1 > bottom_right_x:
                    bottom_right_x = x+w-1
                if y+h-1 > bottom_right_y:
                    bottom_right_y = y+h-1        
            top_left = (top_left_x, top_left_y)
            bottom_right = (bottom_right_x, bottom_right_y)
            #print('top left=',top_left)
            #print('bottom right=',bottom_right)
            cx = int((top_left[1]+bottom_right[1])/2)   #row
            cy = int((top_left[0]+bottom_right[0])/2)   #column
            len_x = int(bottom_right[1]-top_left[1])
            len_y = int(bottom_right[0]-top_left[0])
            CX.append(cx)
            CY.append(cy)
            LEN_X.append(len_x)
            LEN_Y.append(len_y)
            
        cx = int(np.asarray(CX[-3:]).mean())
        cy = int(np.asarray(CY[-3:]).mean())
        len_x = int(np.asarray(LEN_X).max())
        len_y = int(np.asarray(LEN_Y).max())
    
        len_max = max(len_x, len_y)
        print(len_max)
        '''
        if len_max+d < 200:
            len_max = 200
            flag=0
        elif len_max+d > 200 and len_max+d < nx:
            len_max = nx
            flag=1
        else:
            len_max = len_max + d
            flag=2
        '''
        len_max = len_max + d
        print(len_max)      

        for file in os.listdir(path_seg):
            
            data_row_img = pydicom.dcmread(os.path.join(path_seg, file))
            img = data_row_img.pixel_array
            if angle != 0:
                img = rot(img, angle)
            img = crop_or_pad_slice_to_size_specific_point(img, len_max, len_max, cx, cy)
            
            temp_img = img.copy()
            for r in range(0, img.shape[0]):
                for c in range(0, img.shape[1]):
                    if img[r,c,0] == img[r,c,1] == img[r,c,2]:
                        temp_img[r,c,:]=0
                    else:
                        temp_img[r,c,:]=255
                        
            temp_img = temp_img[:,:,0]
            mask = imfill(temp_img)
            mask[mask > 0] = 1
            mask = keep_largest_connected_components(mask)
            '''
            if flag==0:
                img = cv2.resize(img, (nx, ny), interpolation=cv2.INTER_CUBIC)
            if flag==2:
                img = cv2.resize(img, (nx, ny), interpolation=cv2.INTER_AREA)
            mask = cv2.resize(mask, (nx, ny), interpolation=cv2.INTER_LINEAR)
            '''
            MASK.append(mask)
            IMG_SEG.append(img)
            
            mk = mask.copy()
            mk[mk>0]=255
            fig = plt.figure()
            ax1 = fig.add_subplot(121)
            ax1.imshow(img)
            ax2 = fig.add_subplot(122)
            ax2.imshow(mk)
            plt.title('img %s' % file);
            plt.show()
            
            #plt.figure()
            #plt.imshow(img)    

            # save data raw
            data_row_img = pydicom.dcmread(os.path.join(path_raw, file))
            img = data_row_img.pixel_array
            #solo per Pisa patient
            img = cv2.resize(img, (512, 512), interpolation=cv2.INTER_CUBIC)
            #
            if angle != 0:
                img = rot(img, angle)
            img = crop_or_pad_slice_to_size_specific_point(img, len_max, len_max, cx, cy)
            '''
            if flag==0:
                img = cv2.resize(img, (nx, ny), interpolation=cv2.INTER_CUBIC)
            if flag==2:
                img = cv2.resize(img, (nx, ny), interpolation=cv2.INTER_AREA)
            '''
            IMG_RAW.append(img)
                    
            PAZ.append(paz)
                
        if flag_sel:
            # save data selected
            for file in os.listdir(path_sel):
                dcmPath = os.path.join(path_sel, file)
                data_row_img = pydicom.dcmread(dcmPath)
                img = data_row_img.pixel_array
                #solo per Pisa patient
                img = cv2.resize(img, (512, 512), interpolation=cv2.INTER_CUBIC)
                #
                if angle != 0:
                    img = rot(img, angle)
                img = crop_or_pad_slice_to_size_specific_point(img, len_max, len_max, cx, cy)
                '''
                if flag==0:
                    img = cv2.resize(img, (nx, ny), interpolation=cv2.INTER_CUBIC)
                if flag==2:
                    img = cv2.resize(img, (nx, ny), interpolation=cv2.INTER_AREA)
                '''
                IMG_SEL.append(img)
            

        hdf5_file = h5py.File(os.path.join(output_folder, 'pre_proc.hdf5'), "w")
        
        nx=len_max
        ny=len_max
        dt = h5py.special_dtype(vlen=str)
        hdf5_file.create_dataset('paz', (len(PAZ),), dtype=dt)
        hdf5_file.create_dataset('mask', [len(MASK)] + [nx, ny], dtype=np.uint8)
        hdf5_file.create_dataset('img_seg', [len(IMG_SEG)] + [nx, ny, 3], dtype=np.uint8)
        hdf5_file.create_dataset('img_raw', [len(IMG_RAW)] + [nx, ny], dtype=np.float32)

        for i in range(len(PAZ)):
            hdf5_file['paz'][i, ...] = PAZ[i]
            hdf5_file['mask'][i, ...] = MASK[i]
            hdf5_file['img_seg'][i, ...] = IMG_SEG[i]
            hdf5_file['img_raw'][i, ...] = IMG_RAW[i]
        
        if flag_sel:
            hdf5_file.create_dataset('img_sel', [len(IMG_SEL)] + [nx, ny], dtype=np.float32)
            for i in range(len(IMG_SEL)):
                hdf5_file['img_sel'][i, ...] = IMG_SEL[i]

        # After loop:
        hdf5_file.close()

    else:
        print('Already preprocessed this configuration!')


'''
# conta imgs
path =r'F:/ARTEFACTS_SEG/data/ART'
somma=0
for paz in os.listdir(path):
    paz_path = os.path.join(path, paz)
    if os.path.exists(os.path.join(paz_path, 'pre_proc_ventr')):
        data = h5py.File(os.path.join(paz_path, 'pre_proc_ventr', 'pre_proc.hdf5'), "r")
        somma = somma + len(data['img_raw'][()])
        data.close()
print(somma)

path =r'F:/ARTEFACTS_SEG/data/ART'
somma=0
for paz in os.listdir(path):
    paz_path = os.path.join(path, paz)
    if os.path.exists(os.path.join(paz_path, 'pre_proc_ventr')):
        data = h5py.File(os.path.join(paz_path, 'pre_proc_ventr', 'pre_proc.hdf5'), "r")
        try:       
            somma = somma + len(data['img_sel'][()])
        except:
            continue
        data.close()
print(somma)
'''
