#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import numpy as np
import torch
import torchvision 
import nibabel as nib
import skimage.filters
import random
import SimpleITK as sitk
import matplotlib.pyplot as plt 
from pandas_ods_reader import read_ods
import pandas as pd
from skimage.transform import pyramid_laplacian, pyramid_expand, resize


# In[2]:


base = '/media/nguyen-k/A75D-9A3A/CASSIOPET/dcm_baseline'
path_squelette = '/home/nguyen-k/Bureau/segCassiopet/label_squelette'
path_label = '/home/nguyen-k/Bureau/segCassiopet/labels_seg'
base_CT = '/home/nguyen-k/Bureau/segCassiopet/imagesCT'
seuil = 150000
base_nrrd = '/home/nguyen-k/Bureau/segCassiopet/dcm_test'


# In[3]:


def resize_by_axis_pytorch(image, dim1, dim2, ax):
    
    resized_list = []
    
    image = torch.from_numpy(image)
    unstack_img_depth_list = torch.unbind(image, dim = ax)    
    unstack_img_depth_list = [e.numpy() for e in unstack_img_depth_list]
    
    data_transform = torchvision.transforms.Resize((dim1, dim2))
    for i in unstack_img_depth_list:        
        h,w,c = i.shape
        i = torch.tensor(i)
        i = torch.reshape(i, (c,h,w))
        i = data_transform(i)
        c,h,w = i.size(dim=0), i.size(dim=1),i.size(dim=2)
        i = torch.reshape(i, (h,w,c))  
        resized_list.append(i)
    
    stack_img = torch.stack(resized_list, dim = ax)    

    return stack_img


# In[4]:


def load_image_PET(p) :

    #Image patient
    img = nib.load(os.path.join(base, p,'patient_SUV.nii')).get_fdata() 
        
    path = os.path.join(path_squelette, 'cassiopet_' + p.replace('-','')+'.nii.gz')
    if os.path.isfile(path) : 
        img_squelette = nib.load(path).get_fdata()

        #Position de la tête
        img_tete = np.where(img_squelette == 2, 1, 0)
        pos_tete_max = img.shape[2] - 1
        while np.max(img_tete[:,:,pos_tete_max]) == 0 and pos_tete_max > 0:
            pos_tete_max = pos_tete_max - 1
        if pos_tete_max == 0 :
            pos_tete_max = img.shape[2] - 1
            pos_tete_min = img.shape[2] - 1
        else : 
            pos_tete_min = pos_tete_max - 50
            while np.max(img_tete[:,:,pos_tete_min]) == 0 :
                pos_tete_min = pos_tete_min + 1
    else :
        pos_tete_max, pos_tete_min = img.shape[2] - 1, img.shape[2] - 1

    position = pos_tete_min, pos_tete_max

    #Get Spacing
    img_sitk = sitk.ReadImage(os.path.join(base, p,'patient.nii'))  
    s_patient = img_sitk.GetSpacing()

    return img, position, s_patient


# In[5]:


def load_image_CT(p) :

    #Image patient
    path = os.path.join(base_CT, 'cassiopet_'+p.replace('-', '')+'_0001.nii')
    img = nib.load(path).get_fdata() 
    max_HU = np.amax(img)
    img /= max_HU

    #Get Spacing
    img_sitk = sitk.ReadImage(path)  
    s_patient = img_sitk.GetSpacing()

    return img, s_patient


# In[6]:


def load_image_and_label(p) :

    #Image patient
    img = nib.load(os.path.join(base, p,'patient_SUV.nii')).get_fdata()     
    path = os.path.join(path_squelette, 'cassiopet_' + p.replace('-','')+'.nii.gz')

    #Position de la tête
    if os.path.isfile(path) : 
        img_squelette = nib.load(path).get_fdata()
        img_tete = np.where(img_squelette == 2, 1, 0)
        pos_tete_max = img.shape[2] - 1
        while np.max(img_tete[:,:,pos_tete_max]) == 0 and pos_tete_max > 0:
            pos_tete_max = pos_tete_max - 1
        if pos_tete_max == 0 :
            pos_tete_max = img.shape[2] - 1
            pos_tete_min = img.shape[2] - 1
        else : 
            pos_tete_min = pos_tete_max - 50
            while np.max(img_tete[:,:,pos_tete_min]) == 0 :
                pos_tete_min = pos_tete_min + 1
    else :
        pos_tete_max, pos_tete_min = img.shape[2] - 1, img.shape[2] - 1
    
    position = pos_tete_min, pos_tete_max
    
    #Label lésion
    path_1 = os.path.join(path_label, 'cassiopet_' + p.replace('-','')+'.nii.gz')
    if os.path.isfile(path_1) :
        img_label = nib.load(path_1).get_fdata()
        img_squelette = np.where(img_label == 1, 1, img_squelette)
    else :
        img_label = img * 0

    #Get Spacing
    img_sitk = sitk.ReadImage(os.path.join(base, p,'patient.nii'))  
    s_patient = img_sitk.GetSpacing()

    return img, img_label, position, s_patient 


# In[7]:


def preprocessing_image(img, positions, s_patient,flip,blur, scale, sigma, dim, spacing) :
    
    pos_tete_min, _ = positions   
    l_max = pos_tete_min-10

    img = img[ : , :, 0:l_max]

    #NORMALISATION SPACING
    new_dim = [0, 0, 0]
    for i in range(len(s_patient)) :
        new_dim[i] = int(img.shape[i]*s_patient[i]/spacing*scale[i])    
    
    img = np.expand_dims(img, axis = -1)
    img = resize_by_axis_pytorch(img, new_dim[0], new_dim[1], 2)    
    img = resize_by_axis_pytorch(img.numpy(), new_dim[0], new_dim[2], 1)
    img = np.squeeze(img).numpy()    

    #Crop
    if new_dim[0] > dim[0] :
        img = img[int((new_dim[0]-dim[0])/2):int((new_dim[0]+dim[0])/2), :, :]

    if new_dim[1] > dim[1] :
        img = img[:, int((new_dim[1]-dim[1])/2):int((new_dim[1]+dim[1])/2), :]

    if new_dim[2] > dim[2] :
        img = img[:, :, new_dim[2] - dim[2] : new_dim[2]]

    #Padding
    if new_dim[0] < dim[0] : 
        temp = np.zeros((dim[0], img.shape[1], img.shape[2]))
        temp[int((dim[0]-new_dim[0])/2):int((new_dim[0]+dim[0])/2), :, :] = img
        img = temp.copy()

    if new_dim[1] < dim[1] : 
        temp = np.zeros((img.shape[0], dim[1], img.shape[2]))
        temp[:, int((dim[1]-new_dim[1])/2):int((new_dim[1]+dim[1])/2), :] = img
        img = temp.copy()       

    if new_dim[2] < dim[2] : 
        temp = np.zeros((img.shape[0], img.shape[1], dim[2]))
        temp[:, :, dim[2] - new_dim[2] : dim[2]] = img
        img = temp.copy()

    if flip :
        img = np.flip(img, 0)
    if blur :
        img = skimage.filters.gaussian(img, sigma=(sigma[0], sigma[1], sigma[2]), truncate=3.5, channel_axis=True)
            
    return img, new_dim


# In[8]:


def preprocessing_image_and_label(img, img_label, positions, s_patient, scale, flip, blur, sigma, dim, spacing) :

    pos_tete_min, _ = positions   
    l_max = pos_tete_min-10
    
    img, new_dim = preprocessing_image(img, positions, s_patient, flip=flip, blur=blur, scale=scale, sigma=sigma, dim = dim, spacing = spacing)

    #LABEL

    if np.amax(img_label) == 0 :
        img_label = img * 0
    else : 
        img_label = img_label[ : , :, 0:l_max]

        img_label = np.expand_dims(img_label, axis = -1)
        img_label = resize_by_axis_pytorch(img_label, new_dim[0], new_dim[1], 2)    
        img_label = resize_by_axis_pytorch(img_label.numpy(), new_dim[0], new_dim[2], 1)    
        img_label = np.squeeze(img_label).numpy()
        img_label = np.where(img_label > 0.5, 1, 0)   

        #Crop
        if new_dim[0] > dim[0] :
            img_label = img_label[int((new_dim[0]-dim[0])/2):int((new_dim[0]+dim[0])/2), :, :]

        if new_dim[1] > dim[1] :
            img_label = img_label[:, int((new_dim[1]-dim[1])/2):int((new_dim[1]+dim[1])/2), :]

        if new_dim[2] > dim[2] :
            img_label = img_label[:, :, new_dim[2] - dim[2] : new_dim[2]]

        #Padding
        if new_dim[0] < dim[0] : 
            temp = np.zeros((dim[0], img_label.shape[1], img_label.shape[2]))
            temp[int((dim[0]-new_dim[0])/2):int((new_dim[0]+dim[0])/2), :, :] = img_label
            img_label = temp.copy()
        
        if new_dim[1] < dim[1] :
            temp = np.zeros((img_label.shape[0], dim[1], img_label.shape[2]))
            temp[:, int((dim[1]-new_dim[1])/2):int((new_dim[1]+dim[1])/2), :] = img_label
            img_label = temp.copy()   
        
        if new_dim[2] < dim[2] : 
            temp = np.zeros((img_label.shape[0], img_label.shape[1], dim[2]))
            temp[:, :, dim[2] - new_dim[2] : dim[2]] = img_label
            img_label = temp.copy()
        #Flip
        if flip :
            img_label = np.flip(img_label, 0)       
        
        #img_label = np.expand_dims(img_label, axis = 0)  
    #BLUR        
    if blur : 
        img = skimage.filters.gaussian(img, sigma=(sigma[0], sigma[1], sigma[2]), truncate=3.5, channel_axis=True)
        
    return img, img_label


# In[10]:


def score_de_deauville(task=1) : 
    reponse_path = '/home/nguyen-k/Bureau/segCassiopet/Deauville_DL_Allods.ods'
    reponse_data = read_ods(reponse_path, 1)
    reponse_data = reponse_data[reponse_data['Reponse'].notna()] 
    reponse_data.reset_index(drop=True, inplace=True)

    base = '/media/nguyen-k/A75D-9A3A/CASSIOPET/dcm_baseline'
    list_dir = os.listdir(base)

    list_patient = []
    label_classe = []

    if task == 3 : 

        for i in range(0, len(list_dir)) :
            ref = 0
            while ref < reponse_data.shape[0] and reponse_data['Patient'][ref]!= list_dir[i]:
                ref = ref + 1  

            path_PET = os.path.join(base, list_dir[i],'patient_SUV.nii')
            path_CT = os.path.join(base_CT, 'cassiopet_'+list_dir[i].replace('-', '')+'_0001.nii')
            if ref != reponse_data.shape[0] and os.path.isfile(path_PET) and os.path.isfile(path_CT) :
                list_patient.append(reponse_data['Patient'][ref])
                label_classe.append(reponse_data['Reponse'][ref]-1)
    
    else :

        for i in range(0, len(list_dir)) :
            ref = 0
            while ref < reponse_data.shape[0] and reponse_data['Patient'][ref]!= list_dir[i]:
                ref = ref + 1  
            if task == 1 : 
                path_PET = os.path.join(base, list_dir[i],'patient_SUV.nii')
            else : 
                path_PET = os.path.join(base, list_dir[i],'patient.nii')

            if ref != reponse_data.shape[0] and os.path.isfile(path_PET)  :
                list_patient.append(reponse_data['Patient'][ref])
                label_classe.append(reponse_data['Reponse'][ref]-1)
            
    return list_patient, label_classe


# In[13]:


def get_clinic_data(patient) :
    survie_data = read_ods('/home/nguyen-k/Bureau/segCassiopet/Clinics.ods', 1)
    ref = 0
    while ref < survie_data.shape[0] and survie_data['Patient'][ref]!= patient:
        ref = ref + 1 
    if ref != survie_data.shape[0] :
        age = survie_data['Age'][ref]
        if survie_data['Sex'][ref] == 'Male' :
            sex = 1
        else : sex = 0
        if survie_data['Risk_Result'][ref] == 'STANDARD RISK' : 
            risk = 0
        else : risk = 1
    else :
        print(patient, 'not found')
        age = 0
        sex = 2
        risk = 2

    clinic_data = (age, sex, risk)

    return clinic_data     


# In[ ]:


def Laplace_pyr_fusion(img_PET,img_CT,max_layer=5,downscale=2):
    
    lapl_pyr_CT = tuple(pyramid_laplacian(img_CT, max_layer=max_layer, downscale=downscale))
    lapl_pyr_PET = tuple(pyramid_laplacian(img_PET, max_layer=max_layer, downscale=downscale))

    fused_level = []
    for i in range(len(lapl_pyr_CT)):
        fused = (lapl_pyr_PET[i] + lapl_pyr_CT[i])/2
        fused_level.append(fused)

    orig = fused_level[len(lapl_pyr_CT)-1]
    for i in range(len(lapl_pyr_CT)-1,0,-1):
        up = pyramid_expand(orig, upscale=downscale)
        up = resize(up,fused_level[i-1].shape)
        orig = up + fused_level[i-1]

    return orig


# In[14]:


if __name__ == '__main__':
    
    i = '050-11'  
    img_patient_ini, img_label_ini, positions, s_patient = load_image_and_label(i) 
    img_patient, img_label = preprocessing_image_and_label(img_patient_ini, img_label_ini, positions, s_patient)
    img_patient = np.squeeze(img_patient)
    img_label = np.squeeze(img_label)
    print(img_patient.shape)
    plt.figure(figsize=(20, 10))
    plt.subplot(1,2,1)
    plt.imshow(np.rot90(img_patient[:, int(img_patient.shape[1]/2), :]))
    plt.subplot(1,2,2)
    plt.imshow(np.rot90(img_patient[int(img_patient.shape[0]/2), :, :]))
    plt.figure(figsize=(20, 10))
    plt.subplot(1,2,1)
    plt.imshow(np.rot90(img_label[:, int(img_label.shape[1]/2), :]))
    plt.subplot(1,2,2)
    plt.imshow(np.rot90(img_label[int(img_label.shape[0]/2), :, :]))


# In[15]:


if __name__ == '__main__':
    p = '050-11'
    img_patient = nib.load(os.path.join(base, p,'patient.nii')).get_fdata() 
    img_patient[160:172, :, :] = 0

    img_tete = img_patient[:, :, 400:450]
    img_tete = np.where(img_tete > 8000, 1, 0)
    plt.figure(figsize=(20,10))
    plt.subplot(1, 3, 1)
    plt.imshow(img_tete[int(img_tete.shape[0]/2), :, :])
    plt.subplot(1, 3, 2)
    plt.imshow(img_tete[:, int(img_tete.shape[1]/2), :])
    plt.subplot(1, 3, 3)
    plt.imshow(img_tete[:, :, int(img_tete.shape[2]/2)])

    img_squelette = img_patient > 200
    img_tete_1 = img_patient * 0
    img_tete_1[:, :, 400:450] = img_tete
    img_squelette = np.where(img_tete_1>0, 2, img_squelette)
    plt.figure(figsize=(20,10))
    plt.subplot(1, 3, 1)
    plt.imshow(img_squelette[int(img_squelette.shape[0]/2), :, :])
    plt.subplot(1, 3, 2)
    plt.imshow(img_squelette[:, int(img_squelette.shape[1]/2), :])
    plt.subplot(1, 3, 3)
    plt.imshow(img_squelette[:, :, int(img_squelette.shape[2]/2)])

    img_squelette = nib.Nifti1Image(img_squelette,None)
    path = os.path.join(path_squelette, 'cassiopet_' + p.replace('-','')+'.nii.gz')
    nib.save(img_squelette, path)

