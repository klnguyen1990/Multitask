#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import numpy as np
from torch.utils.data import Dataset
import random

os.system('jupyter nbconvert --to python preprocessing_Pytorch.ipynb')
from preprocessing_Pytorch import load_image_and_label, preprocessing_image_and_label, load_image_CT, preprocessing_image, Laplace_pyr_fusion


# In[11]:


class dataloader_PET(Dataset):
    def __init__(self, patient_id, classe, isTransform, scale, sigma, dim, spacing):
        
        self.list = patient_id
        self.classe = classe
        self.isTransform = isTransform     
        self.scale = scale
        self.sigma = sigma
        self.dim = dim
        self.spacing = spacing

    def __len__(self):
        return len(self.classe)

    def __getitem__(self, idx):
        #print('idx =',idx)
        label_classe = self.classe[idx]              

        #Image Transformation 
        if self.isTransform :  
            img_patient_ini, img_label_ini, positions, s_patient = load_image_and_label(self.list[idx])
            scale_random = (random.uniform(self.scale[0], self.scale[1]), random.uniform(self.scale[0], self.scale [1]), 
                        random.uniform(self.scale[0], self.scale[1]))
            sigma_random = (random.uniform(self.sigma[0], self.sigma[1]), random.uniform(self.sigma[0], self.sigma[1]), 
                        random.uniform(self.sigma[0], self.sigma[1]))       
            flip = bool(random.getrandbits(1))
            blur = bool(random.getrandbits(1))
            img_patient, img_label = preprocessing_image_and_label(img=img_patient_ini, img_label=img_label_ini, 
                                                                positions=positions, s_patient=s_patient, dim=self.dim, spacing=self.spacing,  
                                                                scale=scale_random, flip=flip, blur=blur, sigma=sigma_random)
            
        else:
            img_patient_ini, img_label_ini, positions, s_patient = load_image_and_label(self.list[idx])   
            img_patient, img_label = preprocessing_image_and_label(img=img_patient_ini, img_label=img_label_ini, 
                                                                positions=positions, s_patient=s_patient, dim=self.dim, spacing=self.spacing, 
                                                                scale=(1, 1, 1), flip=False, blur=False, sigma=(1, 1, 1)) 
                                                                
        img_patient = np.expand_dims(img_patient, axis = 0)                                           
        img_label = np.expand_dims(img_label, axis = 0)     

        img_patient = np.array(img_patient,dtype=np.float32)
        img_label = np.array(img_label,dtype=np.int8)

        return img_patient, img_label, label_classe, self.list[idx]


# In[12]:


class dataloader_PET_CT_fusion(Dataset):
    def __init__(self, patient_id, classe, isTransform, scale=(1, 1, 1), sigma=(1, 1, 1), dim=(128, 96, 192), spacing=4, min_HU=100):
        
        self.list = patient_id
        self.classe = classe
        self.isTransform = isTransform     
        self.scale = scale
        self.sigma = sigma
        self.dim = dim
        self.spacing_PET = spacing
        self.min_HU = min_HU

    def __len__(self):
        return len(self.classe)

    def __getitem__(self, idx):
        label_classe = self.classe[idx]              

        img_patient_PET_ini, img_label_ini, positions, spacing_patient_PET = load_image_and_label(self.list[idx]) 
        img_patient_CT_ini, spacing_patient_CT = load_image_CT(self.list[idx], min_HU=self.min_HU)    

        spacing_CT = self.spacing_PET * (img_patient_CT_ini.shape[2] * spacing_patient_CT[2])/(img_patient_PET_ini.shape[2] * spacing_patient_PET[2])
        positions_CT_0 = int(positions[0] / img_patient_PET_ini.shape[2] * img_patient_CT_ini.shape[2])
        positions_CT_1 = int(positions[1] / img_patient_PET_ini.shape[2] * img_patient_CT_ini.shape[2])
        positions_CT = [positions_CT_0, positions_CT_1]

        #Image Transformation 
        if self.isTransform :              
            scale_random = (random.uniform(self.scale[0], self.scale[1]), random.uniform(self.scale[0], self.scale [1]), 
                        random.uniform(self.scale[0], self.scale[1]))
            sigma_random = (random.uniform(self.sigma[0], self.sigma[1]), random.uniform(self.sigma[0], self.sigma[1]), 
                        random.uniform(self.sigma[0], self.sigma[1])) 
            flip = bool(random.getrandbits(1))
            blur = bool(random.getrandbits(1))
            
            img_PET, img_label = preprocessing_image_and_label(img=img_patient_PET_ini, img_label=img_label_ini, 
                                                                positions=positions, s_patient=spacing_patient_PET, spacing=self.spacing_PET, dim=self.dim, 
                                                                scale=scale_random, flip=flip, blur=blur, sigma=sigma_random)
            
            img_CT, _ = preprocessing_image(img=img_patient_CT_ini, positions=positions_CT, s_patient=spacing_patient_CT, spacing=spacing_CT, dim=self.dim, 
                                        scale=scale_random, flip=flip, blur=blur, sigma=sigma_random) 

            
        else:
            img_PET, img_label = preprocessing_image_and_label(img=img_patient_PET_ini, img_label=img_label_ini, 
                                                                positions=positions, s_patient=spacing_patient_PET, spacing=self.spacing_PET, dim=self.dim, 
                                                                scale=(1, 1, 1), flip=False, blur=False, sigma=(1, 1, 1))
            
            img_CT, _ = preprocessing_image(img=img_patient_CT_ini, positions=positions_CT, s_patient=spacing_patient_CT, spacing=spacing_CT, dim=self.dim, 
                                        scale=(1, 1, 1), flip=False, blur=False, sigma=(1, 1, 1)) 
                                                                 
        #img_patient = (img_PET + img_CT)/2
        img_corps = np.where(img_PET > 3e-3, 1, 0)
        img_CT = img_CT * img_corps
        img_patient = Laplace_pyr_fusion(img_PET,img_CT,max_layer=5,downscale=2)    
        img_patient = np.expand_dims(img_patient, axis=0)                          
        img_label = np.expand_dims(img_label, axis = 0)         

        img_patient = np.array(img_patient,dtype=np.float32)
        img_label = np.array(img_label,dtype=np.int8)

        return img_patient, img_label, label_classe, self.list[idx]


# In[ ]:


class dataloader_PET_CT_2_channels(Dataset):
    def __init__(self, patient_id, classe, isTransform, scale=(1, 1, 1), sigma=(1, 1, 1), dim=(128, 96, 192), spacing=4, min_HU=100):
        
        self.list = patient_id
        self.classe = classe
        self.isTransform = isTransform     
        self.scale = scale
        self.sigma = sigma
        self.dim = dim
        self.spacing_PET = spacing
        self.min_HU = min_HU

    def __len__(self):
        return len(self.classe)

    def __getitem__(self, idx):
        label_classe = self.classe[idx]              

        img_patient_PET_ini, img_label_ini, positions, spacing_patient_PET = load_image_and_label(self.list[idx]) 
        img_patient_CT_ini, spacing_patient_CT = load_image_CT(self.list[idx], min_HU=self.min_HU)    

        spacing_CT = self.spacing_PET * (img_patient_CT_ini.shape[2] * spacing_patient_CT[2])/(img_patient_PET_ini.shape[2] * spacing_patient_PET[2])
        positions_CT_0 = int(positions[0] / img_patient_PET_ini.shape[2] * img_patient_CT_ini.shape[2])
        positions_CT_1 = int(positions[1] / img_patient_PET_ini.shape[2] * img_patient_CT_ini.shape[2])
        positions_CT = [positions_CT_0, positions_CT_1]

        #Image Transformation 
        if self.isTransform :              
            scale_random = (random.uniform(self.scale[0], self.scale[1]), random.uniform(self.scale[0], self.scale [1]), 
                        random.uniform(self.scale[0], self.scale[1]))
            sigma_random = (random.uniform(self.sigma[0], self.sigma[1]), random.uniform(self.sigma[0], self.sigma[1]), 
                        random.uniform(self.sigma[0], self.sigma[1])) 
            flip = bool(random.getrandbits(1))
            blur = bool(random.getrandbits(1))
            
            img_PET, img_label = preprocessing_image_and_label(img=img_patient_PET_ini, img_label=img_label_ini, 
                                                                positions=positions, s_patient=spacing_patient_PET, spacing=self.spacing_PET, dim=self.dim, 
                                                                scale=scale_random, flip=flip, blur=blur, sigma=sigma_random)
            
            img_CT, _ = preprocessing_image(img=img_patient_CT_ini, positions=positions_CT, s_patient=spacing_patient_CT, spacing=spacing_CT, dim=self.dim, 
                                        scale=scale_random, flip=flip, blur=blur, sigma=sigma_random) 

            
        else:
            img_PET, img_label = preprocessing_image_and_label(img=img_patient_PET_ini, img_label=img_label_ini, 
                                                                positions=positions, s_patient=spacing_patient_PET, spacing=self.spacing_PET, dim=self.dim, 
                                                                scale=(1, 1, 1), flip=False, blur=False, sigma=(1, 1, 1))
            
            img_CT, _ = preprocessing_image(img=img_patient_CT_ini, positions=positions_CT, s_patient=spacing_patient_CT, spacing=spacing_CT, dim=self.dim, 
                                        scale=(1, 1, 1), flip=False, blur=False, sigma=(1, 1, 1)) 

        img_CT = img_CT * np.where(img_PET > 1e-3, 1, 0)

        img_PET = np.expand_dims(img_PET, axis=0)      
        img_CT = np.expand_dims(img_CT, axis=0)    
        img_patient = np.concatenate((img_PET, img_CT), axis=0, dtype=np.float32)     

        img_label = np.expand_dims(img_label, axis = 0)         
        img_label = np.array(img_label,dtype=np.int8)

        return img_patient, img_label, label_classe, self.list[idx]


# In[ ]:


class dataloader_PET_CT_2paths(Dataset):
    def __init__(self, patient_id, classe, isTransform, scale=(1, 1, 1), sigma=(1, 1, 1), dim=(128, 96, 192), spacing=4, min_HU=100):
        
        self.list = patient_id
        self.classe = classe
        self.isTransform = isTransform     
        self.scale = scale
        self.sigma = sigma
        self.dim = dim
        self.spacing_PET = spacing
        self.min_HU = min_HU

    def __len__(self):
        return len(self.classe)

    def __getitem__(self, idx):
        label_classe = self.classe[idx]              

        img_patient_PET_ini, img_label_ini, positions, spacing_patient_PET = load_image_and_label(self.list[idx]) 
        img_patient_CT_ini, spacing_patient_CT = load_image_CT(self.list[idx], min_HU=self.min_HU)    

        spacing_CT = self.spacing_PET * (img_patient_CT_ini.shape[2] * spacing_patient_CT[2])/(img_patient_PET_ini.shape[2] * spacing_patient_PET[2])
        positions_CT_0 = int(positions[0] / img_patient_PET_ini.shape[2] * img_patient_CT_ini.shape[2])
        positions_CT_1 = int(positions[1] / img_patient_PET_ini.shape[2] * img_patient_CT_ini.shape[2])
        positions_CT = [positions_CT_0, positions_CT_1]

        #Image Transformation 
        if self.isTransform :              
            scale_random = (random.uniform(self.scale[0], self.scale[1]), random.uniform(self.scale[0], self.scale [1]), 
                        random.uniform(self.scale[0], self.scale[1]))
            sigma_random = (random.uniform(self.sigma[0], self.sigma[1]), random.uniform(self.sigma[0], self.sigma[1]), 
                        random.uniform(self.sigma[0], self.sigma[1])) 
            flip = bool(random.getrandbits(1))
            blur = bool(random.getrandbits(1))
            
            img_PET, img_label = preprocessing_image_and_label(img=img_patient_PET_ini, img_label=img_label_ini, 
                                                                positions=positions, s_patient=spacing_patient_PET, spacing=self.spacing_PET, dim=self.dim, 
                                                                scale=scale_random, flip=flip, blur=blur, sigma=sigma_random)
            
            img_CT, _ = preprocessing_image(img=img_patient_CT_ini, positions=positions_CT, s_patient=spacing_patient_CT, spacing=spacing_CT, dim=self.dim, 
                                        scale=scale_random, flip=flip, blur=blur, sigma=sigma_random) 

            
        else:
            img_PET, img_label = preprocessing_image_and_label(img=img_patient_PET_ini, img_label=img_label_ini, 
                                                                positions=positions, s_patient=spacing_patient_PET, spacing=self.spacing_PET, dim=self.dim, 
                                                                scale=(1, 1, 1), flip=False, blur=False, sigma=(1, 1, 1))
            
            img_CT, _ = preprocessing_image(img=img_patient_CT_ini, positions=positions_CT, s_patient=spacing_patient_CT, spacing=spacing_CT, dim=self.dim, 
                                        scale=(1, 1, 1), flip=False, blur=False, sigma=(1, 1, 1)) 

        #img_CT = img_CT * np.where(img_PET > 3e-3, 1, 0)

        img_PET = np.expand_dims(img_PET, axis=0)  
        img_PET = np.array(img_PET,dtype=np.float32) 

        img_CT = np.expand_dims(img_CT, axis=0)     
        img_CT = np.array(img_CT,dtype=np.float32) 

        img_label = np.expand_dims(img_label, axis = 0)         
        img_label = np.array(img_label,dtype=np.int8)

        return img_PET, img_CT, img_label, label_classe, self.list[idx]

