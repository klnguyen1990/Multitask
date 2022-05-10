#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import numpy as np
import matplotlib.pyplot as plt
import nrrd
from scipy.ndimage.interpolation import rotate
from preprocessing_Pytorch import load_image_and_label, preprocessing_image


# In[2]:


base_nrrd = '/home/nguyen-k/Bureau/segCassiopet/dcm_test'
max_SUV = 0.4


# In[3]:


def nb_lesion_focale(p) :
    nb = 0
    if os.path.isfile(os.path.join(base_nrrd, p, 'majorityLabel1.nrrd')) : 
        list = os.listdir(os.path.join(base_nrrd, p))
        for i in list :
            if 'nrrd' in i :
                nb += 1
    return nb


# In[4]:


def count_nb_lesion(patient, image_seg, dim, spacing) : 
    smooth = 1e-6
    nb_lesion = nb_lesion_focale(patient)
    nb_lesion_pred = 0
    _, _, positions, s_patient = load_image_and_label(patient)         
        
    for l in range(1, nb_lesion+1) :
        img_nrrd, _ = nrrd.read(os.path.join(base_nrrd, patient,'majorityLabel'+str(l)+'.nrrd')) 
        img_nrrd, _ = preprocessing_image(img=img_nrrd, positions=positions, s_patient=s_patient,flip=False,blur=False, 
                                            scale=(1, 1, 1), sigma=(1, 1, 1), dim=dim, spacing=spacing)
        intersection_ct = image_seg * img_nrrd
        if np.sum(intersection_ct)/(np.sum(img_nrrd)+smooth) > 0.5 :
            nb_lesion_pred +=1
            
    return nb_lesion, nb_lesion_pred


# In[5]:


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


# In[6]:


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# In[7]:


def create_mip_from_3D(img_data, angle=0):

    img_data = np.array(img_data, dtype=np.float32)
    img_data+=1e-5    
    vol_angle= rotate(img_data,angle)

    mip=np.amax(vol_angle,axis=1)
    mip-=1e-5
    mip[mip<1e-5]=0
    mip=np.flipud(mip.T)

    return mip


# In[8]:


def plot_image_mip(img, angle, text, x, y, z) : 
    mip = create_mip_from_3D(img, angle=angle)    
    ax = plt.subplot(x, y, z)
    plt.imshow(mip, cmap='gray')
    ax.set_yticklabels([])
    ax.set_xticklabels([])
    plt.title(text)
    
    return mip


# In[ ]:


def plot_graphic(fold, nb_train, train_c1, train_c2, train_c3, nb_val, val_c1, val_c2, val_c3, dim, spacing, scale, sigma, nb_parametres, score_weight, 
                drop_encode, weight_decay, l1_fc1, l2_fc1, learning_rate, lr, patience, epoch, train_micro_f1_score, val_micro_f1_score, val_macro_f1_score, 
                val_weighted_f1_score, train_loss, val_loss, train_loss_class, val_loss_class, train_dice, val_dice, train_loss_seg, val_loss_seg): 
    plt.subplot(2, 3, 1)
    plt.text(0, 0.1, ' DATA - FOLD ' + str(fold) +                     '\n Train data = ' + str(nb_train) + ' - C1 - C2 - C3 = ' + str(train_c1) + ' - ' + str(train_c2) + ' - ' + str(train_c3) +                     '\n Validation data = ' + str(nb_val) + ' - C1 - C2 - C3 = ' + str(val_c1) + ' - ' + str(val_c2) + ' - ' + str(val_c3) +                     '\n Image dimensions = ' + str(dim[0]) + ' x '+ str(dim[1]) + ' x ' + str(dim[2]) + ' - Spacing = ' + str(spacing) +                     '\n ' +                     '\n IMAGE GENERATOR' +                     '\n Scale = ' + str(scale[0]) + ' - ' + str(scale[1]) +                     '\n Gaussian blur = Random - Sigma = '  + str(sigma[0]) + ' - ' + str(sigma[1]) +                     '\n ' +                     '\n MODELE CLASSIFICATION - SEGMENTATION CT - RECONSTRUCTION' +                     '\n Nombre de paramètres trainables = ' + str(round(nb_parametres/1e6, 2)) + ' M' +                     '\n Score : Segmentation : Reconstruction = ' + str(score_weight) + ' : 1 : 1' +                     '\n Drop encode = ' + str(drop_encode) +                     '\n Optimiseur Adam - Weight decay = ' + str(weight_decay) +                     '\n Fc1 Regulateur : L1 = ' + str(l1_fc1) + ' - L2 = ' + str(l2_fc1) +                     '\n Learning rate = ' + str(learning_rate) + ' - Learning rate scheduler = ' + str(lr) + ' - Patience = ' + str(patience))

    plt.subplot(2, 3, 2)
    plt.plot(range(epoch),train_micro_f1_score[0:epoch], linestyle='solid', color='green', label='Train F1 score (Accuracy)')
    plt.plot(range(epoch), val_micro_f1_score[0:epoch], linestyle='solid', color='black', label='Val micro F1 score (Accuracy)')
    plt.plot(range(epoch), val_macro_f1_score[0:epoch], linestyle='dashed', color='black', label='Val macro F1 score')
    plt.plot(range(epoch), val_weighted_f1_score[0:epoch], linestyle='dotted', color='black', label='Val weighted F1 score')
    plt.legend(loc='upper left')
    plt.xlabel("Epoch")          
                                
    plt.subplot(2, 3, 4)
    plt.plot(range(epoch),train_loss[0:epoch]/nb_train,linestyle='solid', color='blue', label='Train Loss Total')
    plt.plot(range(epoch),val_loss[0:epoch]/nb_val, linestyle='solid',color='red', label='Val Loss Total')
    plt.legend(loc='upper right')
    plt.xlabel("Epoch")  

    plt.subplot(2, 3, 5)    
    plt.plot(range(epoch),train_loss_class[0:epoch]/nb_train,linestyle='dotted', color='blue', label='Train Loss Class')
    plt.plot(range(epoch),val_loss_class[0:epoch]/nb_val, linestyle='dotted',color='red', label='Val Loss Class')  
    plt.legend(loc='upper right')
    plt.xlabel("Epoch")  

    plt.subplot(2, 3, 3)    
    plt.plot(range(epoch),train_dice[0:epoch], linestyle='solid',color='blue', label='Train Dice')  
    plt.plot(range(epoch),val_dice[0:epoch], linestyle='solid',color='red', label='Val Dice')  
    plt.legend(loc='lower right')
    plt.xlabel("Epoch")  

    plt.subplot(2, 3, 6)    
    plt.plot(range(epoch),train_loss_seg[0:epoch]/nb_train,linestyle='dashed', color='blue', label='Train Loss Seg')
    plt.plot(range(epoch),val_loss_seg[0:epoch]/nb_val, linestyle='dashed',color='red', label='Val Loss Seg')  
    plt.legend(loc='upper right')
    plt.xlabel("Epoch")  


# In[9]:


def plot_image(id, image_1, image_2, image_label, image_seg, labels, predictions, threshold, dice, nb_lesion, nb_lesion_pred, x, y) : 

    image_1 = np.where(image_1 > 0.2, 0.2, image_1)
    image_2 = np.where(image_2 > 0.2, 0.2, image_2)

    text = 'Patient '+ id[0] + ' - PET'
    mip_PET_1 = plot_image_mip(image_1, 0, text, x, y, 1)
    text = 'Classe = '+ str(labels[0].cpu().numpy())+' - Prediction = '+str(predictions[0].cpu().numpy())
    mip_PET_2 = plot_image_mip(image_1, 90, text, x, y, 6)

    text = 'Reconstruction'
    plot_image_mip(image_2, 0, text, x, y, 2)
    plot_image_mip(image_2, 90, '', x, y, 7)

    text = 'Label'
    mip_label_1 = plot_image_mip(image_label, 0, text, x, y, 3)
    text = 'Nombre de lésions : '+str(nb_lesion)
    mip_label_2 = plot_image_mip(image_label, 90, text, x, y, 8)
    
    text = 'Segmentation - Threshold '+str(threshold)
    mip_seg_1 = plot_image_mip(image_seg, 0, text, x, y, 4)
    text = 'Nombre de lésions détectées : '+str(nb_lesion_pred)
    mip_seg_2 = plot_image_mip(image_seg, 90, text, x, y, 9)

    tp_1 = mip_label_1 * mip_seg_1
    tp_2 = mip_label_2 * mip_seg_2

    fp_1 =  mip_seg_1 - tp_1
    fp_2 =  mip_seg_2 - tp_2

    fn_1 = mip_label_1 - tp_1
    fn_2 = mip_label_2 - tp_2

    seg1 = 2*tp_1 + fp_1 + 3*fn_1
    seg2 = 2*tp_2 + fp_2 + 3*fn_2

    seg1 = np.ma.masked_where(seg1 == 0, seg1)
    seg2 = np.ma.masked_where(seg2 == 0, seg2)

    ax = plt.subplot(x, y, 5)
    plt.imshow(mip_PET_1, cmap='gray')
    plt.imshow(seg1, cmap='jet', alpha=0.7)
    plt.title('Dice = ' + str(np.round(dice, 2)))
    ax.set_yticklabels([])
    ax.set_xticklabels([])

    ax = plt.subplot(x, y, 10)
    plt.imshow(mip_PET_2, cmap='gray')
    plt.imshow(seg2, cmap='jet', alpha=0.7)
    ax.set_yticklabels([])
    ax.set_xticklabels([])


# In[10]:


def data_balance(list_train_ini, train_label_classe_ini) : 
    train_c1 = train_label_classe_ini.tolist().count(0)
    train_c2 = train_label_classe_ini.tolist().count(1)
    train_c3 = train_label_classe_ini.tolist().count(2)

    list_class = [train_c1,train_c2,train_c3]
    ref_index = np.argmax(list_class)
    ratio = [0,0,0]

    for i in range(len(list_class)):
        ratio[i] = 1*list_class[ref_index]/list_class[i]
        if ratio[i] <= 1 and ratio[i] > 0 :
            ratio[i] = 1
        else:
            ratio[i] = round(ratio[i])
            
    list_train = []
    train_label_classe = []
    for i in range(len(list_train_ini)):
        nb = ratio[int(train_label_classe_ini[i])]
        for j in range(nb) : 
            list_train.append(list_train_ini[i])
            train_label_classe.append(train_label_classe_ini[i])
    return list_train, train_label_classe


# In[ ]:


def get_list(path, fold, dir_p) :

    path_train_val = os.path.join(path, 'Fold'+str(fold))
    list_train_ini = np.load(path_train_val+'/list_train.npy')
    train_label_classe_ini = np.load(path_train_val+'/train_label_classe.npy')
    list_val = list(np.load(path_train_val+'/list_val.npy'))
    val_label_classe = list(np.load(path_train_val+'/val_label_classe.npy'))

    list_train, train_label_classe = data_balance(list_train_ini, train_label_classe_ini)   

    np.save(dir_p + '/list_train.npy', list_train)
    np.save(dir_p + '/train_label_classe.npy', train_label_classe)
    np.save(dir_p + '/list_val.npy', list_val)
    np.save(dir_p + '/val_label_classe.npy', val_label_classe)   

    return list_train, train_label_classe, list_val, val_label_classe


# In[ ]:


def save_resultat(dir_p, train_loss, val_loss, train_loss_class, val_loss_class, train_loss_seg, val_loss_seg, train_loss_reconst, val_loss_reconst, 
                    train_micro_f1_score, val_micro_f1_score, val_macro_f1_score, val_weighted_f1_score, train_dice, val_dice) :     
    np.save(dir_p+'/train_loss.npy', train_loss)
    np.save(dir_p+'/val_loss.npy', val_loss)
    np.save(dir_p+'/train_loss_class.npy', train_loss_class)
    np.save(dir_p+'/val_loss_class.npy', val_loss_class)
    np.save(dir_p+'/train_loss_seg.npy', train_loss_seg)
    np.save(dir_p+'/val_loss_seg.npy', val_loss_seg)
    np.save(dir_p+'/train_loss_reconst.npy', train_loss_reconst)
    np.save(dir_p+'/val_loss_reconst.npy', val_loss_reconst)
    np.save(dir_p+'/train_micro_f1_score.npy', train_micro_f1_score)
    np.save(dir_p+'/val_micro_f1_score.npy', val_micro_f1_score)
    np.save(dir_p+'/val_macro_f1_score.npy', val_macro_f1_score)
    np.save(dir_p+'/val_weighted_f1_score.npy', val_weighted_f1_score)
    np.save(dir_p+'/train_dice.npy', train_dice)
    np.save(dir_p+'/val_dice.npy', val_dice)

