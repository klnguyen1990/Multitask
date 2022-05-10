#!/usr/bin/env python
# coding: utf-8

# In[2]:


from torch import nn, cat


# In[3]:


class nnUnet(nn.Module):
    def __init__(self,trainer):
       super(nnUnet,self).__init__()
       self.conv_blocks_context = nn.Sequential(*list(trainer.conv_blocks_context.children()))
       self.conv_blocks_localization = nn.Sequential(*list(trainer.conv_blocks_localization.children())) 
       self.td = nn.Sequential(*list(trainer.td.children()))
       self.tu = nn.Sequential(*list(trainer.tu.children()))
       self.seg_outputs = nn.Sequential(*list(trainer.seg_outputs.children()))
       
    def forward(self,x):

        #ENCODER
        skips = []
        #seg_outputs = []
        
        for d in range(len(self.conv_blocks_context) - 1):
            x = self.conv_blocks_context[d](x)        
            skips.append(x)            
        
        x = self.conv_blocks_context[-1](x)  
        x_encode = x      
        
        #SEGMENTATION
        for u in range(len(self.tu)):            
            x = self.tu[u](x)
            x = cat((x, skips[-(u + 1)]), dim=1)
            x = self.conv_blocks_localization[u](x)
        #seg_outputs = self.seg_outputs[-1](x)
        #seg_outputs = torch.nn.functional.softmax(seg_outputs,1)    

        return x_encode, x

        #return x


# In[4]:


class nnUnet_UPS(nn.Module):
    def __init__(self,trainer):
       super(nnUnet_UPS,self).__init__()
       self.conv_blocks_context = nn.Sequential(*list(trainer.conv_blocks_context.children()))
       self.conv_blocks_localization = nn.Sequential(*list(trainer.conv_blocks_localization.children())) 
       self.td = nn.Sequential(*list(trainer.td.children()))
       self.tu = nn.Sequential(*list(trainer.tu.children()))
       self.seg_outputs = nn.Sequential(*list(trainer.seg_outputs.children()))
       
    def forward(self,x):

        skips = []
        
        for d in range(len(self.conv_blocks_context) - 1):
            x = self.conv_blocks_context[d](x)        
            skips.append(x)            
        
        x = self.conv_blocks_context[-1](x)  
        x_encode = x       

        return skips, x_encode


# In[5]:


class Decoder3D(nn.Module):
    def __init__(self, in_channels, out_channels, stride=2):
        super(Decoder3D, self).__init__()

        layers = [
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.ConvTranspose3d(out_channels, out_channels, kernel_size=stride, stride=stride)
        ]

        self.decoder = nn.Sequential(*layers)

    def forward(self, x):
        return self.decoder(x)


# In[6]:


class Decoder3D_UPS(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels, scale_factor):
        super(Decoder3D_UPS, self).__init__()

        layers = [
            nn.Conv3d(in_channels, middle_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(middle_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(middle_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=scale_factor)
        ]

        self.decoder = nn.Sequential(*layers)

    def forward(self, x):
        return self.decoder(x)


# In[7]:


class Multitask_UPS_Net(nn.Module):
    def __init__(self, model, drop_encode=0.5):
        super(Multitask_UPS_Net, self).__init__()
        self.nnunet = model
        self.upsample_seg_1 = Decoder3D_UPS(320, 320, 320, scale_factor=(2, 1, 2))
        self.upsample_seg_2 = Decoder3D_UPS(640, 256, 256, scale_factor=2)
        self.upsample_seg_3 = Decoder3D_UPS(512, 128, 128, scale_factor=2)
        self.upsample_seg_4 = Decoder3D_UPS(256, 64, 64, scale_factor=2)
        self.upsample_seg_5 = Decoder3D_UPS(128, 32, 32, scale_factor=2)
        self.seg = nn.Conv3d(32, 1, kernel_size=3, stride=1, padding=1)    

        channel = (320, 320, 256, 128, 64, 32)
        self.upsample_1 = Decoder3D(channel[0], channel[1], channel[1], channel[1], stride=(2, 1, 2))
        self.upsample_2 = Decoder3D(channel[1], channel[2], channel[2], channel[2], stride=2)
        self.upsample_3 = Decoder3D(channel[2], channel[3], channel[3], channel[3], stride=2)
        self.upsample_4 = Decoder3D(channel[3], channel[4], channel[4], channel[4], stride=2)
        self.upsample_5 = Decoder3D(channel[4], channel[5], channel[5], channel[5], stride=2)
        self.reconst = nn.Conv3d(channel[5], 1, kernel_size=3, stride=1, padding=1)    

        self.fc1 = nn.Linear(320, 3)
        self.dropout = nn.Dropout(p=drop_encode)    
        self.avgpool = nn.AdaptiveAvgPool3d((1,1,1)) 

    def forward(self, image):
        
        skips, encode = self.nnunet(image)        

        #RECONSTRUCTION
        reconst = self.upsample_1(encode)
        reconst = self.upsample_2(reconst)
        reconst = self.upsample_3(reconst)
        reconst = self.upsample_4(reconst)
        reconst = self.upsample_5(reconst)
        reconst = self.reconst(reconst)

        #SEGMENTATION UPSAMPLING
        seg = self.upsample_seg_1(encode)        
        seg = cat((seg, skips[4]), dim=1)

        seg = self.upsample_seg_2(seg)
        seg = cat((seg, skips[3]), dim=1)

        seg = self.upsample_seg_3(seg)
        seg = cat((seg, skips[2]), dim=1)

        seg = self.upsample_seg_4(seg)
        seg = cat((seg, skips[1]), dim=1)

        seg = self.upsample_seg_5(seg)
        seg = cat((seg, skips[0]), dim=1)
        
        seg = self.upsample_seg_6(seg)
        seg = self.seg(seg)
        seg = nn.Sigmoid()(seg)

        #CLASSIFICATION
        
        classe = self.avgpool(encode)
        classe = classe.view(classe.size()[0], -1)
        classe = nn.ReLU()(classe)

        classe = self.dropout(classe)
        classe = self.fc1(classe)
        classe = nn.Softmax(dim=1)(classe) 

        return classe, reconst, seg


# In[ ]:


class Multitask_Net(nn.Module):
    def __init__(self, model,drop_encode=0.5):
        super(Multitask_Net,self).__init__()

        self.nnunet = model
        self.seg = nn.Conv3d(32, 1, kernel_size=3, stride=1, padding=1)    

        channel = (320, 320, 256, 128, 64, 32)
        self.upsample_1 = Decoder3D(channel[0], channel[1], stride=(2, 1, 2))
        self.upsample_2 = Decoder3D(channel[1], channel[2], stride=2)
        self.upsample_3 = Decoder3D(channel[2], channel[3], stride=2)
        self.upsample_4 = Decoder3D(channel[3], channel[4], stride=2)
        self.upsample_5 = Decoder3D(channel[4], channel[5], stride=2)
        self.reconst = nn.Conv3d(channel[5], 1, kernel_size=3, stride=1, padding=1)

        self.fc1 = nn.Linear(320, 3)
        self.dropout = nn.Dropout(p=drop_encode)    
        self.avgpool = nn.AdaptiveAvgPool3d((1,1,1)) 

    def forward(self, image):
        
        encode, seg = self.nnunet(image)
        seg = self.seg(seg)
        seg = nn.Sigmoid()(seg)

        #RECONSTRUCTION
        reconst = self.upsample_1(encode)
        reconst = self.upsample_2(reconst)
        reconst = self.upsample_3(reconst)
        reconst = self.upsample_4(reconst)
        reconst = self.upsample_5(reconst)
        reconst = self.reconst(reconst)

        #CLASSIFICATION
        
        classe = self.avgpool(encode)
        #classe = self.avgpool(seg)
        classe = classe.view(classe.size()[0], -1)
        classe = nn.ReLU()(classe)

        classe = self.dropout(classe)
        classe = self.fc1(classe)
        classe = nn.Softmax(dim=1)(classe) 

        return classe, reconst, seg

