from __future__ import print_function, division

import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torchvision
from torchvision import datasets, models, transforms, utils
from torch.utils.data import Dataset, DataLoader
from . import lr_scheduler 
import matplotlib.pyplot as plt
from skimage import io, transform, color
import PIL
from PIL import Image
import numpy as np
import pandas as pd
import cv2
import time
import os
import copy
from model_summary import *
import pretrainedmodels
import tqdm
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")


class gain_dataset(Dataset):

    def __init__(self, csv_file, root_dir, transform=None):
     
        self.data_frame = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform
        self.mask_dir = self.root_dir.replace('CBIS-DDSM_classification','masks')
            
    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir,self.data_frame.iloc[idx]['name'])
        #image = io.imread(img_name)
        #image = Image.fromarray(image)
        image = Image.open(img_name)
            
        label = self.data_frame.iloc[idx]['category']
        
        mask_name = os.path.join(self.mask_dir,self.data_frame.iloc[idx]['name'].replace('.j','_mask.j'))
        mask = io.imread(mask_name)
        mask = np.array([mask,mask,mask]).transpose((1,2,0))
        mask = Image.fromarray(mask)
        
        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)

        return {'image':image,'category':label,'mask':mask}
    
    def get_x_y(self,idx):
        img_name = os.path.join(self.root_dir,self.data_frame.iloc[idx]['name'])
        image = io.imread(img_name)
        label = self.data_frame.iloc[idx]['category']
        
        return image,label
    
    def get_x_name(self,idx):
        img_name = os.path.join(self.root_dir,self.data_frame.iloc[idx]['name'])
        return img_name
    
def get_classification_dataloader(data_dir, train_csv_path, image_size, img_mean, img_std, batch_size=1):

    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize(image_size),#row to column ratio should be 1.69
            transforms.RandomHorizontalFlip(0.5),
            transforms.RandomVerticalFlip(0.5),
            transforms.RandomRotation(15),
            transforms.ToTensor(),
            #transforms.Normalize([0.223, 0.231, 0.243], [0.266, 0.270, 0.274])
            transforms.Normalize(img_mean,img_std)
        ]),
        'valid': transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
            #transforms.Normalize([0.223, 0.231, 0.243], [0.266, 0.270, 0.274])
            transforms.Normalize(img_mean,img_std)
        ]),
        'test': transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
            #transforms.Normalize([0.223, 0.231, 0.243], [0.266, 0.270, 0.274])
            transforms.Normalize(img_mean,img_std)
        ])
    }
    
    image_datasets = {}
    dataloaders = {}
    dataset_sizes = {}
    
    for x in ['train', 'valid', 'test']:
        image_datasets[x] = gain_dataset(train_csv_path.replace('train',x),root_dir=data_dir,transform=data_transforms[x])
        
        if x!= 'test':
            dataloaders[x] = torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size,shuffle=True, num_workers=8)
        else:
            dataloaders[x] = torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size,shuffle=False, num_workers=8)
        dataset_sizes[x] = len(image_datasets[x])
        
    device = torch.device("cuda:0")
     
    return dataloaders,dataset_sizes,image_datasets,device

def denorm_img(img_ten,img_mean,img_std):

    bz,nc,h,w = img_ten.shape
    output = []
    img_num = img_ten.numpy()
    #img_num = img_ten
    
    for i in range(bz):
        
        #import pdb;pdb.set_trace()
        img = img_ten[i].numpy().squeeze()
        #img = img_ten[i].squeeze()
        
        img[0,:,:] = img[0,:,:]*img_std[0]
        img[1,:,:] = img[1,:,:]*img_std[1]
        img[2,:,:] = img[2,:,:]*img_std[2]

        img[0,:,:] = img[0,:,:] + img_mean[0]
        img[1,:,:] = img[1,:,:] + img_mean[1]
        img[2,:,:] = img[2,:,:] + img_mean[2]
        
        output.append(img)
    
    output = np.array(output)
        
    return output

def dice(pred, targs):
    
    max_pred = pred.max()
    pred[pred>0.8*max_pred] = 1
    pred[pred<0.8*max_pred] = 0
    
    targs = (targs>0)#.float()
    pred = (pred>0)#.float()
    return 2. * (pred*targs).sum() / (pred+targs).sum()

# ir2 = pretrainedmodels.__dict__['inceptionresnetv2'](num_classes=1000, pretrained='imagenet')
# ir1 = nn.Sequential(*list(ir2.children())[:-6])
# summary(ir1.cuda(),(3,540,320))

class depthwise_block(nn.Module):
    def __init__(self, in_c, out_c, stride=1):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(in_c)
        self.conv1 = nn.Conv2d(in_c, in_c, kernel_size=3, 
                               stride=stride, padding=1,
                               groups=in_c, bias=False)
        self.conv2 = nn.Conv2d(in_c, out_c, kernel_size=1,
                              stride=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_c)
    
    def forward(self, inp):
        out = F.relu(self.bn1(self.conv1(inp)))
        out = F.relu(self.bn2(self.conv2(out)))
        return out

class mblnetv1(nn.Module):
    def __init__(self, block, inc_list, inc_scale, num_blocks_list, stride_list, num_classes):
        super().__init__()
        self.num_blocks = len(num_blocks_list)
        inc_list1 = [o//inc_scale for o in inc_list]
        self.in_planes = inc_list1[0]
        self.conv1 = nn.Conv2d(3, self.in_planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.in_planes)
        
        lyrs = []
        for inc, nb, strl in zip(inc_list1[1:], num_blocks_list, stride_list):
            lyrs.append(self._make_layer(block, inc, nb, strl))
            
        self.lyrs = nn.Sequential(*lyrs)
        self.linear = nn.Linear(inc_list1[-1], num_classes)
        self.gap = nn.AdaptiveAvgPool2d((1,1))#nn.AvgPool2d((14,14),stride=1)

    
    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes
        return nn.Sequential(*layers)
    
    def forward(self, inp):
        out = F.relu(self.bn1(self.conv1(inp)))
        out = self.lyrs(out)
        out = self.gap(out)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return F.log_softmax(out, dim=-1)

#md_mbl = mblnetv1(depthwise_block, 
#              inc_list=[64, 64, 128, 256, 512], 
#              inc_scale = 1, 
#              num_blocks_list=[2, 2, 2, 2], 
#              stride_list=[1, 2, 2, 2], 
#              num_classes=2)
#summary(md_mbl.cuda(),(3,320,150))

#import pdb;pdb.set_trace()

# ir = ir2_cam(ir1)

# ir.base[-1][-1].relu

class SaveFeatures:
    def __init__(self, m):
        self.handle = m.register_forward_hook(self.hook_fn)
    def hook_fn(self, m, inp, outp):
        self.features = outp
    def remove(self):
        self.handle.remove()

def returnCAM(feature_conv, weight_softmax, class_idx, output_shape):
    # generate the class activation maps upsample to 256x256
    size_upsample = output_shape
    bz, nc, h, w = feature_conv.shape
    output_cam = []
    for i in range(bz):
        #import pdb;pdb.set_trace()
        idx = class_idx[0][i]
        cam = weight_softmax[idx].dot(feature_conv[i].reshape((nc, h*w)))
        cam = cam.reshape(h, w)
        #cam = cam - np.min(cam)
        #cam_img = cam / np.max(cam)
        #print('cam img shape',cam_img.shape)
        cam_img = cv2.resize(cam,(size_upsample[0],size_upsample[1]))
        #cam_img[cam_img<0] = 0
        output_cam.append(cam_img)
    output_cam = np.array(output_cam)
    
    final_output_cam = np.zeros((bz,3,size_upsample[1],size_upsample[0]))
    final_output_cam[:,0,:,:] = output_cam
    final_output_cam[:,1,:,:] = output_cam
    final_output_cam[:,2,:,:] = output_cam
    
    return final_output_cam

def gain_train_model(model, dataloaders, dataset_sizes, device, classification_loss, optimizer, scheduler, img_mean, img_std, num_epochs=25):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1),flush=True)
        print('-' * 10,flush=True)
        
        #import pdb;pdb.set_trace()

        # Each epoch has a training and validation phase
        for phase in ['train', 'valid']:
            if phase == 'train':
                scheduler.step()
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0
            #running_dice = 0

            #tqdm bar
            pbar = tqdm(total=dataset_sizes[phase])            
            # Iterate over data.
            for sampled_batch in dataloaders[phase]:
                
                inputs = sampled_batch['image']
                labels = sampled_batch['category']
                #mask = sampled_batch['mask']
                
                inputs = inputs.float().to(device)
                labels = labels.long().to(device)
                #print('labels shape',labels.shape)
                #print(mask.shape)
                #mask = denorm_img(mask,img_mean,img_std).squeeze()
                #print('mask shape',mask.shape)
                #mask[mask>0.2] = 1
                #mask[mask<0.2] = 0
                
                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    
                    #CAM computation need to take place in eval mode
#                     if phase == 'train':
#                         print('yaay')
#                         model.eval()
                    
                    #Save features for the forward pass
                    #sfs = SaveFeatures(model.base[-1][-1].relu)
                    outputs = torch.exp(model(inputs))
                    #print('outputs shape',outputs.shape)
                    #sfs.remove()
                    
                    #print(sfs.features.requires_grad)
                    #Get the features obtained after forward pass
                    #features = sfs.features.detach().cpu().numpy()
                    #print('Features shape',features.shape)
                    
                    #This will get the prediction for the sample
                    _, preds = torch.max(outputs, 1)
                    
                    #Get the weights of the model
                    #params = list(model.parameters())
                    #weight_softmax = np.squeeze(params[-2].data.cpu().numpy())
                    
                    #Get the CAM
                    #cam_orig = np.array(returnCAM(features,weight_softmax,[preds],(inputs.size(-1),inputs.size(-2))))
                    #print('cam orig shape',cam_orig.shape)
                    #cam_orig = F.relu(torch.from_numpy(cam_orig))
                    loss = classification_loss(outputs, labels)
                    
                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
                #running_dice += dice(cam_orig.numpy(),mask)
                
                pbar.update(inputs.shape[0])
            pbar.close()


            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]
            #epoch_dice = running_dice / dataset_sizes[phase]
            
            #import pdb;pdb.set_trace()
            #torch.save(model.state_dict(),'inc_res_cam_'+str(epoch_acc.cpu().numpy())+'_acc.pt')
            
 
            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'valid' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                torch.save(model.state_dict(),'mbn_cam_last_acc.pt')

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    torch.save(model.state_dict(),'inc_res_cam.pt')
    return model

data_dir = '../Data/CBIS-DDSM_classification_1/'
train_csv = '../CSV/gain_train.csv'
#image_size = (640,384)
image_size = (320,192)
num_classes = 2
batch_size = 8
num_epochs = 50
img_mean = [0.223, 0.231, 0.243]
img_std = [0.266, 0.270, 0.274]

dataloaders,dataset_sizes,dataset,device = get_classification_dataloader(data_dir,train_csv,image_size,img_mean,img_std,batch_size)

#ir2 = pretrainedmodels.__dict__['inceptionresnetv2'](num_classes=1000, pretrained='imagenet')
#ir1 = nn.Sequential(*list(ir2.children())[:-6])
#inc_res_2 = ir2_cam(ir1).to(device)

md_mbl = mblnetv1(depthwise_block,
              inc_list=[64, 64, 128, 256, 512],
              inc_scale = 1,
              num_blocks_list=[2, 2, 2, 2],
              stride_list=[1, 2, 2, 2],
              num_classes=2)
md_mbl = md_mbl.to(device)
model = md_mbl

summary(model.cuda(),(3,image_size[0],image_size[1]))

classification_loss = nn.CrossEntropyLoss()
params = model.parameters()
# Observe that all parameters are being optimized
#optimizer_ft = optim.SGD(inc_res_2.parameters(), lr=0.001, momentum=0.9)
optimizer_ft = optim.Adam(params, lr=0.01, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
# Decay LR by a factor of 0.1 every 7 epochs
lr_sched = lr_scheduler.StepLR(optimizer_ft, step_size=10, gamma=0.1)
#lr_sched = lr_scheduler.CosineAnnealingLR(optimizer_ft, 5, eta_min=0, last_epoch=-1)
#lr_sched = lr_scheduler.ExponentialLR(optimizer_ft, 0.1, last_epoch=-1)

model_ft = gain_train_model(model, dataloaders, dataset_sizes, device, classification_loss, optimizer_ft, lr_sched, img_mean, img_std, num_epochs)
