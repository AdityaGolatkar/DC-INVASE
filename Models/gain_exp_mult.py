from __future__ import print_function, division

import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torchvision
from torchvision import datasets, models, transforms, utils
from torch.utils.data import Dataset, DataLoader
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

        return {'image':image,'category':label,'mask':mask, 'name':img_name}
    
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
            transforms.RandomRotation(30),
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
            dataloaders[x] = torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size,shuffle=True, num_workers=4)
        else:
            dataloaders[x] = torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size,shuffle=False, num_workers=4)
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
    targs = (targs>0)#.float()
    pred = (pred>0)#.float()
    return 2. * (pred*targs).sum() / (pred+targs).sum()

#ir2 = pretrainedmodels.__dict__['inceptionresnetv2'](num_classes=1000, pretrained='imagenet')
#ir1 = nn.Sequential(*list(ir2.children())[:-1])
#summary(ir1.cuda(),(3,540,320))

#vggnet = models.vgg11_bn(pretrained=True)
#vgg_conv = nn.Sequential(*list(vggnet.children())[0][:-1])

class vgg_gain(nn.Module):
    def __init__(self,vgg_base):
        super().__init__()
        self.vgg_base = vgg_base
        self.gap = nn.AdaptiveAvgPool2d((1,1))#nn.AvgPool2d((14,14),stride=1)
        self.fc = nn.Linear(512,2)


    def forward(self, x):
        x = self.vgg_base(x)
        x = self.gap(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return F.log_softmax(x)


#v = vgg_gain(vgg_conv)
#v.vgg_base[28]

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
        cam_img[cam_img<0] = 0
        output_cam.append(cam_img)
    output_cam = np.array(output_cam)
    
    final_output_cam = np.zeros((bz,3,size_upsample[1],size_upsample[0]))
    final_output_cam[:,0,:,:] = output_cam
    final_output_cam[:,1,:,:] = output_cam
    final_output_cam[:,2,:,:] = output_cam
    
    return final_output_cam

def mining_loss(mining_output,labels):
    mining_output = mining_output
    min_loss = 0
    #import pdb;pdb.set_trace()
    for i in range(mining_output.shape[0]):
        min_loss+=mining_output[i][labels[i]]
    
    min_loss = min_loss/mining_output.shape[0]
    return min_loss

def gain_train_model(model, dataloaders, dataset_sizes, device, stream_loss, optimizer, scheduler, img_mean, img_std, num_epochs=25, sigma=0, w=1, alpha=1):
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
            running_dice = 0

            #tqdm bar
            pbar = tqdm(total=dataset_sizes[phase])            
            # Iterate over data.
            for sampled_batch in dataloaders[phase]:
                
                inputs = sampled_batch['image']
                labels = sampled_batch['category']
                mask = sampled_batch['mask']
                
                inputs = inputs.float().to(device)
                labels = labels.long().to(device)
                #print('labels shape',labels.shape)
                #print(mask.shape)
                mask = denorm_img(mask,img_mean,img_std).squeeze()
                #print('mask shape',mask.shape)
                mask[mask>0.1] = 1
                mask[mask<0.1] = 0
                
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
                    sfs = SaveFeatures(model.vgg_base[27])
                    outputs = torch.exp(model(inputs))
                    #print('outputs shape',outputs.shape)
                    sfs.remove()
                    
                    #print(sfs.features.requires_grad)
                    #Get the features obtained after forward pass
                    features = sfs.features.detach().cpu().numpy()
                    #print('Features shape',features.shape)
                    
                    #This will get the prediction for the sample
                    _, preds = torch.max(outputs, 1)
                    
                    #Get the weights of the model
                    params = list(model.parameters())
                    weight_softmax = np.squeeze(params[-2].data.cpu().numpy())
                    
                    #Get the CAM
                    cam_orig = np.array(returnCAM(features,weight_softmax,[preds],(inputs.size(-1),inputs.size(-2))))
                    #print('cam orig shape',cam_orig.shape)
                    
                    #Convert cam to tensor
                    cam = torch.from_numpy(cam_orig).float().to(device)
                    #print('cam shape',cam.shape)
                    #import pdb
                    #pdb.set_trace()
                    
                    #T(A) as defined in the paper
                    t_cam = F.sigmoid(w*(cam - sigma))
                    #print('t cam shape',t_cam.shape)
                    #print('inputs shape',inputs.shape)
                    
                    #Mining input
                    mining_input = inputs - t_cam*inputs
                    #print('mining_input shape',mining_input.shape)
                    
                    #Compute the mining output
                    mining_output = torch.exp(model(mining_input))
                    
#                     #Convert to training mode
#                     if phase == 'train':
#                         model.train()
                    
                    #Compute the stream loss
                    loss_stream = stream_loss(outputs, labels)
                    
                    #print('labels shape',labels.shape)
                    #Compute the mining loss
                    loss_mining = mining_loss(mining_output,labels)
                    
                    #import pdb;pdb.set_trace()

                    #Total loss is the sum of the two loss
                    loss = loss_stream + alpha*loss_mining
                    
                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
                running_dice += dice(cam_orig,mask)
                
                pbar.update(inputs.shape[0])
            pbar.close()


            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]
            epoch_dice = running_dice / dataset_sizes[phase]
                                        
            print('{} Loss: {:.4f} Acc: {:.4f} Dice: {:.4f}'.format(
                phase, epoch_loss, epoch_acc, epoch_dice))

            # deep copy the model
            if phase == 'valid' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                torch.save(model.state_dict(),'gain_vgg_'+str(epoch_acc)+'_acc.pt')

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    #model.save_state_dict('vgg_gain.pt')
    return model

data_dir = '../Data/CBIS-DDSM_classification_1/'
train_csv = '../CSV/gain_train.csv'
#image_size = (640,384)
image_size = (320,192)
num_classes = 2
num_epochs = 50
sigma = 0
w = 1
alpha = 1
img_mean = [0.223, 0.231, 0.243]
img_std = [0.266, 0.270, 0.274]

dataloaders,dataset_sizes,dataset,device = get_classification_dataloader(data_dir,train_csv,image_size,img_mean,img_std,12)

#i = iter(dataloaders['train']).next()
# d = denorm_img(i['image'],img_mean,img_std)
# plt.imshow(d.transpose((1,2,0)))

vggnet = models.vgg11_bn(pretrained=True)
vgg_conv = nn.Sequential(*list(vggnet.children())[0][:-1])
v1 = vgg_gain(vgg_conv).to(device)

params = v1.parameters()
stream_loss = nn.CrossEntropyLoss()
# Observe that all parameters are being optimized
#optimizer_ft = optim.SGD(params, lr=0.01, momentum=0.9)
optimizer_ft = optim.Adam(params, lr=0.01, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
# Decay LR by a factor of 0.1 every 7 epochs
lr_sched = lr_scheduler.StepLR(optimizer_ft, step_size=10, gamma=0.1)

model_ft = gain_train_model(v1, dataloaders, dataset_sizes, device, stream_loss, optimizer_ft, lr_sched, img_mean, img_std, num_epochs, sigma, w, alpha)
