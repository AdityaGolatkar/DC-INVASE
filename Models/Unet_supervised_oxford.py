#Pytorch
import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler

#Torchvision
import torchvision
from torchvision import datasets, models, transforms, utils
  
#Pytorch
import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler

#Torchvision
import torchvision
from torchvision import datasets, models, transforms, utils
from torch.utils.data import Dataset, DataLoader

#Image Processing
import matplotlib.pyplot as plt
from skimage import io, transform, color
import PIL
from PIL import Image

#Others
import sklearn.metrics
from sklearn.metrics import *
import numpy as np
import pandas as pd
import cv2
import time
import os
import copy
from model_summary import *
import pretrainedmodels
import tqdm
from tqdm import tqdm as tqdm
import warnings
warnings.filterwarnings("ignore")



class dataset(Dataset):

    def __init__(self, csv_file, root_dir, transform=None):

        self.data_frame = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform
        self.mask_dir = self.root_dir.replace('images','masks')
        
    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir,self.data_frame.iloc[idx]['name'])
        image = Image.open(img_name)
        
        mask_name = os.path.join(self.mask_dir,self.data_frame.iloc[idx]['name'])
        mask = io.imread(mask_name)
        mask = np.array([mask,mask,mask]).transpose((1,2,0))
        mask = Image.fromarray(mask)

        label = self.data_frame.iloc[idx]['category']       

        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)
    
        return {'image':image, 'category':label, 'mask':mask, 'name':self.data_frame.iloc[idx]['name']}
    

def get_dataloader(data_dir, train_csv_path, image_size, img_mean, img_std, batch_size=1):

    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize(image_size),
            transforms.RandomHorizontalFlip(0.5),
            transforms.RandomAffine(translate=(0,0.2),degrees=15,shear=15),
            transforms.ToTensor(),
            transforms.Normalize(img_mean,img_std)
        ]),
        'valid': transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize(img_mean,img_std)
        ]),
        'test': transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize(img_mean,img_std)
        ])
    }

    image_datasets = {}
    dataloaders = {}
    dataset_sizes = {}

    for x in ['train', 'valid', 'test']:
        if x == 'test':
            bs = 1
            sh = False
        else:
            bs = batch_size
            sh = True
        image_datasets[x] = dataset(train_csv_path.replace('train',x),root_dir=data_dir,transform=data_transforms[x])
        dataloaders[x] = torch.utils.data.DataLoader(image_datasets[x], batch_size=bs,shuffle=sh, num_workers=8)    
        dataset_sizes[x] = len(image_datasets[x])

    device = torch.device("cuda:0")

    return dataloaders,dataset_sizes,image_datasets,device

def build_model():
    class unetConv2(nn.Module):
        def __init__(self, in_size, out_size, is_batchnorm):
            super(unetConv2, self).__init__()

            if is_batchnorm:
                self.conv1 = nn.Sequential(nn.Conv2d(in_size, out_size, 3, 1, 1),
                                           nn.BatchNorm2d(out_size),
                                           nn.ReLU(),)
                self.conv2 = nn.Sequential(nn.Conv2d(out_size, out_size, 3, 1, 1),
                                           nn.BatchNorm2d(out_size),
                                           nn.ReLU(),)
            else:
                self.conv1 = nn.Sequential(nn.Conv2d(in_size, out_size, 3, 1, 1),
                                           nn.ReLU(),)
                self.conv2 = nn.Sequential(nn.Conv2d(out_size, out_size, 3, 1, 1),
                                           nn.ReLU(),)
        def forward(self, inputs):
            outputs = self.conv1(inputs)
            outputs = self.conv2(outputs)
            return outputs

    class unetUp(nn.Module):
        def __init__(self, in_size, out_size, is_deconv):
            super(unetUp, self).__init__()
            self.conv = unetConv2(in_size, out_size, False)
            if is_deconv:
                self.up = nn.ConvTranspose2d(in_size, out_size, kernel_size=2, stride=2)
            else:
                self.up = nn.UpsamplingBilinear2d(scale_factor=2)

        def forward(self, inputs1, inputs2):
            outputs2 = self.up(inputs2)
            offset = outputs2.size()[2] - inputs1.size()[2]
            padding = 2 * [offset // 2, offset // 2]
            outputs1 = F.pad(inputs1, padding)
            return self.conv(torch.cat([outputs1, outputs2], 1))

    class unet(nn.Module):

        def __init__(self, feature_scale=1, n_classes=1, is_deconv=True, in_channels=3, is_batchnorm=True):
            super(unet, self).__init__()
            self.is_deconv = is_deconv
            self.in_channels = in_channels
            self.is_batchnorm = is_batchnorm
            self.feature_scale = feature_scale

            filters = [32, 64, 128, 256, 512]
            filters = [int(x / self.feature_scale) for x in filters]

            #downsampling
            self.conv1 = unetConv2(self.in_channels, filters[0], self.is_batchnorm)
            self.maxpool1 = nn.MaxPool2d(kernel_size=2)

            self.conv2 = unetConv2(filters[0], filters[1], self.is_batchnorm)
            self.maxpool2 = nn.MaxPool2d(kernel_size=2)

            self.conv3 = unetConv2(filters[1], filters[2], self.is_batchnorm)
            self.maxpool3 = nn.MaxPool2d(kernel_size=2)

            self.conv4 = unetConv2(filters[2], filters[3], self.is_batchnorm)
            self.maxpool4 = nn.MaxPool2d(kernel_size=2)
                    
            self.center = unetConv2(filters[3], filters[4], self.is_batchnorm)

            # upsampling
            self.up_concat4 = unetUp(filters[4], filters[3], self.is_deconv)
            self.up_concat3 = unetUp(filters[3], filters[2], self.is_deconv)
            self.up_concat2 = unetUp(filters[2], filters[1], self.is_deconv)
            self.up_concat1 = unetUp(filters[1], filters[0], self.is_deconv)

            # final conv (without any concat)
            self.final = nn.Conv2d(filters[0], n_classes, 1)

        def forward(self, inputs):
            conv1 = self.conv1(inputs)
            maxpool1 = self.maxpool1(conv1)

            conv2 = self.conv2(maxpool1)
            maxpool2 = self.maxpool2(conv2)

            conv3 = self.conv3(maxpool2)
            maxpool3 = self.maxpool3(conv3)

            conv4 = self.conv4(maxpool3)
            maxpool4 = self.maxpool4(conv4)

            center = self.center(maxpool4)
            up4 = self.up_concat4(conv4, center)
            up3 = self.up_concat3(conv3, up4)
            up2 = self.up_concat2(conv2, up3)
            up1 = self.up_concat1(conv1, up2)

            final = self.final(up1)

            return final
        
    model = unet()
    return model

def denorm_img(img_ten,img_mean,img_std):

    bz,nc,h,w = img_ten.shape
    output = []
    img_num = img_ten.numpy()
    
    for i in range(bz):
        
        img = img_ten[i].numpy().squeeze()
        
        img[0,:,:] = img[0,:,:]*img_std[0]
        img[1,:,:] = img[1,:,:]*img_std[1]
        img[2,:,:] = img[2,:,:]*img_std[2]

        img[0,:,:] = img[0,:,:] + img_mean[0]
        img[1,:,:] = img[1,:,:] + img_mean[1]
        img[2,:,:] = img[2,:,:] + img_mean[2]
        
        img = img.mean(axis=0)
        img[img>=0.2*img.max()] = 1
        img[img<0.2*img.max()] = 0
        
        output.append(img)
    
    output = np.array(output)
    return output

def get_IoU(pred, targs):
    return (pred*targs).sum() / ((pred+targs).sum() - (pred*targs).sum())
    

class unet_oxford():
    def __init__(self):
        
        #Initialization
        self.data_dir =  '../Data/oxford_pets/sparse_images/'
        self.train_csv = '../CSV/oxford_pet_train.csv'
        self.num_epochs = 100
        self.input_shape = (224,224)#(640,512) #(640,512)#(224,224)#(640,384) (640,512)
        self.batch_size = 1
        self.img_mean = [0,0,0]
        self.img_std = [1,1,1]
        self.exp_name = 'Weights/unet_oxford'
        
        #Define the three models
        self.model = build_model()
        
        #Put them on the GPU
        self.model = self.model.cuda()
        
        #Get the dataloaders
        self.dataloaders,self.dataset_sizes,self.dataset,self.device = get_dataloader(self.data_dir,self.train_csv,\
                                                        self.input_shape,self.img_mean,self.img_std,self.batch_size)
        

        self.optimizer = optim.Adam(self.model.parameters(),lr=0.0001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)


        
    def train(self):
        
        since = time.time()
        best_epoch_acc = 0.0
        best_epoch_iou = 0.0
        
        for epoch in range(self.num_epochs):
            print('Epoch {}/{}'.format(epoch, self.num_epochs - 1),flush=True)
            print('-' * 10,flush=True)

            # Each epoch has a training and validation phase
            for phase in ['train', 'valid']:
                if phase == 'train':
                    
                    #Set the models to training mode
                    self.model.train()
                
                else:
                    #Set the models to evaluation mode
                    self.model.eval()
                    
                #Keep a track of all the three loss
                running_loss = 0.0
                
                #Metrics : predictor auc and selector iou
                running_iou = 0
                
                #tqdm bar
                pbar = tqdm(total=self.dataset_sizes[phase])

                # Iterate over data.
                for sampled_batch in self.dataloaders[phase]:

                    inputs = sampled_batch['image']
                    labels = sampled_batch['category']
                    mask = torch.Tensor(denorm_img(sampled_batch['mask'],self.img_mean,self.img_std)).to(self.device)
                    
                    
                    #Input needs to be float and labels long
                    inputs = inputs.float().to(self.device)
                    labels = labels.long().to(self.device)

                    # zero the parameter gradients
                    self.optimizer.zero_grad()
                
                    # forward
                    # track history if only in train
                    with torch.set_grad_enabled(phase == 'train'):
                        
                        outputs = self.model(inputs)
                    
                        out4loss = outputs.view(-1)
                        mask4loss = mask.view(-1)
                        #import pdb;pdb.set_trace()
                        
                        #loss = nn.MSELoss()(F.sigmoid(out4loss),mask4loss)
                        loss = nn.BCEWithLogitsLoss()(out4loss,mask4loss)
                        
                        
                        # backward + optimize only if in training phase
                        if phase == 'train':
                            
                            loss.backward()
                            self.optimizer.step()
                                    
                    preds = outputs
                    preds[preds>0] = 1
                    preds[preds<=0] = 0
                    #print(preds.mean(),mask.mean())
                    
                    # statistics
                    running_loss += loss.item() * inputs.size(0)
                    running_iou += get_IoU(preds,mask) * inputs.size(0)
                    #print(running_iou)
                    

                    pbar.update(inputs.shape[0])
                pbar.close()

                epoch_loss = running_loss / self.dataset_sizes[phase]
                epoch_iou = running_iou / self.dataset_sizes[phase]
                
                print('{} Sel_Loss: {:.4f} IoU: {:.4f}'.format(
                    phase, epoch_loss, epoch_iou))

                # deep copy the model
                if phase == 'valid' and epoch_iou > best_epoch_iou:
                    best_epoch_iou = epoch_iou
                    torch.save(self.model.state_dict(),self.exp_name+'.pt')
                    
        time_elapsed = time.time() - since
        print('Training complete in {:.0f}m {:.0f}s'.format(
            time_elapsed // 60, time_elapsed % 60))
        print('Best IoU: {:4f}'.format(best_epoch_iou))

        torch.save(self.model.state_dict(),self.exp_name+'_final.pt')
        
        print('Training completed finally !!!!!')
        
        
    def test_model(self):
                
        self.selector.load_state_dict(torch.load(self.exp_name+'_sel.pt'))
        self.selector.eval()
        
        mIoU = 0
        total = 0
        mode = 'test'

        with torch.no_grad():
            for data in self.dataloaders[mode]:

                images = data['image']
                mask = data['mask']

                images = images.to(self.device)
                
                sel_prob = make_prob(self.selector(images))
                iou = get_IoU(sel_prob,mask)
                
                total += labels.size(0)
                mIoU += iou

        print("mIoU:", 1.0*mIoU/total)
        
    def test_model_acc(self):
                
        self.predictor.load_state_dict(torch.load(self.exp_name+'_pred.pt'))
        self.predictor.eval()
        
        acc = 0
        total = 0
        mode = 'test'

        with torch.no_grad():
            for data in self.dataloaders[mode]:

                images = data['image']
                labels = data['category']
                
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                output = self.predictor(images)
                _,out = torch.max(output,1)
                
                total += labels.size(0)
                acc += torch.sum(out==labels.data)

        print("mIoU:", 1.0*acc.double()/total)
    