import torch
from torch.utils.data import DataLoader,Dataset
import torch.nn as nn
import torch.nn.functional as F
import os 
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

import os
os.environ['KMP_DUPLICATE_LIB_OK']='TRUE'

datadir='points'
save_dir='result'
class frustum_dataset(Dataset):
    def __init__(self,data_dir,mode) :
        self.datadir=os.path.join(data_dir,mode)
        self.file_list=os.listdir(self.datadir)
    
    def __len__(self):
        return len(self.file_list)

    def __getitem__(self,idx):
        sample=np.load(os.path.join(self.datadir,self.file_list[idx]))
        loca=self.file_list[idx]
        id=int(loca[(loca.find('_')+1):(loca.find('.'))])
        points=sample['points'].reshape(-1,3)
        colors=sample['colors'].reshape(-1,3)
        boxes=sample['boxes']
        #intrinsics=sample['intrinsics']
        segment=(sample['segment']==id).reshape(-1,1).squeeze()
        if points.shape[0]>0:
            x0,y0,z0=points[points.shape[0]//2,:]
            angle=np.arctan(z0/x0)
            rot_matrix=np.array([[np.sin(angle),np.cos(angle)],[-np.cos(angle),np.sin(angle)]])
            points[:,[0,2]]=points[:,[0,2]]@rot_matrix
        else:
            points=np.zeros((100,3),dtype=np.float32)
            colors=np.zeros((100,3),dtype=np.float32)
            segment=(np.zeros(100)==1).reshape(-1,1).squeeze()
            boxes=np.array([0,0,0,0],dtype=np.int32)

        return torch.tensor(points),torch.tensor(colors),torch.tensor(segment),torch.tensor(boxes),torch.tensor(id)
    
def collate_fn(batch):
    batch_size=len(batch)
    max_points=max([s[0].shape[0] for s in batch])
    max_idx=[s[0].shape[0] for s in batch].index(max_points)
    mask=torch.zeros((batch_size,max_points))

    points=torch.zeros((batch_size,max_points,3))
    colors=torch.zeros((batch_size,max_points,3))
    segments=torch.zeros((batch_size,max_points),dtype=torch.int64)
    boxes=torch.zeros((batch_size,4),dtype=torch.int64)
    id=torch.zeros((batch_size,1),dtype=torch.int64)

    for i,(raw_points,raw_colors,raw_segments,raw_boxes,raw_id) in enumerate(batch):
        num=raw_points.shape[0]
        points[i,0:num,:]=raw_points
        colors[i,0:num,:]=raw_colors
        segments[i,0:num]=raw_segments
        boxes[i,:]=raw_boxes
        id[i]=raw_id
        mask[i,0:num]=1
    
    max_box=boxes[max_idx,:]
    max_shape=torch.tensor([[max_box[1]-max_box[0],max_box[3]-max_box[2]]])

    return points,colors,segments,boxes,id,mask,max_shape

#need the points to be B x channel x points_num
class PointNet_tiny(nn.Module):
    def __init__(self,in_channel,num_class) :
        super(PointNet_tiny,self).__init__()
        self.conv1 = nn.Conv1d(in_channel, 64, 1)
        #self.conv2 = nn.Conv1d(64, 64, 1)
        #self.conv3 = nn.Conv1d(64, 64, 1)
        self.conv4 = nn.Conv1d(64, 128, 1)
        self.conv5 = nn.Conv1d(128, 1024, 1)
        self.bn1 = nn.BatchNorm1d(64)
        #self.bn2 = nn.BatchNorm1d(64)
        #self.bn3 = nn.BatchNorm1d(64)
        self.bn4 = nn.BatchNorm1d(128)
        self.bn5 = nn.BatchNorm1d(1024)

        self.n_classes = num_class
        self.dconv1 = nn.Conv1d(1088+num_class, 512, 1)
        self.dconv2 = nn.Conv1d(512, 256, 1)
        self.dconv3 = nn.Conv1d(256, 128, 1)
        #self.dconv4 = nn.Conv1d(128, 128, 1)
        #self.dropout = nn.Dropout(p=0.5)
        self.dconv5 = nn.Conv1d(128, 2, 1)
        self.dbn1 = nn.BatchNorm1d(512)
        self.dbn2 = nn.BatchNorm1d(256)
        self.dbn3 = nn.BatchNorm1d(128)
        #self.dbn4 = nn.BatchNorm1d(128)

    def forward(self,points,one_hot,mask):
        batch_size=points.shape[0]
        num_points=points.shape[2]
        out1=F.relu(self.bn1(self.conv1(points)))
        #out2=F.relu(self.bn2(self.conv2(out1)))
        #out3=F.relu(self.bn3(self.conv3(out2)))
        out4=F.relu(self.bn4(self.conv4(out1)))
        out5=F.relu(self.bn5(self.conv5(out4)))

        out5=out5*(mask.view(batch_size,1,-1))
        glob=torch.max(out5,dim=2,keepdim=True)[0]
        one_hot=one_hot.view(batch_size,-1,1)
        expand_global_feat = torch.cat([glob, one_hot],1)#bs,1027,1
        expand_global_feat_repeat = expand_global_feat.view(batch_size,-1,1)\
                .repeat(1,1,num_points)# bs,1027,n
        concat_feat = torch.cat([out1,expand_global_feat_repeat],1)

        out6=F.relu(self.dbn1(self.dconv1(concat_feat)))
        out7=F.relu(self.dbn2(self.dconv2(out6)))
        out8=F.relu(self.dbn3(self.dconv3(out7)))
        #out9=F.relu(self.dbn4(self.dconv4(out8)))
        #x=self.dropout(out9)
        out10=self.dconv5(out8)
        seg_pred = out10.transpose(2,1).contiguous()

        return seg_pred

class PointNet(nn.Module):
    def __init__(self,in_channel,num_class) :
        super(PointNet,self).__init__()
        self.conv1 = nn.Conv1d(in_channel, 64, 1)
        self.conv2 = nn.Conv1d(64, 64, 1)
        self.conv3 = nn.Conv1d(64, 64, 1)
        self.conv4 = nn.Conv1d(64, 128, 1)
        self.conv5 = nn.Conv1d(128, 1024, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(64)
        self.bn3 = nn.BatchNorm1d(64)
        self.bn4 = nn.BatchNorm1d(128)
        self.bn5 = nn.BatchNorm1d(1024)

        self.n_classes = num_class
        self.dconv1 = nn.Conv1d(1088+num_class, 512, 1)
        self.dconv2 = nn.Conv1d(512, 256, 1)
        self.dconv3 = nn.Conv1d(256, 128, 1)
        self.dconv4 = nn.Conv1d(128, 128, 1)
        self.dropout = nn.Dropout(p=0.5)
        self.dconv5 = nn.Conv1d(128, 2, 1)
        self.dbn1 = nn.BatchNorm1d(512)
        self.dbn2 = nn.BatchNorm1d(256)
        self.dbn3 = nn.BatchNorm1d(128)
        self.dbn4 = nn.BatchNorm1d(128)

    def forward(self,points,one_hot,mask):
        batch_size=points.shape[0]
        num_points=points.shape[2]
        out1=F.relu(self.bn1(self.conv1(points)))
        out2=F.relu(self.bn2(self.conv2(out1)))
        out3=F.relu(self.bn3(self.conv3(out2)))
        out4=F.relu(self.bn4(self.conv4(out3)))
        out5=F.relu(self.bn5(self.conv5(out4)))

        out5=out5*(mask.view(batch_size,1,-1))
        glob=torch.max(out5,dim=2,keepdim=True)[0]
        one_hot=one_hot.view(batch_size,-1,1)
        expand_global_feat = torch.cat([glob, one_hot],1)#bs,1027,1
        expand_global_feat_repeat = expand_global_feat.view(batch_size,-1,1)\
                .repeat(1,1,num_points)# bs,1027,n
        concat_feat = torch.cat([out2,expand_global_feat_repeat],1)

        out6=F.relu(self.dbn1(self.dconv1(concat_feat)))
        out7=F.relu(self.dbn2(self.dconv2(out6)))
        out8=F.relu(self.dbn3(self.dconv3(out7)))
        out9=F.relu(self.dbn4(self.dconv4(out8)))
        x=self.dropout(out9)
        out10=self.dconv5(x)
        seg_pred = out10.transpose(2,1).contiguous()

        return seg_pred

def train(batch_size,epochs,lr,num_class):
    train_set=frustum_dataset(datadir,'train')
    val_set=frustum_dataset(datadir,'val')
    train_loader=DataLoader(train_set,batch_size,True,collate_fn=collate_fn)
    val_loader=DataLoader(val_set,batch_size,False,collate_fn=collate_fn)
    model=PointNet(6,num_class).cuda()
    opti=torch.optim.Adam(model.parameters(),lr)
    loss_fun=nn.CrossEntropyLoss(reduction='none')

    train_loss=[]
    val_acc=[]

    for epoch in range(epochs):
        print('epoch {} start training'.format(epoch+1))
        
        model.train()
        epoch_loss=0.0
        for points,colors,segments,boxes,id,mask,max_shape in tqdm(train_loader):
            
            batch_size=points.shape[0]
            dim0=max_shape[0,0].item()
            dim1=max_shape[0,1].item()
            mask=mask.cuda()
            concat_points=torch.cat((points,colors),dim=2).permute(0,2,1).contiguous().cuda()
            one_hote=F.one_hot(id,num_class).cuda()
            seg_pred=model(concat_points,one_hote,mask).reshape((batch_size,dim0,dim1,2)).contiguous().permute(0,3,1,2).contiguous()
            seg=segments.reshape((batch_size,dim0,dim1)).contiguous().cuda()
            loss_mask=mask.reshape((batch_size,dim0,dim1)).contiguous()
            loss=loss_fun(seg_pred,seg)
            effet_loss=(loss*loss_mask).sum()

            opti.zero_grad()
            effet_loss.backward()
            opti.step()
            epoch_loss+=effet_loss.item()
        train_loss.append(epoch_loss)
        print('epoch {} training loss:{}'.format(epoch+1,epoch_loss))
        
        print('epoch {} start evaling'.format(epoch+1))
        model.eval()
        epoch_right=0
        epoch_num=0
        with torch.no_grad():
            for points,colors,segments,boxes,id,mask,max_shape in tqdm(val_loader):
                dim0=max_shape[0,0].item()
                dim1=max_shape[0,1].item()
                batch_size=points.shape[0]

                mask=mask.cuda()
                concat_points=torch.cat((points,colors),dim=2).permute(0,2,1).contiguous().cuda()
                one_hote=F.one_hot(id,num_class).cuda()
                seg_pred=model(concat_points,one_hote,mask).reshape((batch_size,dim0,dim1,2)).contiguous().permute(0,3,1,2).contiguous()
                seg=segments.reshape((batch_size,dim0,dim1)).contiguous().cuda()
                loss_mask=mask.reshape((batch_size,dim0,dim1)).contiguous()
                
                pred=torch.argmax(seg_pred,dim=1)
                right_points=((pred==seg)*loss_mask).sum().item()
                all_points=loss_mask.sum().item()
                epoch_right+=right_points
                epoch_num+=all_points
        val_acc.append(epoch_right/epoch_num)
        print('epoch {} evaling acc:{}'.format(epoch+1,epoch_right/epoch_num))
        torch.save(model.state_dict(),os.path.join(save_dir,'model.pth'))

    print('training success !')
    print('save the model and results to {} file'.format(save_dir))
    torch.save(model.state_dict(),os.path.join(save_dir,'model.pth'))
    print('save the training details to {} file'.format(save_dir))
    x=np.arange(epochs)
    train_loss=np.array(train_loss)
    plt.plot(x,train_loss,color='blue')
    plt.xlabel('epoch')
    plt.ylabel('training loss')
    loss_dir=os.path.join(save_dir,'loss.png')
    plt.savefig(loss_dir)
    plt.clf()
    val_acc=np.array(val_acc)
    plt.plot(x,val_acc)
    plt.xlabel('epoch')
    plt.ylabel('evaling accuracy')
    acc_dir=os.path.join(save_dir,'acc.png')
    plt.savefig(acc_dir)
    plt.clf()

if __name__ == "__main__":
    train(4,4,0.001,79)
            





        





