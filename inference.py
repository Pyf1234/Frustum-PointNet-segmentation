import os
from PIL import Image
import numpy as np
import pickle
from tqdm import tqdm
import torch
from frustrum_pointnet import PointNet
import torch.nn.functional as F
import matplotlib.pyplot as plt

def load_pickle(filename):
    with open(filename, "rb") as f:
        return pickle.load(f)

def load_data(data_dir):
    rgb_dir = data_dir + "_color_kinect.png"
    label_dir = data_dir + "_label_kinect.png"
    meta_dir = data_dir + "_meta.pkl"
    depth_dir = data_dir+"_depth_kinect.png"
    depth=np.array(Image.open(depth_dir)) / 1000
    image = np.array(Image.open(rgb_dir)) / 255
    label = np.array(Image.open(label_dir))
    meta = load_pickle(meta_dir)
    
    return image,depth,label,meta

def lift(depth,meta):
    intrinsic = meta["intrinsic"]/1000
    z = depth
    v, u = np.indices(z.shape)
    uv1 = np.stack([u + 0.5, v + 0.5, np.ones_like(z)], axis=-1)
    points_viewer = uv1 @ np.linalg.inv(intrinsic).T * z[..., None]  # [H, W, 3]

    return points_viewer

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

def process(image,depth,label,meta,points_viewer):
    object_ids=meta['object_ids']
    batch=[]
    for obj_id in object_ids:
        mask=(label==obj_id)
        if mask.sum()==0:
            continue
        indices=np.nonzero(mask)
        dim1_min=np.min(indices[1])-1
        dim1_max=np.max(indices[1])+2
        dim0_min=np.min(indices[0])-1
        dim0_max=np.max(indices[0])+2
        box=np.array([dim1_min,dim1_max,dim0_min,dim0_max]).astype(np.int32)
        patch_image=image[dim0_min:dim0_max,dim1_min:dim1_max,:].astype(np.float32)
        patch_xyz=points_viewer[dim0_min:dim0_max,dim1_min:dim1_max].astype(np.float32)
        patch_label=label[dim0_min:dim0_max,dim1_min:dim1_max].astype(np.int32)

        points=patch_xyz.reshape(-1,3)
        colors=patch_image.reshape(-1,3)
        boxes=box
        segment=(patch_label==obj_id).reshape(-1,1).squeeze()

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

        points=torch.tensor(points)
        colors=torch.tensor(colors)
        segment=torch.tensor(segment)
        boxes=torch.tensor(boxes)
        id=torch.tensor(obj_id)
        batch.append((points,colors,segment,boxes,id))
    points,colors,segments,boxes,id,mask,max_shape=collate_fn(batch)
    return points,colors,segments,boxes,id,mask,max_shape

def load_model(pth_dir):
    model=PointNet(6,79)
    model.load_state_dict(torch.load(pth_dir))
    model.eval()
    return model

def predict(model,points,colors,id,mask,max_shape):
    num_class=79
    batch_size=points.shape[0]
    dim0=max_shape[0,0].item()
    dim1=max_shape[0,1].item()
    mask=mask
    concat_points=torch.cat((points,colors),dim=2).permute(0,2,1).contiguous()
    one_hote=F.one_hot(id,num_class)
    seg_pred=model(concat_points,one_hote,mask).reshape((batch_size,dim0,dim1,2)).contiguous().permute(0,3,1,2).contiguous()
    pred=torch.argmax(seg_pred,dim=1)
    return pred

def post_process(pred,ids,boxes,image,result_dir):
    pred=np.array(pred)
    ids=np.array(ids)
    boxes=np.array(boxes)
    image_dim0=image.shape[0]
    image_dim1=image.shape[1]
    label=np.ones((image_dim0,image_dim1),dtype=np.int32)*80
    for i,id in enumerate(ids):
        box_i=boxes[i,:]
        dim0=box_i[3]-box_i[2]
        dim1=box_i[1]-box_i[0]
        pred_i=pred[i,:,:]
        pred_i[np.where(pred_i==1)]=id
        pred_i[np.where(pred_i==0)]=80
        pred_i=pred_i.reshape(-1,1)
        area=dim0*dim1
        pred_i=pred_i[0:area].reshape(dim0,dim1)

        patch_i=label[box_i[2]:box_i[3],box_i[0]:box_i[1]]
        patch_i[np.where(patch_i==80)]=pred_i[np.where(patch_i==80)]
        label[box_i[2]:box_i[3],box_i[0]:box_i[1]]=patch_i

    plt.imshow(label)
    result_dir=os.path.join(result_dir,'demo.png')
    img=Image.fromarray(label.astype(np.uint8))
    img.save(result_dir)
    print('evaling success , you can find the result in {}'.format(result_dir))

def inference(model,data_dir,result_dir):
    image,depth,label,meta=load_data(data_dir)
    points_viewer=lift(depth,meta)
    points,colors,segments,boxes,id,mask,max_shape=process(image,depth,label,meta,points_viewer)
    pred=predict(model,points,colors,id,mask,max_shape)
    post_process(pred,id,boxes,image,result_dir)

    

        



