import os
from PIL import Image
import numpy as np
import pickle
from tqdm import tqdm

data_dir='training_data'
output_dir='points'
if not os.path.exists(output_dir):
    os.mkdir(output_dir)

def load_pickle(filename):
    with open(filename, "rb") as f:
        return pickle.load(f)

def convert(in_dir,out_dir,mode):
    txt_dir=os.path.join(in_dir,'splits','v2',mode+'.txt')
    image_dir=os.path.join(in_dir,'v2.2')
    with open(txt_dir, "r") as f:
            image_filelist = [
                os.path.join(image_dir, line.strip())
                for line in f
                if line.strip()
            ]
    points_dir=os.path.join(out_dir,mode)
    if not os.path.exists(points_dir):
        os.mkdir(points_dir)
    
    print('{} set converting begin'.format(mode))
    for location in tqdm(image_filelist):
        rgb_dir = location + "_color_kinect.png"
        label_dir = location + "_label_kinect.png"
        meta_dir = location + "_meta.pkl"
        depth_dir = location+"_depth_kinect.png"
        depth=np.array(Image.open(depth_dir)) / 1000
        image = np.array(Image.open(rgb_dir)) / 255
        label = np.array(Image.open(label_dir))
        meta = load_pickle(meta_dir)
        object_ids=meta['object_ids']

        intrinsic = meta["intrinsic"]/1000
        z = depth
        v, u = np.indices(z.shape)
        uv1 = np.stack([u + 0.5, v + 0.5, np.ones_like(z)], axis=-1)
        points_viewer = uv1 @ np.linalg.inv(intrinsic).T * z[..., None]  # [H, W, 3]
        
        for obj_id in object_ids:
            mask=(label==obj_id)
            if mask.sum()==0:
                continue
            indices=np.nonzero(mask)
            dim1_min=np.min(indices[1])-1
            dim1_max=np.max(indices[1])+2
            dim0_min=np.min(indices[0])-1
            dim0_max=np.max(indices[0])+2
            box=np.array([dim1_min,dim1_max,dim0_min,dim0_max])
            patch_image=image[dim0_min:dim0_max,dim1_min:dim1_max,:]
            patch_xyz=points_viewer[dim0_min:dim0_max,dim1_min:dim1_max]
            patch_label=label[dim0_min:dim0_max,dim1_min:dim1_max]

            name = os.path.basename(meta_dir)[:-9] + f"_{obj_id}.npz"
            np.savez_compressed(
                os.path.join(points_dir,name),
                points=patch_xyz.astype(np.float32),
                colors=patch_image.astype(np.float32),
                boxes=box.astype(np.int32),
                intrinsics=np.array(intrinsic,dtype=np.float32),
                segment=patch_label.astype(np.int32)
            )


convert(data_dir,output_dir,'val')



