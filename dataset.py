import torch
from torch.utils.data import Dataset
import os
import numpy as np
import json
from PIL import Image,ImageOps
from torchvision import transforms
from torchvision.transforms.functional import rotate,crop,resize


class fscoco_train(Dataset):
    def __init__(self,root= "DATA/fscoco-seg/train",transform=None,augment=False,SKETCH_SIZE=512):
        self.root = root
        self.transform = transform
        self.augment = augment
        self.sketch_dir = os.path.join(root,"sketches")
        self.text_dir = os.path.join(root,"text")
        
        self.augmentation = transforms.Compose([
            transforms.RandomRotation(20),  # Rotate by ±10°
            transforms.RandomCrop(450),  # Crop of size 200x200 for example
            transforms.Resize((SKETCH_SIZE, SKETCH_SIZE))  # Resize back to 224x224
        ])
        
        # Get list of subdirectories for sketches and captions
        self.sketch_subdirs = sorted(os.listdir(self.sketch_dir))
        self.txt_subdirs = sorted(os.listdir(self.text_dir))

        # Get list of all sketch files and corresponding caption files
        self.sketch_files = []
        self.txt_files = []
        for sketch_subdir, txt_subdir in zip(self.sketch_subdirs, self.txt_subdirs):
            sketch_subdir_path = os.path.join(self.sketch_dir, sketch_subdir)
            txt_subdir_path = os.path.join(self.text_dir, txt_subdir)
            
            sketch_files_subdir = sorted(os.listdir(sketch_subdir_path))
            txt_files_subdir = sorted(os.listdir(txt_subdir_path))

            self.sketch_files += [os.path.join(sketch_subdir, sketch_file) for sketch_file in sketch_files_subdir]
            self.txt_files += [os.path.join(txt_subdir, txt_file) for txt_file in txt_files_subdir]
            
    def __len__(self):
        return len(self.sketch_files)
    
    def __getitem__(self,index):
        text_path = os.path.join(self.text_dir,self.txt_files[index])
        with open(text_path,"r") as f:
            caption = f.read()
        
        sketch_path = os.path.join(self.sketch_dir,self.sketch_files[index])
        sketch = Image.open(sketch_path).convert("RGB")
        
        sketch_aug = ImageOps.invert(sketch)
        aug_sketch = self.augmentation(sketch_aug)
        aug_sketch = ImageOps.invert(aug_sketch)
        
        if self.transform:
            sketch = self.transform(sketch)
            aug_sketch = self.transform(aug_sketch)

        if self.augment:
            sketch = torch.stack([sketch,aug_sketch])

        return sketch,caption
    
    
class fscoco_test(Dataset):
    def __init__(self,root= "DATA/fscoco-seg/test"):
        self.root = root
        self.img_dir = os.path.join(root,"images")
        self.text_dir = os.path.join(root,"captions")
        self.stroke_dir = os.path.join(root,"vector_sketches")
        self.label_dir = os.path.join(root,"classes")
        
        # Get list of subdirectories for images and captions
        self.img_files = sorted(os.listdir(self.img_dir))
        self.txt_files = sorted(os.listdir(self.text_dir))
        self.strokes_files = sorted(os.listdir(self.stroke_dir))
        self.label_files = sorted(os.listdir(self.label_dir))

    def __len__(self):
        return len(self.strokes_files)
    
    def __getitem__(self,index):
        
        img_path = os.path.join(self.img_dir,self.img_files[index])
        strokes_path = os.path.join(self.stroke_dir,self.strokes_files[index])
        classes_path = os.path.join(self.label_dir,self.label_files[index])
        with open(classes_path,"r") as f:
            classes = json.load(f)
        pen_state = np.load(strokes_path,allow_pickle=True) # (n,3) array, where n is the number of pen states 
        text_path = os.path.join(self.text_dir,self.txt_files[index])
        
        with open(text_path,"r") as f:
            caption = f.read()
        
        return pen_state,classes, caption,img_path