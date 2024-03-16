import models
import torch

from torchvision.transforms import InterpolationMode
BICUBIC = InterpolationMode.BICUBIC
from torch import nn
from torch.utils.data import DataLoader
from dataset import fscoco_train
from vpt.launch import default_argument_parser
from torchvision.transforms import Compose, Resize, ToTensor, Normalize
from utils import setup, get_similarity_map,zero_clapping,get_train_classes,\
                tensor_to_binary_img, sketch_text_pairs, get_threshold,triplet_loss_func_L1
import wandb
from models import clip
import os

def main(args):
    # set up cfg and args
    cfg = setup(args)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    preprocess_no_T =  Compose([Resize((224, 224), interpolation=BICUBIC), #ToTensor(),
    Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))])

    model, preprocess = models.load("CS-ViT-B/16", device=device,cfg=cfg,train_bool=True)
    print("Model loaded successfully")

    train_dataset = fscoco_train(transform=preprocess,augment=False) # Load the training dataset
    train_dataloader = DataLoader(train_dataset, batch_size=cfg.bz, shuffle=True, num_workers=8)

    print("Extracting classes from training dataset.. This might take a minute.")
    train_classes = get_train_classes(train_dataset,max_classes=cfg.max_classes)
    
    if torch.cuda.device_count() > 1:  # If we have more than one GPU
        model = nn.DataParallel(model)
    
    learnable_threshold = nn.Parameter(torch.tensor(cfg.threshold))  # Initialize with default threshold
    threshold_optimizer = torch.optim.AdamW([learnable_threshold], lr=1e-4)
    
    vit_optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate)

    if cfg.WANDB:
        wandb.init(project="sketch_segmentation", name=cfg.MODEL.PROMPT.LOG)
    
    num_of_tokens = cfg.MODEL.PROMPT.NUM_TOKENS
    num_epochs = 20
    
    print("Starting training")
    for epoch in range(num_epochs):
        model.train()

        for batch_idx, (sketches,captions) in enumerate(train_dataloader):
            sketches = sketches.view(-1,3,224,224).to(device)
            sketches_w,classes,captions_pair = sketch_text_pairs(sketches,captions,max_classes=cfg.max_classes)
            sketches_w_binary = tensor_to_binary_img(sketches_w,device)
            sketches_b =  1 - sketches_w_binary
            tokenized_captions = clip.tokenize(captions_pair).to(device)
            tokenized_classes = clip.tokenize(classes).to(device)            

            scene_features,class_features= model(sketches_w, tokenized_classes,layer_num=[12],return_logits=False,mode="train")
            
            similarity = scene_features @ class_features.T
            patches_similarity = similarity[:, num_of_tokens+1:, :]
            similarity_maps = get_similarity_map(patches_similarity,sketches_w.shape[2:])
            
            threshold_value = get_threshold(learnable_threshold)            
            weights_hard_sm = zero_clapping(similarity_maps,threshold_value)
            weights_hard_sm = weights_hard_sm.unsqueeze(1).repeat(1,3,1,1)
            w_sketches = sketches_b * weights_hard_sm
            w_sketches_white = w_sketches.max() - w_sketches 
            w_sketches_white = preprocess_no_T(w_sketches_white)
            
            w_sketch_features,caption_features= model(w_sketches_white, tokenized_captions,layer_num=[7,10,12],return_logits=False,mode="train")

            w_sketch_features_l7 = w_sketch_features[0]
            w_sketch_features_l10 = w_sketch_features[1]
            w_sketch_features_l12 = w_sketch_features[2]
                        
            class_to_idx = {name: i for i, name in enumerate(train_classes)}
            labels = torch.tensor([class_to_idx[name] for name in classes]).to(device)
            triplet_loss_scene = triplet_loss_func_L1(scene_features[:,0,:],caption_features,labels,margin=cfg.margin)
            triplet_loss_final_layer = triplet_loss_func_L1(w_sketch_features_l12[:,0,:],class_features,labels,margin=cfg.margin)
            triplet_loss_l7 = triplet_loss_func_L1(w_sketch_features_l7[:,0,:],class_features,labels,margin=cfg.margin)
            triplet_loss_l10 = triplet_loss_func_L1(w_sketch_features_l10[:,0,:],class_features,labels,margin=cfg.margin)
            
            loss = triplet_loss_scene + triplet_loss_final_layer + triplet_loss_l7 + triplet_loss_l10 
            
            vit_optimizer.zero_grad()
            threshold_optimizer.zero_grad()
            loss.backward()
            vit_optimizer.step()
            threshold_optimizer.step()
            
            print(f'[Training] [iteration/epoch/total_epochs] [{batch_idx}/{epoch}/{num_epochs}] -> Loss: {float(loss)} [threshold: {threshold_value}]', end = "\r") 

            if cfg.WANDB:
                wandb.log({"Train loss": loss.item()})
                wandb.log({"Threshold value":threshold_value})
                
                
        if epoch % cfg.save_every == 0:
            os.makedirs(f"checkpoint/{cfg.MODEL.PROMPT.LOG}", exist_ok=True)
            torch.save(model.state_dict(), f"checkpoint/{cfg.MODEL.PROMPT.LOG}/model_{epoch}.pth")
        
if __name__ == '__main__':
    args = default_argument_parser().parse_args()    
    main(args)
