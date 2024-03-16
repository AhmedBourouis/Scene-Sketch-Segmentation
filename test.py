import models
import torch
from torchvision.transforms import InterpolationMode
BICUBIC = InterpolationMode.BICUBIC
from utils import setup, get_similarity_map, pen_state_to_strokes, prerender_stroke, \
                    pixel_level_segmentation, compute_accuracy, compute_miou
from torchvision.transforms import Compose, Resize, ToTensor, Normalize
from collections import OrderedDict
from vpt.launch import default_argument_parser
import numpy as np
from PIL import Image
from dataset import fscoco_test
import json

def main(args):
    
    cfg = setup(args)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    preprocess =  Compose([Resize((224, 224), interpolation=BICUBIC), ToTensor(),
    Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))])

    Ours, preprocess = models.load("CS-ViT-B/16", device=device,cfg=cfg,train_bool=False)
    state_dict = torch.load(cfg.checkpoint_path)
    
    # Trained on 2 gpus so we need to remove the prefix "module." to test it on a single GPU
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:] # remove `module.`
        new_state_dict[name] = v 
    Ours.load_state_dict(new_state_dict)     
    Ours.eval()     
    print("Model loaded successfully")

    segmented_sketches = fscoco_test()
    
    num_of_tokens = cfg.MODEL.PROMPT.NUM_TOKENS
    
    # Read the classes from json file
    with open('DATA/fscoco-seg/test/all_classes.json', 'r') as f:
        all_classes = json.load(f)
    
    average_P_metric = []
    average_C_metric = []

    total_miou = 0
    for idx, (pen_state, labels,caption, img_path) in enumerate(segmented_sketches):
        # Preprocess the vector sketch
        strokes = pen_state_to_strokes(pen_state) 
        strokes = prerender_stroke(strokes).squeeze(1).float()  
        strokes_seg = np.array(strokes)
        o_sketch = np.array(strokes).sum(0)*255
        o_sketch = np.repeat(o_sketch[np.newaxis, :, :], 3, axis=0)
        o_sketch = np.transpose(o_sketch, (1, 2, 0)).astype('uint8')
        o_sketch = np.where(o_sketch>0, 255, o_sketch)
        o_sketch = 255 - o_sketch
        pil_img = Image.fromarray(o_sketch)
        sketch_tensor = preprocess(pil_img).unsqueeze(0).to(device)
        
        classes_per_sketch = np.unique(labels)
        classes = classes_per_sketch.tolist()
                
        with torch.no_grad():
            text_features = models.encode_text_with_prompt_ensemble(Ours, classes, device,no_module=True)
            redundant_features = models.encode_text_with_prompt_ensemble(Ours, [""], device,no_module=True)   
        classes = ["blank_pixel"] + classes 
        
        gt_sketch_seg = pixel_level_segmentation(strokes_seg, labels, classes,size=strokes_seg.shape[-1])

        with torch.no_grad():
            sketch_features = Ours.encode_image(sketch_tensor,layers=[12],text_features=text_features-redundant_features,mode="test") 
            sketch_features = sketch_features / sketch_features.norm(dim=1, keepdim=True)
            
        similarity = sketch_features @ (text_features - redundant_features).t()
        patches_similarity = similarity[0,:, num_of_tokens +1:, :]
        pixel_similarity = get_similarity_map(patches_similarity,pil_img.size)
        if len(pixel_similarity.shape) == 4:
            pixel_similarity = pixel_similarity[torch.arange(pixel_similarity.shape[0]),:,:,torch.arange(pixel_similarity.shape[0])]
            pred_sketch_seg = torch.argmax(pixel_similarity, dim=0).cpu().numpy()
        else:
            pred_sketch_seg = torch.argmax(pixel_similarity, dim=-1).cpu().numpy()
        
        mapping_indices = {i: all_classes.index(j) for i, j in enumerate(classes)}
        pred_sketch_seg = np.vectorize(mapping_indices.get)(pred_sketch_seg+1)
        gt_sketch_seg = np.vectorize(mapping_indices.get)(gt_sketch_seg)
        pred_sketch_seg[gt_sketch_seg == 0] = 0

        P_metric, C_metric = compute_accuracy(strokes_seg,labels, all_classes, gt_sketch_seg, pred_sketch_seg)
        iou = compute_miou(gt_sketch_seg, pred_sketch_seg,all_classes)
        
        average_P_metric.append(P_metric)
        average_C_metric.append(C_metric)
        total_miou += iou
        
        print(f"Processing {idx+1}/{len(segmented_sketches)} sketches", end = "\r")
        
    P_value = sum(average_P_metric)/len(average_P_metric)
    C_value = sum(average_C_metric)/len(average_C_metric)
    miou_value = total_miou/len(segmented_sketches)    
    
    print("\n")
    print(f"Pixel Accuracy: {round(P_value,2)}")
    print(f"Stroke Accuracy: {round(C_value,2)}")
    print(f"mIoU: {round(miou_value,2)}")
    
    
if __name__ == '__main__':
    # mp.set_start_method('spawn')
    args = default_argument_parser().parse_args()
    main(args)


