import models
import torch
from torchvision.transforms import InterpolationMode
BICUBIC = InterpolationMode.BICUBIC
import torch.nn.functional as F
from utils import setup, get_similarity_map, display_segmented_sketch
from vpt.launch import default_argument_parser
from torchvision.transforms import Compose, Resize, ToTensor, Normalize
from collections import OrderedDict
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

def main(args):
    
    cfg = setup(args)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    preprocess =  Compose([Resize((224, 224), interpolation=BICUBIC), ToTensor(),
    Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))])

    Ours, preprocess = models.load("CS-ViT-B/16", device=device,cfg=cfg,zero_shot=False)
    state_dict = torch.load(cfg.checkpoint_path)
    
    # Trained on 2 gpus so we need to remove the prefix "module." to test it on a single GPU
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:] # remove `module.`
        new_state_dict[name] = v 
    Ours.load_state_dict(new_state_dict)     
    Ours.eval()     
    
    
    sketch_img_path = cfg.sketch_path
    classes = ['tree','bench','grass'] # set the condidate classes here
    
    colors = plt.get_cmap("tab10").colors    
    classes_colors = colors[:len(classes)]

    pil_img = Image.open(sketch_img_path).convert('RGB')
    binary_sketch = np.array(pil_img)
    binary_sketch = np.where(binary_sketch>0, 255, binary_sketch)
    sketch_tensor = preprocess(pil_img).unsqueeze(0).to(device)

    with torch.no_grad():
        text_features = models.encode_text_with_prompt_ensemble(Ours, classes, device,no_module=True)
        redundant_features = models.encode_text_with_prompt_ensemble(Ours, [""], device,no_module=True)            

    num_of_tokens = cfg.MODEL.PROMPT.NUM_TOKENS
    
    with torch.no_grad():
        sketch_features = Ours.encode_image(sketch_tensor,layers=12,text_features=text_features-redundant_features,mode="test")
        sketch_features = sketch_features / sketch_features.norm(dim=1, keepdim=True)
    similarity = sketch_features @ (text_features - redundant_features).t()
    patches_similarity = similarity[0, num_of_tokens +1:, :]
    pixel_similarity = get_similarity_map(patches_similarity.unsqueeze(0),pil_img.size)
    pixel_similarity[pixel_similarity<cfg.threshold] = 0
    pixel_similarity_array = pixel_similarity.cpu().numpy().transpose(2,0,1)
    
    display_segmented_sketch(pixel_similarity_array,binary_sketch,classes,classes_colors)

        
if __name__ == '__main__':
    # mp.set_start_method('spawn')
    args = default_argument_parser().parse_args()
    main(args)


