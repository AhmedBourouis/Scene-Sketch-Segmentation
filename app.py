
import argparse
import gradio as gr
import os
from PIL import Image
import models
import torch
from torchvision.transforms import InterpolationMode
BICUBIC = InterpolationMode.BICUBIC
import torch.nn.functional as F
from utils import setup, get_similarity_map, display_segmented_sketch,visualize_attention_maps_with_tokens
from vpt.launch import default_argument_parser
from torchvision.transforms import Compose, Resize, ToTensor, Normalize
from collections import OrderedDict
import numpy as np
import matplotlib.pyplot as plt

args = default_argument_parser().parse_args()
cfg = setup(args)

device = "cuda" if torch.cuda.is_available() else "cpu"
Ours, preprocess = models.load("CS-ViT-B/16", device=device,cfg=cfg,train_bool=False)
state_dict = torch.load("checkpoint/sketch_seg_best_miou.pth")

# Trained on 2 gpus so we need to remove the prefix "module." to test it on a single GPU
new_state_dict = OrderedDict()
for k, v in state_dict.items():
    name = k[7:] # remove `module.`
    new_state_dict[name] = v 
Ours.load_state_dict(new_state_dict)     
Ours.eval()     
print("Model loaded successfully")
    
def run(sketch, caption, threshold, seed):
    
    # set the condidate classes here
    classes = [caption] 
    
    colors = plt.get_cmap("tab10").colors
    classes_colors = colors[1:len(classes)+1]

    # sketch = sketch['image']
    sketch = np.array(sketch)
    
    pil_img = Image.fromarray(sketch).convert('RGB')
    sketch_tensor = preprocess(pil_img).unsqueeze(0).to(device)

    with torch.no_grad():
        text_features = models.encode_text_with_prompt_ensemble(Ours, classes, device,no_module=True)
        redundant_features = models.encode_text_with_prompt_ensemble(Ours, [""], device,no_module=True)            

    num_of_tokens = 3
    with torch.no_grad():
        sketch_features = Ours.encode_image(sketch_tensor,layers=[12],text_features=text_features-redundant_features,mode="test").squeeze(0)
        sketch_features = sketch_features / sketch_features.norm(dim=1, keepdim=True)
    similarity = sketch_features @ (text_features - redundant_features).t()
    patches_similarity = similarity[0, num_of_tokens +1:, :]
    pixel_similarity = get_similarity_map(patches_similarity.unsqueeze(0),pil_img.size).cpu()
    # visualize_attention_maps_with_tokens(pixel_similarity, classes)
    pixel_similarity[pixel_similarity<threshold] = 0
    pixel_similarity_array = pixel_similarity.cpu().numpy().transpose(2,0,1)
    
    display_segmented_sketch(pixel_similarity_array,sketch,classes,classes_colors,live=True)
    
    rgb_image = Image.open('output.png')

    return rgb_image


css = """
{background-color: #808080} 
.feedback textarea {font-size: 24px !important}
"""


js = """
function createGradioAnimation() {
    var container = document.createElement('div');
    container.id = 'gradio-animation';
    container.style.fontSize = '2em';
    container.style.fontWeight = 'bold';
    container.style.textAlign = 'center';
    container.style.marginBottom = '20px';

    var text = 'Welcome to Gradio!';
    for (var i = 0; i < text.length; i++) {
        (function(i){
            setTimeout(function(){
                var letter = document.createElement('span');
                letter.style.opacity = '0';
                letter.style.transition = 'opacity 0.5s';
                letter.innerText = text[i];

                container.appendChild(letter);

                setTimeout(function() {
                    letter.style.opacity = '1';
                }, 50);
            }, i * 250);
        })(i);
    }

    var gradioContainer = document.querySelector('.gradio-container');
    gradioContainer.insertBefore(container, gradioContainer.firstChild);

    return 'Animation created';
}
"""


demo = gr.Interface(
    fn=run,
    theme="gstaff/sketch",#"gstaff/xkcd",   
    description='Upload a skecth [X] or draw your own.'\
                ' Check run examples down the page.',
    # css=css,
    inputs=[
        gr.Image(value=Image.new(
            'RGB', (512, 512), color=(255, 255, 255)), # tool='sketch', shape=(512, 512), 
            type='pil', label='Sketch'
            ),
        gr.Textbox(label="Caption", placeholder="Describe which objects to segment"),
        gr.Slider(label="Threshold", value=0.6, step=0.05, minimum=0, maximum=1),
        # gr.components.Number(label="Seed", default=0, precision=0), 
    ], 
    outputs=[gr.Image(type="pil", label="Segmented Sketch") ],
    examples=[
        ['demo/sketch_1.png', 'giraffe', 0.6],
        ['demo/sketch_2.png', 'tree', 0.6],
        ['demo/sketch_3.png', 'person', 0.6],
    ],
    allow_flagging=False,
    # layout="vertical",
    title="Scene Sketch Semantic Segmentation")

demo.launch(share=False)
