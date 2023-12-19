import torch
import numpy as np
from torchvision.transforms import InterpolationMode
BICUBIC = InterpolationMode.BICUBIC
from vpt.src.configs.config import get_cfg
import os
from time import sleep
from random import randint
from vpt.src.utils.file_io import PathManager
import matplotlib.pyplot as plt
from matplotlib.colors import rgb_to_hsv, hsv_to_rgb
import warnings
warnings.filterwarnings("ignore")

def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)

    output_dir = cfg.OUTPUT_DIR
    lr = cfg.SOLVER.BASE_LR
    wd = cfg.SOLVER.WEIGHT_DECAY
    output_folder = os.path.join(
        cfg.DATA.NAME, cfg.DATA.FEATURE, f"lr{lr}_wd{wd}")

    # train cfg.RUN_N_TIMES times
    count = 1
    while count <= cfg.RUN_N_TIMES:
        output_path = os.path.join(output_dir, output_folder, f"run{count}")
        # pause for a random time, so concurrent process with same setting won't interfere with each other. # noqa
        sleep(randint(3, 30))
        if not PathManager.exists(output_path):
            PathManager.mkdirs(output_path)
            cfg.OUTPUT_DIR = output_path
            break
        else:
            count += 1
            
    cfg.freeze()
    return cfg


def get_similarity_map(sm, shape):
    
    # sm: torch.Size([1, 196, 1]) 
    # min-max norm
    sm = (sm - sm.min(1, keepdim=True)[0]) / (sm.max(1, keepdim=True)[0] - sm.min(1, keepdim=True)[0]) # torch.Size([1, 196, 1])

    # reshape
    side = int(sm.shape[1] ** 0.5) # square output, side = 14
    sm = sm.reshape(sm.shape[0], side, side, -1).permute(0, 3, 1, 2) 

    # interpolate
    sm = torch.nn.functional.interpolate(sm, shape, mode='bilinear') 
    sm = sm.permute(0, 2, 3, 1) 
    
    return sm.squeeze(0)


def display_segmented_sketch(pixel_similarity_array,binary_sketch,classes,classes_colors,save_path=None):
    # Find the class index with the highest similarity for each pixel
    class_indices = np.argmax(pixel_similarity_array, axis=0)
    # Create an HSV image placeholder
    hsv_image = np.zeros(class_indices.shape + (3,))  # Shape (512, 512, 3)
    hsv_image[..., 2] = 1  # Set Value to 1 for a white base
    
    # Set the hue and value channels
    for i, color in enumerate(classes_colors):
        rgb_color = np.array(color).reshape(1, 1, 3)
        hsv_color = rgb_to_hsv(rgb_color)
        mask = class_indices == i
        if i < len(classes):  # For the first N-2 classes, set color based on similarity
            hsv_image[..., 0][mask] = hsv_color[0, 0, 0]  # Hue
            hsv_image[..., 1][mask] = pixel_similarity_array[i][mask] > 0  # Saturation
            hsv_image[..., 2][mask] = pixel_similarity_array[i][mask]  # Value
        else:  # For the last two classes, set pixels to black
            hsv_image[..., 0][mask] = 0  # Hue doesn't matter for black
            hsv_image[..., 1][mask] = 0  # Saturation set to 0
            hsv_image[..., 2][mask] = 0  # Value set to 0, making it black
    
    mask_tensor_org = binary_sketch[:,:,0]/255
    hsv_image[mask_tensor_org==1] = [0,0,1]

    # Convert the HSV image back to RGB to display and save
    rgb_image = hsv_to_rgb(hsv_image)

    # Calculate centroids and render class names
    for i, class_name in enumerate(classes):
        mask = class_indices == i
        if np.any(mask):
            y, x = np.nonzero(mask)
            centroid_x, centroid_y = np.mean(x), np.mean(y)
            plt.text(centroid_x, centroid_y, class_name, color=classes_colors[i], ha='center', va='center',fontsize=14,   # color=classes_colors[i]
            bbox=dict(facecolor='lightgrey', edgecolor='none', boxstyle='round,pad=0.2', alpha=0.8))

    # Display the image with class names
    plt.imshow(rgb_image)
    plt.axis('off')
    plt.tight_layout()
    # plt.show()
    
    if save_path:
        save_dir = "/".join(save_path.split("/")[:-1])
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
        
    else:
        plt.show()
