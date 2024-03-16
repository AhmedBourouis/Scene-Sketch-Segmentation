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
import nltk
import string
import warnings
from bresenham import bresenham
import scipy
from scipy import stats
from sklearn.metrics import accuracy_score


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
    
    save_dir = "/".join(save_path.split("/")[:-1])
    if save_dir !='':
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
        
    else:
        plt.show()


def sketch_text_pairs(sketch_batch,captions,max_classes=3):
    all_sketches = []
    all_classes = []
    all_captions = []   
    for (sketch,caption) in zip(sketch_batch,captions):
        caption = caption.replace('\n',' ')
        translator = str.maketrans('', '', string.punctuation)
        caption = caption.translate(translator).lower()
        words = nltk.word_tokenize(caption)
        classes = get_noun_phrase(words)
        classes = list(set(classes))
        if len(classes) >max_classes:
            classes = classes[:max_classes]
        if len(classes) ==0:
            classes = caption
            sketch = sketch.unsqueeze(0)
        else:
            sketch = sketch.repeat(len(classes),1,1,1) 
            caption = [caption]*len(classes)
        all_sketches.append(sketch)
        all_classes.append(classes)
        all_captions.append(caption)
        
    return torch.cat(all_sketches), flatten(all_classes), flatten(all_captions)
   
   
def flatten(lst):
    result = []
    for i in lst:
        if isinstance(i, list):
            result.extend(flatten(i))
        else:
            result.append(i)
    return result


def get_noun_phrase(tokenized):
    # Taken from Su Nam Kim Paper...
    grammar = r"""
        NBAR:
            {<NN.*|JJ>*<NN.*>}  # Nouns and Adjectives, terminated with Nouns
        NP:
            {<NBAR>}
            {<NBAR><IN><NBAR>}  # Above, connected with in/of/etc...
    """
    chunker = nltk.RegexpParser(grammar)

    chunked = chunker.parse(nltk.pos_tag(tokenized))
    continuous_chunk = []
    current_chunk = []

    for subtree in chunked:
        if isinstance(subtree, nltk.Tree):
            current_chunk.append(' '.join([token for token, pos in subtree.leaves()]))
        elif current_chunk:
            named_entity = ' '.join(current_chunk)
            if named_entity not in continuous_chunk:
                continuous_chunk.append(named_entity)
                current_chunk = []
        else:
            continue

    return continuous_chunk


def tensor_to_binary_img(tensor,device):
    mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(-1, 1, 1).to(device)
    std = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(-1, 1, 1).to(device)
    images_unnormalized = tensor * std + mean
    images_gray = images_unnormalized.mean(dim=1)
    threshold = 0.5
    binary_images = (images_gray > threshold).float()
    binary_images = binary_images.unsqueeze(1)
    binary_images = binary_images.repeat(1,3,1,1)
    return binary_images

def zero_clapping(similarity_maps,threshold):
    batch_size= similarity_maps.shape[0]
    indices = torch.arange(batch_size).to(similarity_maps.device)
    diagonal_ps = similarity_maps[torch.arange(batch_size), :, :, indices]
    max_ps = torch.max(diagonal_ps, dim=1)[0]
    min_ps = torch.min(diagonal_ps, dim=1)[0]
    diagonal_ps = (diagonal_ps - min_ps[:, None]) / (max_ps[:, None] - min_ps[:, None])
    diagonal_ps[diagonal_ps < threshold] = 0
    diagonal_ps[diagonal_ps >= threshold] = 1
    weights = diagonal_ps
    return weights


def get_threshold(learnable_threshold):
    noise = torch.normal(mean=0, std=0.005, size=learnable_threshold.shape).to(learnable_threshold.device)
    learnable_threshold.data.add_(noise)
    threshold_value = 0.4 + 0.5 * torch.sigmoid(learnable_threshold)
    return threshold_value


def get_train_classes(dataset,max_classes=3):
    train_classes = []
    for i, (_,caption) in enumerate(dataset):
        caption = caption.replace('\n',' ')
        translator = str.maketrans('', '', string.punctuation)
        caption = caption.translate(translator).lower()
        words = nltk.word_tokenize(caption)
        classes = get_noun_phrase(words)
        # remove synonyms in classes
        classes = list(set(classes))
        if len(classes) >max_classes:
            classes = classes[:max_classes]
        if len(classes) ==0:
            classes = caption
        train_classes.append(classes)
    train_classes = np.unique(flatten(train_classes)) 
    return train_classes

def triplet_loss_func_L1(sketch_embeddings, class_embeddings, labels, margin=0.2):
    batch_size = sketch_embeddings.shape[0]

    # pairwise distance matrix using L1 distance
    sketch_embeddings = sketch_embeddings.contiguous()
    dists = torch.cdist(sketch_embeddings, class_embeddings, p=1)  
    # diagonal mask (for positive samples)
    diag_mask = torch.eye(batch_size, dtype=bool).to(sketch_embeddings.device)
    # Positive distances (diagonal elements)
    positive_dists = dists[diag_mask]
    # Class mask (for negative samples)
    class_mask = labels[:, None] == labels[None, :]  # Shape: (batch_size, batch_size)
    class_mask = class_mask.to(sketch_embeddings.device)
    # Negative distances
    negative_dists = dists.clone()
    negative_dists[diag_mask | class_mask] = float('inf')  # Set diagonal elements and same class elements to a large value
    min_negative_dists, _ = torch.min(negative_dists, dim=1)
    # Compute triplet loss
    loss = torch.clamp(margin + positive_dists - min_negative_dists, min=0.0)

    return loss.mean()

def visualize_attention_maps_with_tokens(pixel_similarity, tokens):
    # Convert the tensor to a numpy array and transpose it to match the dimensions required by imshow
    attention_maps = pixel_similarity.numpy().transpose(2, 0, 1)

    # Create a subplot for each attention map
    num_attention_maps = attention_maps.shape[0]
    fig, axes = plt.subplots(1, num_attention_maps, figsize=(15, 5))

    # Plot each attention map with corresponding text token
    for i in range(num_attention_maps):
        ax = axes[i]
        ax.imshow(attention_maps[i], cmap='gray', vmin=0, vmax=1)
        # ax.set_title(f'Attention Map {i+1}')
        ax.axis('off')

        # Add the corresponding text token as annotation below the attention map
        ax.annotate(tokens[i], xy=(0.5, -0.1), xycoords='axes fraction', ha='center')

    plt.tight_layout()
    # plt.savefig('attention_maps.png')
    plt.show()  
    
## Sketch Preprocessing tools

def pen_state_to_strokes(sketches):
    strokes=[]
    i_prev=0
    for i in range(len(sketches)):
        if sketches[i,2]==1:
            strokes.append(sketches[i_prev:i+1])
            i_prev=i+1
    return strokes

def preprocess(sketch_points, side):
    sketch_points = sketch_points.astype(np.float)
    sketch_points[:, :2] = sketch_points[:, :2] / np.array([256, 256])
    sketch_points[:, :2] = sketch_points[:, :2] * side
    sketch_points = np.round(sketch_points)
    return sketch_points

def mydrawPNG(vector_image, Side):
    raster_image = np.zeros((int(Side), int(Side)), dtype=np.float32)
    initX, initY = int(vector_image[0, 0]), int(vector_image[0, 1])
    pixel_length = 0

    for i in range(0, len(vector_image)):
        if i > 0:
            if vector_image[i - 1, 2] == 1:
                initX, initY = int(vector_image[i, 0]), int(vector_image[i, 1])

        cordList = list(bresenham(initX, initY, int(vector_image[i, 0]), int(vector_image[i, 1])))
        pixel_length += len(cordList)

        for cord in cordList:
            if (cord[0] > 0 and cord[1] > 0) and (cord[0] < Side and cord[1] < Side):
                raster_image[cord[1], cord[0]] = 255.0
        initX, initY = int(vector_image[i, 0]), int(vector_image[i, 1])

    raster_image = scipy.ndimage.binary_dilation(raster_image) 
    return raster_image

def rasterize_Sketch(stroke_points,side): # points that constitute one stroke [ # of points , (x,y,penup)]
    stroke_points = preprocess(stroke_points, side)
    raster_stroke = mydrawPNG(stroke_points, side)
    
    return raster_stroke

def prerender_stroke(stroke_list,side= 512):# fig, xlim=[0,255], ylim=[0,255]):
    R = []
    for stroke in stroke_list:
        stroke = np.array([stroke,])
        R.append( torch.tensor(rasterize_Sketch(stroke[0],side)).unsqueeze(0) )
    return torch.stack(R, 0)

def pixel_level_segmentation(strokes_seg, labels, all_classes,size):
    num_classes = len(all_classes)
    sketch_seg = np.zeros((num_classes, size,size))  # Initialize segmentation array
    
    blank_pix_index = all_classes.index('blank_pixel')  # index of 'blank_pix'
    # Initially assign all pixels to 'blank_pix'
    sketch_seg[blank_pix_index] = np.ones((size, size))

    for stroke, label in zip(strokes_seg, labels):
        # Get the class index for the given label
        class_index = all_classes.index(label)

        # For each pixel in the stroke, assign its class in the sketch segmentation
        # Before assigning, remove the pixel from the 'blank_pix' class
        sketch_seg[class_index] += stroke  # This works because stroke values are either 0 or 1
        sketch_seg[blank_pix_index] -= stroke
        # sketch_seg[unlabeled_pix_index] -= stroke

    # Now each pixel in sketch_seg has a one-hot encoding across all classes.
    # We convert this to a single channel with the class labels as integers.
    pixel_level_seg = np.argmax(sketch_seg, axis=0)
    return pixel_level_seg


def compute_accuracy(strokes_seg, labels,classes, gt_seg, pred_seg):
    blank_pix_index = classes.index('blank_pixel')
    gt_pixels = gt_seg.flatten()[gt_seg.flatten() != blank_pix_index]
    pred_pixels = pred_seg.flatten()[pred_seg.flatten() != blank_pix_index]
    gt_labels_indices = [classes.index(label) for label in labels]
    # Compute pixel accuracy
    if len(gt_pixels) != len(pred_pixels):
        add_pix = len(gt_pixels) - len(pred_pixels)
        pred_pixels = np.append(pred_pixels, np.ones(add_pix)) #*unlabeled_pix_index
        
    P_metric = accuracy_score(gt_pixels, pred_pixels)
    # Compute stroke accuracy
    # For each stroke, determine the most frequent pixel label in prediction
    pred_strokes = []
    for stroke in strokes_seg:
        pred_stroke_pixels = pred_seg[stroke==1] 
        if len(pred_stroke_pixels) == 0: 
            pred_strokes.append(blank_pix_index)
            continue
        pred_strokes.append(stats.mode(pred_stroke_pixels)[0][0])
    C_metric = accuracy_score(gt_labels_indices, pred_strokes)
    return P_metric, C_metric

def compute_miou(gt_seg, pred_seg, all_classes):
    num_classes = len(all_classes)
    # Initialize a matrix to hold the sum of IoUs for each class
    class_iou_sum = np.zeros(num_classes, dtype=np.float32)
    class_counts = np.zeros(num_classes, dtype=np.int32)

    for c in range(num_classes):
        if c == 0:
            continue
        # Compute the intersection and union for the current class
        intersection = np.sum((gt_seg == c) & (pred_seg == c))
        union = np.sum((gt_seg == c) | (pred_seg == c))
        
        # If the union is 0, this class is not present in either ground truth or prediction
        if union == 0:
            continue
        
        # Increment the class IoU sum and class counts
        class_iou_sum[c] += intersection / union
        class_counts[c] += 1
    
    # Compute mean IoU only for classes that are present (avoid division by 0)
    miou = np.sum(class_iou_sum[class_counts > 0]) / np.sum(class_counts[class_counts > 0])
    return miou
