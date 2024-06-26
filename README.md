# Open Vocabulary Semantic Scene Sketch Understanding
This is the official implementation of the "Open Vocabulary Scene Sketch Semantic Understanding", CVPR 2024, paper, by [*Ahmed Bourouis*](https://ahmedbourouis.github.io/ahmed-bourouis/), [*Judith Ellen Fan*](https://profiles.stanford.edu/judith-fan), and [*Yulia Gryaditskaya*](https://yulia.gryaditskaya.com/).

[**Project webpage**](https://ahmedbourouis.github.io/Scene_Sketch_Segmentation/)

We present the first language-supervised scene sketch segmentation method. Our approach employs a dual-level network architecture, designed to effectively disentangle different object categories within scene sketches, utilizing only brief captions as guidance.    

<div align="center">
<img src="figs/teaser.png" width="800"/>
</div>

# News
- **21.05.2024**: Try out our [demo](https://huggingface.co/spaces/ahmedbrs/scene-sketch-seg)! 
- **26.02.2024**: Train code is released
- **26.02.2024**: Paper accepted at CVPR'24 :tada: 
- **05.12.2023**: Demo code is released.


# Inference
- Given a desirable set of categories (or a brief caption) for a given sketch image, we encode these categories with the ViT-based text encoder, fine-tuned with our model. 
- We compute the per-patch cosine similarity between the class embeddings and the scene sketch patch embeddings. The resulting similarity matrix represents the category label probabilities for each patch. 
- To generate a pixel-level similarity map, we reshape the resulting per-patch similarity maps and then upscale them to the dimensions of the original scene sketch using bi-cubic interpolation.

<div align="center">
<img src="figs/inference.gif" width="85%">

</div>

## Isolate individual categories
If we want to isolate just a few categories in the sketch, we only retain pixels with category-sketch similarity scores above a pre-set threshold value. Below we visualize this process for different threshold values. 
<div align="center">
<img src="figs/thresholding.png" width="800"/>
</div>

# Usage

## Running demo.py
- The version requirements of core dependencies.
```
pip install torch==1.9.1+cu111 torchvision==0.10.1+cu111 torchaudio==0.9.1 -f https://download.pytorch.org/whl/torch_stable.html

```
- Download checkpoint `sketch_seg_best_miou.pth` from [here](https://drive.google.com/drive/folders/1TdA5B-cZWJOZgZzVzHhAAIEoM9NOBFvM?usp=sharing) and save it in `checkpoint` folder. 


- A use case can be found in `demo.py`. From just a sketch image and candidate categories, we can generate scene sketch segmentation maps.
 
```
python demo.py --config-file vpt/configs/prompt/cub.yaml checkpoint_path checkpoint/sketch_seg_best_miou.pth sketch_path demo/sketch_1.png output_path demo/results/output.png threshold 0.6
```
### Hyper-parameters
- `config-file`:
  main config setups for experiments and explanations for each of them. 
- `checkpoint_path`:
  path for the model checkpoint.
- `sketch_path`:
  sketch image example path.
- `output_path`:
  path to save the output sketch.
- `threshold`:
  the threshold value for class-pixel similarity scores to retain.
 

## Training: 
- Download the dataset from [here (link to the dataset webpage)](https://cvssp.org/data/fscoco-seg/) and put it in "/DATA" directory.

- Run the following for training the model.
```
python train.py --config-file vpt/configs/prompt/cub.yaml MODEL.PROMPT.NUM_TOKENS 3 MODEL.PROMPT.LOG "first_run" save_every 5 learning_rate 1e-6 bz 16  WANDB False 
```

### Hyper-parameters
- `config-file`:
  main config setups for experiments and explanations for each of them. 
- `checkpoint_path`:
  path for the model checkpoint.
- `MODEL.PROMPT.NUM_TOKENS`:
  number of trainable visual tokens to use for CLIP fine-tuning.
- `MODEL.PROMPT.LOG`:
  log name.
- `save_every`:
  save model weights every # epochs.
- `learning_rate`:
  learning rate
- `bz`:
  training batch size.
- `WANDB`:
  whether to log your training to WandB or not.
  

# Citation
If you build on this code or compare to it, please cite:
```
@inproceedings{bourouis2024sketch,
title={Open Vocabulary Semantic Scene Sketch Understanding}, 
author={Ahmed Bourouis and Judith Ellen Fan and Yulia Gryaditskaya},
booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
year={2024}
}
```
# Questions
For any questions please contact Ahmed Bourouis (bourouis17@gmail.com) and Yulia Gryaditskaya (yulia.gryaditskaya@gmail.com).

## Acknowledgements

Our code is based on [CLIP_Surgery](https://github.com/xmed-lab/CLIP_Surgery) repository. We thank the authors for releasing their code. If you use our model and code, please consider citing this work as well.
