# Visual-Stream-Modeling

This repository contains the code for the following paper - [Modeling the Human Visual System: Comparative Insights from Response-Optimized and Task-Optimized Vision Models, Language Models, and different Readout Mechanisms](https://arxiv.org/abs/2410.14031). The code analyses various model encoders and readout mechanisms to model the visual cortex of the human brain. 

## Dataset

The dataset used is a preprocessed version of the Natural Scenes Dataset (NSD), and experiments are done only using 4 (1,2,5 and 7) of the 8 subjects. The preprocessed version will be uploaded soon. Create a folder 'data' and store the downloaded dataset inside this folder. Each dataset folder is named after a specific region of the visual cortex and contain the following files - 

1. Noise Celing
   1. individual subjects - nc_<sub_id>.npy
   2. Noise Celing for all subjects for all subjects - noise_ceiling_1257.npy
   3. Noise Celing for all subjects that are valid voxels (greater than 0) - noise_ceiling_1257_filtered.npy
2. Dictionary containing coco ids as the key with subject responses as values (coco_id : {subject_id : fmri_response}) -
   1. <brain_region>.pickle - responses for all voxels for all subjects
   2. <brain_region>_1257.pickle - responses for all voxels for subjects 1,2,5 and 7
   3. <brain_region>_1257_filtered.pickle - esponses for voxels with valid noise celings for subjects 1,2,5 and 7
3. train, val and test splits - <brain_region>_splits_1257.pickle
4. response sizes or number of voxels per subject - <brain_region>_resp_sizes_1257_filtered.npy

## Training

In order to train response optimized models with visual input - 

```bash
python3 vanilla_training.py --brain_region <brain_region> --readout <readout> --alpha <only for linear ridge regression readouts>
```

Readouts can have the following values - linear_ridge, semantic_transformer, spatial_linear and gaussian2d. Each of them are explained in more details in section 2.2. The alpha values is only used in case of the linear_ridge readout. Brain regions can have the following values - 

## Evaluation

```bash
python3 vanilla_training.py --brain_region <brain_region> --readout <readout> --alpha <only for linear ridge regression readouts> --evaluate
```
