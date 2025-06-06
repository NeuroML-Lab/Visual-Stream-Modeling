# Visual-Stream-Modeling

This repository contains the code for the following paper - [Modeling the Human Visual System: Comparative Insights from Response-Optimized and Task-Optimized Vision Models, Language Models, and different Readout Mechanisms](https://arxiv.org/abs/2410.14031). The code analyses various model encoders and readout mechanisms to model the visual cortex of the human brain. 

## Dataset

The dataset used is a preprocessed version of the Natural Scenes Dataset (NSD), and experiments are done only using 4 (1,2,5 and 7) of the 8 subjects. The preprocessed version can be downloaded from [here](https://osf.io/wh35d/?view_only=202cd5926f094b2693095f744ffdafbc). Create a folder 'data' and store the downloaded dataset inside this folder. Each dataset folder is named after a specific region of the visual cortex and contain the following files - 

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

There will be a seperate folder called 'images' with MS-COCO images used in the NSD experiments.

## Training

In order to train response optimized models with visual input - 

```bash
python3 vanilla_training.py --brain_region "brain_region" --readout "readout" --alpha "only_for_linear_ridge_regression_readouts"
```

Readouts can have the following values - linear_ridge, semantic_transformer, spatial_linear and gaussian2d. Each of them are explained in more details in section 2.2. The alpha values is only used in case of the linear_ridge readout. Brain regions can have the following values - V1v_data, V2v_data, V3v_data, V1d_data, V2d_data, V3d_data, v4_data, ventral_visual_data, dorsal_visual_data and lateral_visual_data.

In order to train task optimized models with visual input - 

```bash
python3 task_optimised_baselines.py --brain_region "brain_region" --readout "readout" --alpha "only_for_linear_ridge_regression_readouts" --task_optimised_model "resnet50 or alexnet"
```

In case you want to use only the first n layers of task optimized models, use the following - 

```bash
python3 task_optimised_baselines.py --brain_region "brain_region" --readout "readout" --alpha "only_for_linear_ridge_regression_readouts" --task_optimised_model "resnet50 or alexnet" --use_sub_layers --sub_layers n
```

In order to models with single caption language input (only linear_ridge readouts in this case) -

```bash
python3 language_stimuli.py --brain_region "brain_region" --alpha "float value" --llm_encoder "clip or mpnet" --training_type only_single_captions
```

In order to models with dense caption language input -

```bash
python3 language_stimuli.py --brain_region "brain_region" --readout "readout" --alpha "only_for_linear_ridge_regression_readouts" --llm_encoder "clip or mpnet" --training_type only_dense_captions
```
 The trained model are stored inside the directory 'outputs_paper'. 

## Evaluation

In order to evaluate response optimized models with visual input - 

```bash
python3 vanilla_training.py --brain_region "brain_region" --readout "readout" --alpha "only_for_linear_ridge_regression_readouts" --evaluate
```

In order to evaluate task optimized models with visual input - 

```bash
python3 task_optimised_baselines.py --brain_region "brain_region" --readout "readout" --alpha "only_for_linear_ridge_regression_readouts" --task_optimised_model "resnet50 or alexnet" --evaluate
```

```bash
python3 task_optimised_baselines.py --brain_region "brain_region" --readout "readout" --alpha "only_for_linear_ridge_regression_readouts" --task_optimised_model "resnet50 or alexnet" --use_sub_layers --sub_layers n --evaluate
```

In order to evaluate models with language input - 

```bash
python3 language_stimuli.py --brain_region "brain_region" --alpha "float value" --llm_encoder "clip or mpnet" --training_type only_single_captions --evaluate
```

```bash
python3 language_stimuli.py --brain_region "brain_region" --readout "readout" --alpha "only_for_linear_ridge_regression_readouts" --llm_encoder "clip or mpnet" --training_type only_dense_captions --evaluate
```

The test correlations are saved inside directory evaluations, and the individual correlations per valid voxel is saved in directory evaluations_paper.

## Pycortex Visualisations - 

A sample visualisation is explained inside the notebook 'plots.ipynb' for subject 1. The same process has to be repeated when comparing any pairs of encoders or readouts, while making sure to maintain the proper indices for the various subjects.

All the above code was run on either a 24G Nvidia RTX 4090 GPU (Cuda version 12.2) or a 46G Nvidia A40 GPU (Cuda Version 12.4). In case you face errors, please raise issues or reach out to the authors.
