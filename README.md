# Visual-Stream-Modeling

This repository contains the code for the following paper - [Modeling the Human Visual System: Comparative Insights from Response-Optimized and Task-Optimized Vision Models, Language Models, and different Readout Mechanisms](https://arxiv.org/abs/2410.14031). The code analyses various model encoders and readout mechanisms to model the visual cortex of the human brain. 

## Dataset

The dataset used is a preprocessed version of the Natural Scenes Dataset (NSD), and experiments are done only using 4 (1,2,5 and 7) of the 8 subjects. The preprocessed version will be uploaded soon. As of now, please reach out to the authors in order to access it. Create a folder 'data' and store the downloaded dataset inside this folder. Each dataset folder is named after a specific region of the visual cortex and contain the following files - 

1. Noise Celing for individual subjects - nc_<sub_id>.npy
2. Noise Celing for all subjects that are valid (greater than 0) - noise_ceiling_1257_filtered.npy
3. 
