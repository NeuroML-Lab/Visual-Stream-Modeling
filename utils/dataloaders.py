import numpy as np
import torch
import os 
import pickle 
from skimage.transform import resize
from torchvision import transforms
from PIL import Image

preprocess = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((224,224)),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

preprocess_language = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((224,224)),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

gaussian_blur = transforms.GaussianBlur(kernel_size=5)
random_resized_crop = transforms.RandomResizedCrop(size=(180,180))
resized = transforms.Resize((224,224))

class Dataset_language(torch.utils.data.Dataset):
    def __init__(self, mode = 'train', brain_region = 'ventral_visual_data', 
                data_path = 'data/', model_type='nc', caption_type='clip',image_type='images', training_type='only_single_captions'):
        with open(os.path.join(data_path,brain_region+'_splits_1257.pickle'), 'rb') as pickle_file:
            self.splits = pickle.load(pickle_file) 
        self.ids = self.splits[mode]
        self.data_path = data_path
        with open(os.path.join(data_path,brain_region+'_1257_' + model_type + '.pickle'), 'rb') as pickle_file:
            self.stimuli_response_dict = pickle.load(pickle_file) 
        self.resp_sizes = np.load(os.path.join(data_path,brain_region+'_resp_sizes_1257_' + model_type + '.npy'))
        self.total_size = len(self.ids) 
        self.n_neurons =  sum(self.resp_sizes)
        self.caption_type = caption_type
        self.image_type = image_type
        self.training_type=training_type
    
    def __len__(self):
        'Denotes the total number of samples'
        return len(self.ids) 
    
    def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        data_ix = self.ids[index]
        X_single, X_dense = None, None
        if self.training_type == 'only_single_captions':
            if self.caption_type == 'mpnet':
                X_single = np.load(os.path.join('data/captions/mpnet_captions/',str(data_ix)+'.npy'))
                X_single = np.mean(X_single, axis = 0)
            elif self.caption_type == 'clip':
                X_single = np.load(os.path.join('data/captions/clip_captions/',str(data_ix)+'.npy'))
                X_single = np.mean(X_single, axis = 0)
        elif self.training_type in ['only_dense_captions', 'only_dense_captions_pre_images', 'only_dense_captions_images']:
            if self.caption_type == 'clip':
                X_dense = np.load(os.path.join('data/captions/clip_dense_captions/',str(data_ix)+'.npy'))
                X_dense = np.transpose(X_dense, (2, 0, 1))
            elif self.caption_type == 'mpnet':
                X_dense = np.load(os.path.join('data/captions/mpnet_dense_captions/',str(data_ix)+'.npy'))
                X_dense = np.transpose(X_dense, (2, 0, 1))
        elif self.training_type in ['only_dense_selected_captions','only_dense_selected_captions_pre_images','only_dense_selected_captions_images']:
            if self.caption_type == 'clip':
                X_dense = np.load(os.path.join('data/captions/clip_selected_dense_captions_uniform/',str(data_ix)+'.npy'))
                X_dense = np.transpose(X_dense, (2, 0, 1))
            elif self.caption_type == 'mpnet':
                X_dense = np.load(os.path.join('data/captions/mpnet_selected_dense_captions_uniform/',str(data_ix)+'.npy'))
                X_dense = np.transpose(X_dense, (2, 0, 1))
        elif self.training_type in ['single_dense_captions','single_dense_captions_images']:
            if self.caption_type == 'clip':
                X_dense = np.load(os.path.join('data/captions/clip_dense_captions/',str(data_ix)+'.npy'))
                X_dense = np.transpose(X_dense, (2, 0, 1))
                X_single = np.load(os.path.join('data/captions/clip_captions/',str(data_ix)+'.npy'))
                X_single = np.mean(X_single, axis = 0)
            elif self.caption_type == 'mpnet':
                X_dense = np.load(os.path.join('data/captions/mpnet_dense_captions/',str(data_ix)+'.npy'))
                X_dense = np.transpose(X_dense, (2, 0, 1))
                X_single = np.load(os.path.join('data/captions/mpnet_captions/',str(data_ix)+'.npy'))
                X_single = np.mean(X_single, axis = 0)
        
        if X_dense is not None:
            X_dense = torch.from_numpy(X_dense).float()
            X_dense = torch.nan_to_num(X_dense, nan=0.0)
        if X_single is not None:
            X_single = torch.from_numpy(X_single).float()
            X_single = torch.nan_to_num(X_single, nan=0.0)

        X_img = Image.open(os.path.join('data/'+self.image_type+'/',str(data_ix)+'.jpg'))
        X_img = preprocess_language(X_img)

        # X = np.asarray(image).astype('float32') 
        y = []
        subjects = [1,2,5,7]
        for i,sub in enumerate(subjects):
            respi = self.stimuli_response_dict[data_ix][sub]
            resp_size = self.resp_sizes[i]
            if respi is None:
                respi = np.ones(resp_size)*(-999.0) ##[-999.0 for i in range(resp_size)]
            y = np.concatenate((y, respi))
        if X_single is None:
            X_single = torch.tensor(0)
        if X_dense is None:
            X_dense = torch.tensor(0)
        return index, X_single, X_dense, X_img, y

class Dataset_visual(torch.utils.data.Dataset):

    def __init__(self, mode = 'train', brain_region = 'ventral_visual_data', dim=(224, 224),  n_channels = 3, 
                data_path = 'data/', model_type='nc', augmented_2N_data = False, image_type='images'):
           
        with open(os.path.join(data_path,brain_region+'_splits_1257.pickle'), 'rb') as pickle_file:
            self.splits = pickle.load(pickle_file) 
        self.ids = self.splits[mode]
        self.data_path = data_path
        with open(os.path.join(data_path,brain_region+'_1257_' + model_type + '.pickle'), 'rb') as pickle_file:
            self.stimuli_response_dict = pickle.load(pickle_file) 
        self.resp_sizes = np.load(os.path.join(data_path,brain_region+'_resp_sizes_1257_' + model_type + '.npy'))
        self.dim = dim
        self.total_size = len(self.ids) 
        self.n_channels = n_channels
        self.n_neurons =  sum(self.resp_sizes)
        self.augmented_2N_data = augmented_2N_data
        self.image_type = image_type


    def __len__(self):
        'Denotes the total number of samples'
        return len(self.ids) 
  
    def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        data_ix = self.ids[index]
        X = Image.open(os.path.join('data/'+self.image_type+'/',str(data_ix)+'.jpg'))
        # X = np.asarray(image).astype('float32') 
        y = []
        subjects = [1,2,5,7]
        for i,sub in enumerate(subjects):
            respi = self.stimuli_response_dict[data_ix][sub]
            resp_size = self.resp_sizes[i]
            if respi is None:
                respi = np.ones(resp_size)*(-999.0) ##[-999.0 for i in range(resp_size)]
            y = np.concatenate((y, respi))
        X = preprocess(X)
        if self.augmented_2N_data:
            X1 = gaussian_blur(X)
            X2 = resized(random_resized_crop(X))
            return X, X1, X2, y
        return index, X, y



    
