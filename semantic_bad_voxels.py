import warnings
warnings.filterwarnings("ignore", category=UserWarning)

import os
import numpy as np
import torch
from utils.train_utils import *
from utils.model_utils import *
from utils.dataloaders import Dataset_language
from models.model_text import *
from neuralpredictors.layers.readouts import SpatialXFeatureLinear, Gaussian2d
from utils.readouts import SemanticSpatialTransformer
from torch.autograd import Variable
from itertools import repeat
from utils.load_submodels import *
from utils.utils import *
from utils.text_readouts import *
from tqdm import tqdm
import argparse
import logging
import matplotlib.pyplot as plt
import yaml

schedule = [1e-4]
criterion_mse = masked_MSEloss
criterion_huber = masked_Huberloss
criterion_corr = masked_Correlation_loss
patience = 20
iter_tracker = 0 
accumulate_gradient= 4
n_epochs = 100
n_feats = 48

NUM_FEATURES_MPNET = 768
NUM_FEATURES_CLIP = 512

config = read_yaml('configurations/saved_models.yaml')

def compute_predictions_only_single_captions(loader, readout, reshape=True, stack=True, return_lag=False):
    y, y_hat = [], []
    for id, x_batch,_, img_batch, y_val in (loader):
        y_mod = readout(x_batch.cuda()).data.cpu().numpy()
        y.append(y_val.numpy())
        y_hat.append(y_mod)
    if stack:
        y, y_hat = np.vstack(y), np.vstack(y_hat) 
    return y, y_hat ##true,pred

def compute_predictions_only_dense_captions(args, loader, model, reshape=True, stack=True, return_lag=False):
    y, y_hat = [], []
    for id, _,x_batch, img_batch, y_val in (loader):
        if args.readout == 'spatial_linear' or args.readout == 'gaussian2d':
            y_mod = model(x_batch.cuda()).data.cpu().numpy()
        elif args.readout == 'semantic_transformer':
            y_mod = model(x_batch.cuda(), img_batch.cuda()).data.cpu().numpy()
        elif 'linear_ridge' in args.readout:
            y_mod = model(x_batch.cuda()).data.cpu().numpy()
        y.append(y_val.numpy())
        y_hat.append(y_mod)
    if stack:
        y, y_hat = np.vstack(y), np.vstack(y_hat) 
    return y, y_hat ##true,pred

def compute_predictions_only_dense_captions_images(args, loader, model, reshape=True, stack=True, return_lag=False):
    y, y_hat = [], []
    for id, _,x_batch, img_batch, y_val in (loader):
        y_mod = model(x_batch.cuda(),img_batch.cuda()).data.cpu().numpy()
        y.append(y_val.numpy())
        y_hat.append(y_mod)
    if stack:
        y, y_hat = np.vstack(y), np.vstack(y_hat) 
    return y, y_hat ##true,pred

def compute_predictions_single_dense_captions(args, loader, model, reshape=True, stack=True, return_lag=False):
    y, y_hat = [], []
    for id, x_single_batch, x_dense_batch, img_batch, y_val in (loader):
        if args.readout == 'spatial_linear':
            print("lmao")
            print(1/0)
        elif args.readout == 'semantic_transformer':
            y_mod = model(x_dense_batch.cuda(),img_batch.cuda(), x_single_batch.cuda()).data.cpu().numpy()
        y.append(y_val.numpy())
        y_hat.append(y_mod)
    if stack:
        y, y_hat = np.vstack(y), np.vstack(y_hat) 
    return y, y_hat ##true,pred

def evaluate_trained_model(args, readout, test_generator):
    readout.eval()
    true, preds = compute_predictions(test_generator, readout)
    test_corr = compute_scores_evaluation(true, preds)
    noise_ceiling = np.load('data/'+args.brain_region+'/noise_ceiling_1257_filtered_clustering_bad.npy')

    evaluation_dir = args.evaluation_dir
    if not os.path.isdir(evaluation_dir):
        os.mkdir(evaluation_dir)

    test_corr_norm = test_corr/noise_ceiling
    plt.scatter(noise_ceiling,np.array(test_corr))
    plt.xlabel('NC Label')
    plt.ylabel('Test Corr Label')
    plt.legend()
    plt.savefig(evaluation_dir+'/metrics.png') 

    np.save(evaluation_dir+'/test_corr.npy', np.array(test_corr))
    np.save(evaluation_dir+'/test_corr_norm.npy', np.array(test_corr_norm))

    test_corr, test_corr_norm = np.nanmean(test_corr), np.nanmean(test_corr_norm)
    with open(evaluation_dir+'/metrics.txt', 'w') as file:
        file.write('test_corr :: '+str(test_corr)+'\n'+'test_corr_norm :: '+str(test_corr_norm))
    return

def load_models(args, readout='semantic_transformer'):
    model_IT, core_IT, readout_IT = load_vanilla_IT_model(config[args.brain_region]['n_neurons'], config[args.brain_region]['it']['vanilla'][readout], readout)
    num_features = core_IT((torch.randn(1, 3, 224, 224))).view(-1).shape[0]
    core_IT.eval()
    for param in core_IT.parameters():
        param.requires_grad = False
    return core_IT, num_features

def define_args():
    parser = argparse.ArgumentParser(description='Encoding model')
    parser.add_argument('--brain_region', default='ventral_visual_data', type=str)
    parser.add_argument('--iteration', default=0, type=int)
    parser.add_argument('--start_epoch', default=0, type=int)
    parser.add_argument('--best_corr', default=0.0, type=float)
    parser.add_argument('--saved_model', default=None, type=str)
    parser.add_argument('--training_type', default='only_single_captions', type=str)
    parser.add_argument('--llm_encoder', default='mpnet', type=str)
    parser.add_argument('--readout', default='semantic_transformer', type=str)
    parser.add_argument('--evaluate', action='store_true')
    parser.add_argument('--evaluation_dir', default='evaluations_paper',type=str)
    parser.add_argument('--alpha', default=0.0, type=float)
    args = parser.parse_args()
    return args

def train_only_single_captions(args, training_generator, validation_generator, brain_region, readout):

    iteration = args.iteration
    start_epoch = args.start_epoch
    best_corr = args.best_corr

    model_dir = './outputs_paper/saved_'+args.brain_region+'_bv_' + args.training_type + '_' + args.llm_encoder
    if not os.path.isdir(model_dir):
        os.mkdir(model_dir)
    else:
        print("Training already done -____-")
        print(1/0)
    logging.basicConfig(filename='./outputs_paper/saved_'+args.brain_region+'_bv_' + args.training_type  + '_' + args.llm_encoder +'.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logging.info("start training with %d %d", start_epoch, iteration)

    opt = torch.optim.Adam
    lr = schedule[0]
    print('Training with learning rate', lr)
    optimizer = opt(readout.parameters(), lr=lr)
    optimizer.zero_grad()
    readout.cuda()
    iter_tracker = 0 

    for epoch in range(start_epoch,n_epochs):
        for img_ids, x_batch, _, img_batch, y_batch in tqdm(training_generator):
            outputs = readout(x_batch.cuda())
            loss = criterion_mse(outputs, y_batch.cuda().float()) + criterion_corr(outputs.cuda(), y_batch.cuda().float())
            loss = loss + readout.l2_regularization().cuda()
            loss.backward(retain_graph=True)
            iteration += 1
            if iteration % accumulate_gradient == accumulate_gradient - 1:
                optimizer.step()
                optimizer.zero_grad()
            if iteration%500==0:
                readout.eval()
                true, preds = compute_predictions_only_single_captions(validation_generator, readout)
                val_corr = compute_scores(true, preds)
                logging.info('Epoch :: %d, Iteration :: %d, Validation Correlation :: %f', epoch, iteration, val_corr)
                is_best = val_corr >= best_corr
                readout.train()
                if is_best:
                    best_corr = val_corr.copy()
                    iter_tracker = 0  
                    logging.info('saving model %d',len(readout.state_dict().keys()))
                    model_base = '%s_%d_%d' % (brain_region, epoch, iteration)
                    logging.info('model :: %s', model_base)
                    save_checkpoint({'epoch': epoch + 1,
                                        'state_dict': readout.state_dict()},
                                        is_best = is_best,
                                        checkpoint = model_dir, model_str = model_base)
                else:
                    iter_tracker += 1
                    if iter_tracker == patience: 
                        print('Training complete')
                        break 
                torch.cuda.empty_cache()
            if iter_tracker == patience:
                break
    return

def train_only_dense_captions(args, training_generator, validation_generator, brain_region, model):

    iteration = args.iteration
    start_epoch = args.start_epoch
    best_corr = args.best_corr

    model_dir = './outputs_paper/'+args.brain_region+'_bv_' + args.training_type + '_' + args.llm_encoder + '_' + args.readout
    if not os.path.isdir(model_dir):
        os.mkdir(model_dir)
    else:
        print("Training already done -____-")
        print(1/0)
    logging.basicConfig(filename='./outputs_paper/'+args.brain_region+'_bv_' + args.training_type  + '_' + args.llm_encoder + '_' + args.readout +'.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logging.info("start training with %d %d", start_epoch, iteration)

    opt = torch.optim.Adam
    lr = schedule[0]
    print('Training with learning rate', lr)
    optimizer = opt(model.parameters(), lr=lr)
    optimizer.zero_grad()
    model.train()
    iter_tracker = 0 
    model.cuda()
    for epoch in range(start_epoch,n_epochs):
        for img_ids, _, x_batch, img_batch, y_batch in tqdm(training_generator):
            if args.readout == 'spatial_linear' or args.readout == 'gaussian2d':
                outputs = model(x_batch.cuda())
            elif args.readout == 'semantic_transformer':
                outputs = model(x_batch.cuda(),img_batch.cuda())
            elif 'linear_ridge' in args.readout:
                outputs = model(x_batch.cuda())
            loss = criterion_mse(outputs, y_batch.cuda().float()) + criterion_corr(outputs.cuda(), y_batch.cuda().float())
            if 'linear_ridge' in args.readout:
                loss = loss + model.readout.l2_regularization().cuda()
            has_nan = torch.isnan(loss).any().item()
            if has_nan:
                print("loss : ", loss)
                print("criterion_mse(outputs, y_batch.cuda().float())  : ", criterion_mse(outputs, y_batch.cuda().float()) )
                print("criterion_corr(outputs.cuda(), y_batch.cuda().float()) : ", criterion_corr(outputs.cuda(), y_batch.cuda().float()))
                print("outputs :: ", outputs)
                print(epoch, iteration)
                print(1/0)
            loss.backward(retain_graph=True)
            iteration += 1
            if iteration % accumulate_gradient == accumulate_gradient - 1:
                optimizer.step()
                optimizer.zero_grad()
            if iteration%500==0:
                model.eval()
                true, preds = compute_predictions_only_dense_captions(args,validation_generator, model)
                val_corr = compute_scores(true, preds)
                logging.info('Epoch :: %d, Iteration :: %d, Validation Correlation :: %f', epoch, iteration, val_corr)
                is_best = val_corr >= best_corr
                model.train()
                if is_best:
                    best_corr = val_corr.copy()
                    iter_tracker = 0  
                    logging.info('saving model %d',len(readout.state_dict().keys()))
                    model_base = '%s_%d_%d' % (brain_region, epoch, iteration)
                    logging.info('model :: %s', model_base)
                    save_checkpoint({'epoch': epoch + 1,
                                        'state_dict': model.state_dict()},
                                        is_best = is_best,
                                        checkpoint = model_dir, model_str = model_base)
                else:
                    iter_tracker += 1
                    if iter_tracker == patience: 
                        print('Training complete')
                        break 
                torch.cuda.empty_cache()
            if iter_tracker == patience:
                break
    return

def train_single_dense_captions(args, training_generator, validation_generator, brain_region, model):

    iteration = args.iteration
    start_epoch = args.start_epoch
    best_corr = args.best_corr

    model_dir = './outputs/saved_'+args.brain_region+'/bv_' + args.training_type + '_' + args.llm_encoder + '_' + args.readout
    if not os.path.isdir(model_dir):
        os.mkdir(model_dir)
    logging.basicConfig(filename='./outputs/saved_'+args.brain_region+'/bv_' + args.training_type  + '_' + args.llm_encoder + '_' + args.readout +'.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logging.info("start training with %d %d", start_epoch, iteration)

    opt = torch.optim.Adam
    lr = schedule[0]
    print('Training with learning rate', lr)
    optimizer = opt(model.parameters(), lr=lr)
    optimizer.zero_grad()
    model.train()
    iter_tracker = 0 
    model.cuda()
    print("train_single_dense_captions :::")
    for epoch in range(start_epoch,n_epochs):
        for img_ids, x_single_batch, x_dense_batch, img_batch, y_batch in tqdm(training_generator):
            if args.readout == 'spatial_linear':
                print("NOt IMplemented lol")
                print(1/0)
            elif args.readout == 'semantic_transformer':
                outputs = model(x_dense_batch.cuda(),img_batch.cuda(), x_single_batch.cuda())
            loss = criterion_mse(outputs, y_batch.cuda().float()) + criterion_corr(outputs.cuda(), y_batch.cuda().float())
            loss.backward(retain_graph=True)
            iteration += 1
            if iteration % accumulate_gradient == accumulate_gradient - 1:
                optimizer.step()
                optimizer.zero_grad()
            if iteration%500==0:
                model.eval()
                true, preds = compute_predictions_single_dense_captions(args,validation_generator, model)
                val_corr = compute_scores(true, preds)
                logging.info('Epoch :: %d, Iteration :: %d, Validation Correlation :: %f', epoch, iteration, val_corr)
                is_best = val_corr >= best_corr
                model.train()
                if is_best:
                    best_corr = val_corr.copy()
                    iter_tracker = 0  
                    logging.info('saving model %d',len(readout.state_dict().keys()))
                    model_base = '%s_%d_%d' % (brain_region, epoch, iteration)
                    logging.info('model :: %s', model_base)
                    save_checkpoint({'epoch': epoch + 1,
                                        'state_dict': model.state_dict()},
                                        is_best = is_best,
                                        checkpoint = model_dir, model_str = model_base)
                else:
                    iter_tracker += 1
                    if iter_tracker == patience: 
                        print('Training complete')
                        break 
                torch.cuda.empty_cache()
            if iter_tracker == patience:
                break
    return

def train_only_dense_captions_images(args, training_generator, validation_generator, brain_region, model):

    iteration = args.iteration
    start_epoch = args.start_epoch
    best_corr = args.best_corr

    model_dir = './outputs/saved_'+args.brain_region+'/bv_' + args.training_type + '_' + args.llm_encoder + '_' + args.readout
    if not os.path.isdir(model_dir):
        os.mkdir(model_dir)
    logging.basicConfig(filename='./outputs/saved_'+args.brain_region+'/bv_' + args.training_type  + '_' + args.llm_encoder + '_' + args.readout +'.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logging.info("start training with %d %d", start_epoch, iteration)

    opt = torch.optim.Adam
    lr = schedule[0]
    print('Training with learning rate', lr)
    optimizer = opt(model.parameters(), lr=lr)
    optimizer.zero_grad()
    model.train()
    iter_tracker = 0 
    model.cuda()

    if args.training_type == 'only_dense_captions_pre_images':
        for param in model.core_img.parameters():
            param.requires_grad = False
        model.core_img.eval()
        
    for epoch in range(start_epoch,n_epochs):
        for img_ids, _, x_batch, img_batch, y_batch in tqdm(training_generator):
            outputs = model(x_batch.cuda(),img_batch.cuda())
            loss = criterion_mse(outputs, y_batch.cuda().float()) + criterion_corr(outputs.cuda(), y_batch.cuda().float())
            has_nan = torch.isnan(loss).any().item()
            if has_nan:
                print("loss : ", loss)
                print("criterion_mse(outputs, y_batch.cuda().float())  : ", criterion_mse(outputs, y_batch.cuda().float()) )
                print("criterion_corr(outputs.cuda(), y_batch.cuda().float()) : ", criterion_corr(outputs.cuda(), y_batch.cuda().float()))
                print("outputs :: ", outputs)
                print(epoch, iteration)
                print(1/0)
            loss.backward(retain_graph=True)
            iteration += 1
            if iteration % accumulate_gradient == accumulate_gradient - 1:
                optimizer.step()
                optimizer.zero_grad()
            if iteration%500==0:
                model.eval()
                true, preds = compute_predictions_only_dense_captions_images(args,validation_generator, model)
                val_corr = compute_scores(true, preds)
                logging.info('Epoch :: %d, Iteration :: %d, Validation Correlation :: %f', epoch, iteration, val_corr)
                is_best = val_corr >= best_corr
                model.train()
                if is_best:
                    best_corr = val_corr.copy()
                    iter_tracker = 0  
                    logging.info('saving model %d',len(readout.state_dict().keys()))
                    model_base = '%s_%d_%d' % (brain_region, epoch, iteration)
                    logging.info('model :: %s', model_base)
                    save_checkpoint({'epoch': epoch + 1,
                                        'state_dict': model.state_dict()},
                                        is_best = is_best,
                                        checkpoint = model_dir, model_str = model_base)
                else:
                    iter_tracker += 1
                    if iter_tracker == patience: 
                        print('Training complete')
                        
                        break 
                torch.cuda.empty_cache()
            if iter_tracker == patience:
                break
    return

def evaluate_single_caption_model(validation_generator, test_generator, readout):
    readout.eval()
    true, preds = compute_predictions_only_single_captions(validation_generator, readout)
    valid_corr = compute_scores_evaluation(true, preds)
    noise_ceiling = np.load('data/'+args.brain_region+'/noise_ceiling_1257_filtered.npy')
    valid_corr_norm = valid_corr/noise_ceiling


    true, preds = compute_predictions_only_single_captions(test_generator, readout)
    test_corr = compute_scores_evaluation(true, preds)
    noise_ceiling = np.load('data/'+args.brain_region+'/noise_ceiling_1257_filtered.npy')
    test_corr_norm = test_corr/noise_ceiling
    evaluation_dir = args.evaluation_dir
    np.save('metrics/bv_' + args.brain_region + '_' + args.training_type + '_' + args.llm_encoder + '_' + args.readout + '.npy', np.array(test_corr))
    test_corr, test_corr_norm = np.nanmean(test_corr), np.nanmean(test_corr_norm)
    valid_corr, valid_corr_norm = np.nanmean(valid_corr), np.nanmean(valid_corr_norm)
    with open(evaluation_dir+'/metrics_bv_' + args.brain_region + '_' + args.training_type + '_' + args.llm_encoder + '_' + args.readout + '.txt', 'w') as file:
        file.write('valid_corr :: '+str(valid_corr)+'\n'+'valid_corr_norm :: '+str(valid_corr_norm) + '\n')
        file.write('test_corr :: '+str(test_corr)+'\n'+'test_corr_norm :: '+str(test_corr_norm))
    return

def evaluate_only_dense_caption_model(args, validation_generator, test_generator, model):
    model.eval()

    true, preds = compute_predictions_only_dense_captions_images(args, validation_generator, model)
    valid_corr = compute_scores_evaluation(true, preds)
    noise_ceiling = np.load('data/'+args.brain_region+'/noise_ceiling_1257_filtered.npy')
    valid_corr_norm = valid_corr/noise_ceiling

    true, preds = compute_predictions_only_dense_captions_images(args,test_generator, model)
    test_corr = compute_scores_evaluation(true, preds)
    noise_ceiling = np.load('data/'+args.brain_region+'/noise_ceiling_1257_filtered.npy')
    test_corr_norm = test_corr/noise_ceiling
    evaluation_dir = args.evaluation_dir
    np.save('metrics/bv_' + args.brain_region + '_' + args.training_type + '_' + args.llm_encoder + '_' + args.readout + '.npy', np.array(test_corr))
    test_corr, test_corr_norm = np.nanmean(test_corr), np.nanmean(test_corr_norm)
    valid_corr, valid_corr_norm = np.nanmean(valid_corr), np.nanmean(valid_corr_norm)
    with open(evaluation_dir+'/metrics_bv_' + args.brain_region + '_' + args.training_type + '_' + args.llm_encoder + '_' + args.readout + '.txt', 'w') as file:
        file.write('valid_corr :: '+str(valid_corr)+'\n'+'valid_corr_norm :: '+str(valid_corr_norm) + '\n')
        file.write('test_corr :: '+str(test_corr)+'\n'+'test_corr_norm :: '+str(test_corr_norm))
    return

if __name__ == "__main__":
    args = define_args()
    print("Starting")

    brain_region = args.brain_region
    training_set = Dataset_language(mode = 'train', brain_region = brain_region, data_path='data/'+brain_region+'/', model_type='filtered', caption_type=args.llm_encoder, training_type=args.training_type)
    training_generator = torch.utils.data.DataLoader(training_set,  **params)
    print('Train data loaded')

    validation_set = Dataset_language(mode = 'val', brain_region = brain_region, data_path='data/'+brain_region+'/', model_type='filtered', caption_type=args.llm_encoder, training_type=args.training_type)
    validation_generator = torch.utils.data.DataLoader(validation_set,  **params_val)
    print('Validation data loaded')

    test_set = Dataset_language(mode = 'test', brain_region = brain_region, data_path='data/'+brain_region+'/', model_type='filtered', caption_type=args.llm_encoder, training_type=args.training_type)
    test_generator = torch.utils.data.DataLoader(test_set,  **params_val)
    print('test data loaded')

    if args.training_type == 'only_single_captions':
        print(args.training_type)
        if args.llm_encoder == 'mpnet':
            print("MPNET")
            readout = RidgeRegression(NUM_FEATURES_MPNET, training_set.n_neurons, alpha=args.alpha)
        elif args.llm_encoder == 'clip':
            print("CLIP")
            readout = RidgeRegression(NUM_FEATURES_CLIP, training_set.n_neurons, alpha=args.alpha)
        if args.alpha != 0.0:
            args.training_type = args.training_type + '_' + str(args.alpha)
        if args.evaluate:
            readout.train()
            model_dir = './outputs_paper/saved_'+args.brain_region+'_bv_' + args.training_type + '_' + args.llm_encoder
            print(model_dir)
            checkpoint = torch.load(model_dir + '/best.pth.tar')
            checkpoint = checkpoint['state_dict']
            readout.load_state_dict(checkpoint)
            evaluate_single_caption_model(validation_generator, test_generator, readout.cuda())
        else:
            train_only_single_captions(args, training_generator, validation_generator, brain_region, readout)
    elif args.training_type in ['only_dense_selected_captions', 'only_dense_captions']:
        print(args.training_type)
        if args.llm_encoder == 'clip':
            print("CLIP")
            core = RCNNTEXT(n_feats = n_feats)
            if args.readout == 'semantic_transformer':
                readout = SemanticSpatialTransformer(core(torch.randn(1, NUM_FEATURES_CLIP,8,8)).size()[1:], training_set.n_neurons,  bias = True)  
                model = Encoder_semantic(core, readout)
            elif args.readout == 'spatial_linear':
                readout = SpatialXFeatureLinear(core(torch.randn(1, NUM_FEATURES_CLIP,8,8)).size()[1:], training_set.n_neurons,  bias = True)  
                model = Encoder(core, readout)
            elif args.readout == 'gaussian2d':
                print("Gaussian 2d readout")
                readout = Gaussian2d(core(torch.randn(1, NUM_FEATURES_CLIP,8,8)).size()[1:], training_set.n_neurons,  bias = True)  
                model = Encoder(core, readout)
            elif args.readout == 'linear_ridge':
                print("Linear Ridge Regression Readout")
                readout = RidgeRegression(core(torch.randn(1, NUM_FEATURES_CLIP,8,8)).view(1, -1).size()[1], training_set.n_neurons, alpha=args.alpha)  
                model = Encoder_Ridge(core, readout)
                if args.alpha != 0.0:
                    args.readout = args.readout + '_' + str(args.alpha)
        elif args.llm_encoder == 'mpnet':
            print("mpnet")
            core = RCNNTEXT(n_feats = n_feats, n_features=768)
            if args.readout == 'semantic_transformer':
                readout = SemanticSpatialTransformer(core(torch.randn(1, NUM_FEATURES_MPNET,8,8)).size()[1:], training_set.n_neurons,  bias = True)  
                model = Encoder_semantic(core, readout)
            elif args.readout == 'spatial_linear':
                readout = SpatialXFeatureLinear(core(torch.randn(1, NUM_FEATURES_MPNET,8,8)).size()[1:], training_set.n_neurons,  bias = True)  
                model = Encoder(core, readout)
            elif args.readout == 'gaussian2d':
                print("Gaussian 2d readout")
                readout = Gaussian2d(core(torch.randn(1, NUM_FEATURES_MPNET,8,8)).size()[1:], training_set.n_neurons,  bias = True)  
                model = Encoder(core, readout)
            elif args.readout == 'linear_ridge':
                print("Linear Ridge Regression Readout")
                readout = RidgeRegression(core(torch.randn(1, NUM_FEATURES_MPNET,8,8)).view(1, -1).size()[1], training_set.n_neurons)  
                model = Encoder_Ridge(core, readout)
                if args.alpha != 0.0:
                    args.readout = args.readout + '_' + str(args.alpha)
        if args.evaluate:
            model.train()
            model_dir = './outputs_paper/'+args.brain_region+'_bv_' + args.training_type + '_' + args.llm_encoder + '_' + args.readout
            print(model_dir)
            checkpoint = torch.load(model_dir + '/best.pth.tar')
            checkpoint = checkpoint['state_dict']
            model.load_state_dict(checkpoint)
            evaluate_only_dense_caption_model(args, validation_generator, test_generator, model.cuda())
        else:
            train_only_dense_captions(args, training_generator, validation_generator, brain_region, model)
    elif args.training_type in ['single_dense_captions']:
        print(args.training_type)
        if args.llm_encoder == 'mpnet':
            print("mpnet")
            core = RCNNTEXT(n_feats = n_feats, n_features=768)
            core_enc_shape = core(torch.randn(1, NUM_FEATURES_MPNET,8,8)).size()[1:]
            readout = SemanticSpatialTransformer(core(torch.randn(1, NUM_FEATURES_MPNET,8,8)).size()[1:], training_set.n_neurons,  bias = True)    
            model = Encoder_Caption_semantic(core, readout, training_set.n_neurons, NUM_FEATURES_MPNET)
        if args.llm_encoder == 'clip':
            print("clip")
            core = RCNNTEXT(n_feats = n_feats)
            readout = SemanticSpatialTransformer(core(torch.randn(1, NUM_FEATURES_CLIP,8,8)).size()[1:], training_set.n_neurons,  bias = True)    
            model = Encoder_Caption_semantic(core, readout, training_set.n_neurons, NUM_FEATURES_CLIP)
        train_single_dense_captions(args, training_generator, validation_generator, brain_region, model)
    elif args.training_type in ['only_dense_captions_images','only_dense_captions_pre_images','only_dense_selected_captions_images']:
        core_img = C8SteerableCNN(n_feats = n_feats)
        if args.training_type == 'only_dense_captions_pre_images':
            print("Pre trained image encoder")
            _, core_img, _ = load_vanilla_IT_model(config[args.brain_region]['n_neurons'], config[args.brain_region]['it']['vanilla']['semantic_transformer'], 'semantic_transformer')
        if args.llm_encoder == 'clip':
            core_text = RCNNTEXT(n_feats = n_feats)
            readout = SemanticSpatialTransformerTextImage(core_img(torch.randn(1, 3, 224, 224)).size()[1:], core_text(torch.randn(1, NUM_FEATURES_CLIP,8,8)).size()[1:], training_set.n_neurons,  bias = True)  
        if args.llm_encoder == 'mpnet':
            core_text = RCNNTEXT(n_feats = n_feats, n_features=768)
            readout = SemanticSpatialTransformerTextImage(core_img(torch.randn(1, 3, 224, 224)).size()[1:], core_text(torch.randn(1, NUM_FEATURES_MPNET,8,8)).size()[1:], training_set.n_neurons,  bias = True) 
        model = Encoder_semantic_text_image(core_img, core_text, readout)
        train_only_dense_captions_images(args, training_generator, validation_generator, brain_region, model)