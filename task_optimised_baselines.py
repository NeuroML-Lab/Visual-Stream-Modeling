import warnings
warnings.filterwarnings("ignore", category=UserWarning)

import os
import numpy as np
import torch
from utils.train_utils import *
from utils.model_utils import *
from utils.dataloaders import Dataset_visual
from models.models_brain import C8SteerableCNN
from neuralpredictors.layers.readouts import SpatialXFeatureLinear, Gaussian2d
from utils.readouts import SemanticSpatialTransformer
from torch.autograd import Variable
from itertools import repeat
from utils.utils import save_checkpoint
from tqdm import tqdm
import argparse
import logging
import matplotlib.pyplot as plt
from torchvision.models import *
from torchvision.models.vision_transformer import vit_b_32

schedule = [1e-4]
criterion_mse = masked_MSEloss
criterion_huber = masked_Huberloss
criterion_corr = masked_Correlation_loss
patience = 20
iter_tracker = 0 
accumulate_gradient= 4
n_epochs = 100
n_feats = 48
alexnet_sub_layers = {1:3, 2:6, 3:8, 4:10, 5:12}

def compute_predictions_semantic(loader, model, reshape=True, stack=True, return_lag=False):
    y, y_hat = [], []
    for id, x_val, y_val in (loader):
        neurons = y_val.size(-1)

        y_mod = model(x_val.cuda().float(), x_val.cuda()).data.cpu().numpy()
        y.append(y_val.numpy())
        y_hat.append(y_mod)
    if stack:
        y, y_hat = np.vstack(y), np.vstack(y_hat) 
    return y, y_hat ##true,pred

def compute_predictions_spatial(loader, model, reshape=True, stack=True, return_lag=False):
    y, y_hat = [], []
    for id, x_val, y_val in (loader):
        neurons = y_val.size(-1)

        y_mod = model(x_val.cuda().float()).data.cpu().numpy()
        y.append(y_val.numpy())
        y_hat.append(y_mod)
    if stack:
        y, y_hat = np.vstack(y), np.vstack(y_hat) 
    return y, y_hat ##true,pred

def define_args():
    parser = argparse.ArgumentParser(description='Encoding model')
    parser.add_argument('--brain_region', default='ventral_visual_data', type=str)
    parser.add_argument('--model_type', default='filtered', type=str)
    parser.add_argument('--readout', default='semantic_transformer', type=str)
    parser.add_argument('--saved_model_it', default=None, type=str)
    parser.add_argument('--iteration', default=0, type=int)
    parser.add_argument('--start_epoch', default=0, type=int)
    parser.add_argument('--best_corr', default=0.0, type=float)
    parser.add_argument('--evaluate', action='store_true')
    parser.add_argument('--evaluation_dir', default='evaluations_paper',type=str)
    parser.add_argument('--task_optimised_model', default='alexnet',type=str)
    parser.add_argument('--alpha', default=0.0, type=float)
    parser.add_argument('--use_sub_layers', action='store_true')
    parser.add_argument('--sub_layers', default=1, type=int)
    args = parser.parse_args()
    return args

def train_model(args, start_epoch,n_epochs, iteration, best_corr, training_generator, validation_generator, model):
    model_dir = 'outputs_paper/' + args.task_optimised_model + '_' + args.brain_region + '_' + args.readout
    
    if not os.path.isdir(model_dir):
        os.makedirs(model_dir, exist_ok=True)
    # else:
    #     print("Training already done -____-")
    #     print(1/0)

    logging.basicConfig(filename='outputs_paper/' + args.task_optimised_model + '_' + args.brain_region + '_' + args.readout + '.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logging.info("start training with %d %d", start_epoch, iteration)
    
    opt = torch.optim.Adam
    lr = schedule[0]
    print('Training with learning rate', lr)
    optimizer = opt(model.readout.parameters(), lr=lr)
    optimizer.zero_grad()

    iter_tracker = 0

    for epoch in range(start_epoch,n_epochs):
        for img_ids, x_batch, y_batch in tqdm(training_generator):
            img_ids = img_ids.numpy()
            if args.readout == 'spatial_linear' or args.readout == 'gaussian2d':
                outputs = model(x_batch.cuda()).cpu()
            elif args.readout == 'semantic_transformer':
                outputs = model(x_batch.cuda(), x_batch.cuda()).cpu()
            elif 'linear_ridge' in args.readout:
                outputs = model(x_batch.cuda()).cpu()
            loss = criterion_mse(outputs.cuda(), y_batch.cuda().float()) + criterion_corr(outputs.cuda(), y_batch.cuda().float())
            if 'linear_ridge' in args.readout:
                loss = loss + model.readout.l2_regularization().cuda()
            loss.backward(retain_graph=True)
            iteration += 1
            if iteration % accumulate_gradient == accumulate_gradient - 1:
                optimizer.step()
                optimizer.zero_grad()
            if iteration%500==0:
                model.eval()
                if args.readout == 'spatial_linear' or args.readout == 'gaussian2d' or 'linear_ridge' in args.readout:
                    true, preds = compute_predictions_spatial(validation_generator, model)
                if args.readout == 'semantic_transformer':
                    true, preds = compute_predictions_semantic(validation_generator, model)
                val_corr = compute_scores(true, preds)
                logging.info('Epoch :: %d, Iteration :: %d, Validation Correlation :: %f', epoch, iteration, val_corr)
                is_best = val_corr >= best_corr
                model.train()
                if is_best:
                    best_corr = val_corr.copy()
                    iter_tracker = 0  
                    logging.info('saving model %d',len(model.state_dict().keys()))
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

def evaluate_trained_model(args, model, validation_generator, test_generator):
    model.eval()

    if args.readout == 'spatial_linear' or args.readout == 'gaussian2d' or 'linear_ridge' in args.readout:
        true, preds = compute_predictions_spatial(validation_generator, model)
    if args.readout == 'semantic_transformer':
        true, preds = compute_predictions_semantic(validation_generator, model)
    valid_corr = compute_scores_evaluation(true, preds)


    if args.readout == 'spatial_linear' or args.readout == 'gaussian2d' or 'linear_ridge' in args.readout:
        true, preds = compute_predictions_spatial(test_generator, model)
    if args.readout == 'semantic_transformer':
        true, preds = compute_predictions_semantic(test_generator, model)
    test_corr = compute_scores_evaluation(true, preds)
    noise_ceiling = np.load('data/'+args.brain_region+'/noise_ceiling_1257_filtered.npy')

    evaluation_dir = args.evaluation_dir
    # if not os.path.isdir(evaluation_dir):
    #     os.mkdir(evaluation_dir)

    test_corr_norm = test_corr/noise_ceiling
    valid_corr_norm = valid_corr/noise_ceiling
    # plt.scatter(noise_ceiling,np.array(test_corr))
    # plt.xlabel('NC Label')
    # plt.ylabel('Test Corr Label')
    # plt.legend()
    # plt.savefig(evaluation_dir+'/metrics.png') 

    # np.save(evaluation_dir+'/test_corr.npy', np.array(test_corr))
    np.save('metrics/' + args.task_optimised_model + '_' + args.brain_region + '_' + args.readout + '.npy', np.array(test_corr))

    test_corr, test_corr_norm = np.nanmean(test_corr), np.nanmean(test_corr_norm)
    valid_corr, valid_corr_norm = np.nanmean(valid_corr), np.nanmean(valid_corr_norm)
    with open(evaluation_dir+'/metrics_' + args.task_optimised_model + '_' + args.brain_region + '_' + args.readout + '.txt', 'w') as file:
        file.write('valid_corr :: '+str(valid_corr)+'\n'+'valid_corr_norm :: '+str(valid_corr_norm) + '\n')
        file.write('test_corr :: '+str(test_corr)+'\n'+'test_corr_norm :: '+str(test_corr_norm))
    return
    

if __name__ == "__main__":
    args = define_args()

    brain_region = args.brain_region
    training_set = Dataset_visual(mode = 'train', brain_region = brain_region, data_path='data/'+brain_region+'/', model_type=args.model_type)
    training_generator = torch.utils.data.DataLoader(training_set,  **params)
    print('Train data loaded')

    validation_set = Dataset_visual(mode = 'val', brain_region = brain_region, data_path='data/'+brain_region+'/', model_type=args.model_type)
    validation_generator = torch.utils.data.DataLoader(validation_set,  **params_val)
    print('Validation data loaded')

    test_set = Dataset_visual(mode = 'test', brain_region = brain_region, data_path='data/'+brain_region+'/', model_type=args.model_type)
    test_generator = torch.utils.data.DataLoader(test_set,  **params_val)
    print('test data loaded')

    n_neurons = training_set.n_neurons
    if args.task_optimised_model == 'resnet50':
        resnet_model = resnet50(pretrained=True)
        if not args.use_sub_layers:
            core = nn.Sequential(*list(resnet_model.children())[:-2])
        else:
            core = nn.Sequential(*list(resnet_model.children())[: (4+args.sub_layers)])
            args.task_optimised_model = 'resnet50_' + str(args.sub_layers)
    if args.task_optimised_model == 'resnet152':
        resnet_model = resnet152(pretrained=True)
        core = nn.Sequential(*list(resnet_model.children())[:-2])
    elif args.task_optimised_model == 'alexnet':
        alexnet_model = alexnet(pretrained=True)
        if not args.use_sub_layers:
            core = nn.Sequential(*list(alexnet_model.children())[:-1])
        else:
            core = nn.Sequential(*list(alexnet_model.children())[:-1])
            core[0] = core[0][:alexnet_sub_layers[args.sub_layers]]
            args.task_optimised_model = 'alexnet_' + str(args.sub_layers)
    elif args.task_optimised_model == 'vgg_11_bn':
        vgg11_bn_model = vgg11_bn(pretrained=True)
        core = nn.Sequential(*list(vgg11_bn_model.children())[:-1])
    for param in core.parameters():
        param.requires_grad = False

    if args.readout == 'spatial_linear':
        print("spatial linear readout")
        readout = SpatialXFeatureLinear(core(torch.randn(1, 3, 224, 224)).size()[1:], n_neurons,  bias = True)  
        model = Encoder(core, readout)
    elif args.readout == 'semantic_transformer':
        print("semantic transformer readout")
        readout = SemanticSpatialTransformer(core(torch.randn(1, 3, 224, 224)).size()[1:], n_neurons,  bias = True)  
        model = Encoder_semantic(core, readout)
    elif args.readout == 'gaussian2d':
        print("Gaussian 2d readout")
        readout = Gaussian2d(core(torch.randn(1, 3, 224, 224)).size()[1:], n_neurons,  bias = True)  
        model = Encoder(core, readout)
    elif args.readout == 'linear_ridge':
        print("Linear Ridge Regression Readout")
        readout = RidgeRegression(core(torch.randn(1, 3, 112, 112)).view(1, -1).size()[1], training_set.n_neurons, alpha=args.alpha)  
        model = Encoder_Ridge(core, readout)
        if args.alpha != 0.0:
            args.readout = args.readout + str(args.alpha)
    # model = model.cuda()
    print("model IT defined")
    print("n_neurons :: ", n_neurons)

    iteration = args.iteration
    start_epoch = args.start_epoch
    best_corr = args.best_corr

    if args.saved_model_it is not None:
        model.train()
        checkpoint = torch.load(parent_saved_dir+'/' + args.saved_model_it)
        checkpoint = checkpoint['state_dict']
        model.load_state_dict(checkpoint)
    
    model.cuda()

    print("evaluation :: ", args.evaluate)
    if args.evaluate:
        model.train()
        model_dir = 'outputs_paper/' + args.task_optimised_model + '_' + args.brain_region + '_' + args.readout
        checkpoint = torch.load(model_dir + '/best.pth.tar')
        checkpoint = checkpoint['state_dict']
        model.load_state_dict(checkpoint)
        evaluate_trained_model(args, model, validation_generator, test_generator)
    else:
        train_model(args, start_epoch,n_epochs, iteration, best_corr, training_generator, validation_generator, model)

    