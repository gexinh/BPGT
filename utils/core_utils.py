
import imp
from utils.model_selection import model_selection
from utils.utils import *

from model.base_model import base_net, AttnClassifier, CLAM_SB, Att_multi_label_module, query2label, transformer_decoder

import time
import os
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from sklearn.preprocessing import label_binarize
from sklearn.preprocessing import Binarizer
from sklearn.metrics import roc_auc_score, roc_curve, f1_score, classification_report, accuracy_score
from sklearn.metrics import auc as calc_auc
from sklearn.metrics import f1_micro as calc_auc


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=20, stop_epoch=50, verbose=False):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 20
            stop_epoch (int): Earliest epoch possible for stopping
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
        """
        self.patience = patience
        self.stop_epoch = stop_epoch
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf

    def __call__(self, epoch, val_loss, model, ckpt_name = 'checkpoint.pt'):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, ckpt_name)
        elif score < self.best_score:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience and epoch > self.stop_epoch:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, ckpt_name)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, ckpt_name):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), ckpt_name)
        self.val_loss_min = val_loss


def train(datasets, cur, args):
    """ Initilize writer
    """
    # os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(str(x) for x in args.gpu)
    print('The number of GPU is {}'.format(torch.cuda.device_count()))
    # torch.distributed.init_process_group(backend='nccl', init_method='tcp://localhost:23456', rank=0, world_size=1)
    # print('The number of GPU is {}'.format(torch.cuda.device_count()))

    print('\nTraining Fold {}!'.format(cur))
    writer_dir = os.path.join(args.results_dir, str(cur))
    if not os.path.isdir(writer_dir):
        os.mkdir(writer_dir)

    if args.log_data:
        from tensorboardX import SummaryWriter
        writer = SummaryWriter(writer_dir, flush_secs=15)

    else:
        writer = None

    """ loda data
    """
    train_split, val_split, test_split = datasets   # what is the construction?
   
    if args.model_type != 'att_ml':
        neg_set = []
        pos_set = []
        for d in datasets:
            label = d.slide_data.iloc[:,args.label_ind+ args.start_ind]
            neg, pos = len(label)-np.count_nonzero(label), np.count_nonzero(label)  
            neg_set.append(neg)
            pos_set.append(pos)

        # save_splits(datasets, ['train', 'val', 'test'], os.path.join(args.results_dir, 'splits_{}.csv'.format(0)))
        print('Done!')
        print("Training set is: [{}/{}]".format(pos_set[0],len(train_split)))
        print("Validation set is: [{}/{}]".format(pos_set[1],len(val_split)))
        print("Test set is: [{}/{}]".format(pos_set[2],len(test_split)))

        n_class = 2

    else:
        # attention-based multi-label

        n_label = len(train_split.slide_data.columns.values.tolist()[args.start_ind:])

        region = train_split.slide_data.iloc[:,args.start_ind-1].tolist()
        unique, counts = np.unique(region, return_counts=True)  
        num_region = len(unique)


        neg_set = []
        pos_set = []

        for d in datasets:
            n_neg = 0
            n_pos = 0
            for i in range(n_label):   
                label = d.slide_data.iloc[:,i+ args.start_ind]
                neg, pos = len(label) - np.count_nonzero(label), np.count_nonzero(label)  
                n_neg += neg
                n_pos += pos
            neg /= n_label
            pos /= n_label

            neg_set.append(neg)
            pos_set.append(pos)

        
    """ Initialize dataloader
    """
    print('\nInit Loaders...', end=' ')
    if args.resample == True:
        from torchsampler import ImbalancedDatasetSampler
        custom_sampler = ImbalancedDatasetSampler(train_split, label_ind=args.label_ind+1)
        train_loader = DataLoader(
            train_split,
            sampler=custom_sampler,
            batch_size=1,
        )
    
    train_loader = DataLoader(train_split, batch_size=1)
    val_loader = DataLoader(val_split, batch_size=1)
    test_loader = DataLoader(test_split, batch_size=1)
    print('Done!')
    # TODO: check the meaning of the arg 'testing'
    # loader = DataLoader(split_dataset, batch_size=1, sampler = WeightedRandomSampler(weights, len(weights)), collate_fn = collate_MIL, **kwargs)	
    # train_loader = get_split_loader(train_split, training=True, testing = args.testing, weighted = args.weighted_sample)
    
    """initialize model

        Args:
            multi-label: sparse net for multi-label classification
            att_ml: attention module 

            binary_cls: dot self-attention module for binary classification 
            clam: weakly supervised for binary classification
    """
    if args.model_type =='multi_label':
        model_dict = {"mapp":args.mapp, "k" : [50, 200, 400], "attn": True, "dropout": args.drop_out, }
        model = base_net(**model_dict)
        # model = GeneTree(**model_dict, loss_fn)
    
    elif args.model_type == 'att_ml':
        args.adj = args.adj.to('cuda')

        # for region segmentation
        from topk.svm import SmoothTop1SVM
        instance_loss_fn = SmoothTop1SVM(n_classes = 2)
        if device.type == 'cuda':
            instance_loss_fn = instance_loss_fn.cuda()

        """ experiment  
        """
        from utils.model_selection import module_selection
        # utilize to select the visual extractor 
        args = model_selection(args)

        model_dict = {"adj":args.un_adj, "dropout": args.drop_out, 'num_classes': n_label, 
                "model_size": args.model_size, 'k_sample': args.k_sample, 
                'dropout_rate':args.dropout_rate,
                "num_layers_decoder":args.decoder_layer,
                "gcn": args.gcn,
                "agg_method": args.agg_method,
                "num_layers_encoder": args.encoder_layer,
                "region_cls":args.region_cls,
                "num_region":num_region,
                "instance_loss_fn": instance_loss_fn,
                "cls_token": args.cls_token,
                'num_graph_layer':args.num_graph_layer,
                'word_embedd': args.word_embedd,
                'graph_module': args.graph_module
                }
        model_dict = module_selection(args, model_dict)
        from model.model_exp import model_markets
        model = model_markets(**model_dict)

        model = model.cuda()
        model.relocate()
    
    #TODO: multi-GPU
    if args.if_ddp:
        model = nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
    # model.relocate()

    # from torchinfo import summary
    # batch_size = 16
    # summary(model, input_size=(batch_size, 100, 1024))
    
    # model.relocate()
    print('Done!')
    print_network(model)  # print model 
    
    """ initialize optimizer
    """
    print('\nInit optimizer ...', end=' ')
    optimizer = get_optim(model, args)
    print('Done!')

    """ scheduler
    """
    from cosine_annealing_warmup import CosineAnnealingWarmupRestarts
    scheduler = CosineAnnealingWarmupRestarts(optimizer,
                                          first_cycle_steps=150,
                                          cycle_mult=1.0,
                                          max_lr=1e-3,
                                          min_lr=args.lr,
                                          warmup_steps=30,
                                          gamma=1.0)

    """ Pipeline
    """
    if args.model_type =='att_ml':
        # TODO: add multi-class classification for region
        for i in range(len(unique)):
            print('The number of region {} is {}'.format(unique[i], counts[i]))  
        
        # class weight
        if args.clsw == True:

            total = np.sum(counts)
            weights = []
            for i in range(len(unique)):
                weights.append((1/counts[i]) * (total/2.0))

            weight_CE = torch.FloatTensor(weights).cuda()
            # print('Positve weight is {}'.format(p_w))
            # print('Negtive weight is {}'.format(n_w))
            # print('--------------------------------------------------\n')
        else:
            weight_CE = torch.FloatTensor(np.ones(shape=num_region)).cuda()

        if args.loss == 'ce':
            category_loss_fn = nn.CrossEntropyLoss(weight=weight_CE, reduction = 'none')
            args.loss_weight = 1

        elif args.loss == 'cb':
            no_of_classes = num_region
            beta = 0.9999
            gamma = args.gamma
            loss_type = "focal"
            samples_per_cls = list(counts)
            from model.loss import class_balanced_loss
            category_loss_fn = class_balanced_loss(samples_per_cls, no_of_classes,loss_type, beta, gamma)
            args.loss_weight = 1


        import utils.pipeline.att_ml as att_ml
        args.n_classes = n_label
        
        if args.early_stopping:
            early_stopping = EarlyStopping(patience = 30, stop_epoch=60, verbose = True)
        else:
            early_stopping = None

        """ initialize loss
        """
        from utils.utils import comparative_multi_label_loss

        loss_fn = comparative_multi_label_loss(margin=4.0)


        if args.mode == 'train':
            """ Training
            """
            for epoch in range(args.max_epochs):
                import time
                start_time = time.time()
                att_ml.train_loop(args, epoch, model, train_loader, optimizer, args.n_classes, args.weight, writer, loss_fn, num_region, unique, category_loss_fn)
                print('time of each epoch is {:.4f}'.format(time.time() - start_time))
                stop = att_ml.validate(args, cur, epoch, model, val_loader, args.n_classes,early_stopping=early_stopping,
                        writer=writer, loss_fn=loss_fn, results_dir=args.results_dir, num_region= num_region, region_list=unique, category_loss_fn=category_loss_fn)
                if stop: 
                    break

                #TODO: use the warm up + cosine annel 
                # scheduler.step()
                if epoch%20==0:
                    results_dict, test_error, test_f1_micro, test_f1_macro, acc_logger = att_ml.summary(args, model, test_loader, args.n_classes,region_list=unique, if_print=True)
                    print('Test error: {:.4f}, f1-micro/macro: [{:.4f}/{:.4f}]'.format(test_error, test_f1_micro, test_f1_macro))


            if args.early_stopping:
                model.load_state_dict(torch.load(os.path.join(args.results_dir, "s_{}_checkpoint.pt".format(cur))))
            else:
                torch.save(model.state_dict(), os.path.join(args.results_dir, "s_{}_checkpoint.pt".format(cur)))
            
            torch.save(model.state_dict(), os.path.join(args.results_dir,"s_{}_checkpoint.pt".format(cur)))

        # load model from checkpoint 
        else:
            model.load_state_dict(torch.load(os.path.join(args.results_dir, "s_{}_checkpoint.pt".format(cur))))            

        
        """ evaluation
        """
        # print Gene AUC/F1
        print('\n ------------------ Test Set ---------------------')
        results_dict, test_error, test_f1_micro, test_f1_macro, acc_logger = att_ml.summary(args, model, test_loader, args.n_classes,region_list=unique, if_print=True)
        
        print('\n ------------------ Final Test Summary ---------------------')
        _, train_error, train_f1_micro, train_f1_macro, _= att_ml.summary(args, model, train_loader, args.n_classes,region_list=unique)
        print('Train error: {:.4f}, f1-micro/macro: [{:.4f}/{:.4f}]'.format(train_error, train_f1_micro, train_f1_macro))

        _, val_error, val_f1_micro, val_f1_macro, _= att_ml.summary(args, model, val_loader, args.n_classes,region_list=unique)
        print('Val error: {:.4f}, f1-micro/macro: [{:.4f}/{:.4f}]'.format(val_error, val_f1_micro, val_f1_macro))

        results_dict, test_error, test_f1_micro, test_f1_macro, acc_logger = att_ml.summary(args, model, test_loader, args.n_classes,region_list=unique)
        print('Test error: {:.4f}, f1-micro/macro: [{:.4f}/{:.4f}]'.format(test_error, test_f1_micro, test_f1_macro))

        print('---------------------- Test Slide-level -----------------------')
        # for i in range(args.n_classes):
        #     acc, correct, count = acc_logger.get_summary(i)
        #     if not acc:
        #         acc = 0.0
        #     print('class {}: acc {}, correct {}/{}'.format(i, acc, correct, count))

        #     if writer:
        #         writer.add_scalar('final/test_class_{}_acc'.format(i), acc, 0)

        if writer:
            writer.add_scalar('final/val_error', val_error, 0)
            writer.add_scalar('final/val_micro_f1', val_f1_micro, 0)
            writer.add_scalar('final/test_error', test_error, 0)
            writer.add_scalar('final/test_micro_f1', test_f1_micro, 0)
            writer.close()
        return test_f1_micro, val_f1_micro, 1-test_error, 1-val_error  
    
    elif args.model_type =='multi_label':
        """ initialize loss
        """
        loss_fn = nn.BCELoss()
        weight = args.weight

        """ Training
        """
        for epoch in range(args.max_epochs):
            """ Train
            """
            train_loop(epoch, model, train_loader, optimizer, args, weight, loss_fn, writer)
            """ Validate
            """
            validate(args, cur, epoch, model, val_loader, writer, loss_fn)
        
        """ Evaluate metric
        """
        # TODO: finish the summary: return results_dict, f1_micro, and error
        val_error, val_auc = eval_summary(model, val_loader, args)
        print('Val error: {:.4f}, f1-micro: {:.4f}'.format(val_error, val_auc))

        test_error, test_auc = eval_summary(model, test_loader, args)
        print('Test error: {:.4f}, f1-micro: {:.4f}'.format(test_error, test_auc))

        """ Save Result
        """ 
        if writer:
            writer.add_scalar('final/val_error', val_error, 0)
            writer.add_scalar('final/val_micro_f1', val_auc, 0)
            writer.add_scalar('final/test_error', test_error, 0)
            writer.add_scalar('final/test_micro_f1', test_auc, 0)
            writer.close()

        return test_auc, val_auc, 1-test_error, 1-val_error
        # return results_dict, test_auc, val_auc, 1-test_error, 1-val_error

    elif args.model_type =='binary_cls':
        """ initialize loss
        """
        loss_fn = nn.CrossEntropyLoss()

        for epoch in range(args.max_epochs):
            """ Train
            """
            train_loop_binary(epoch, model, train_loader, optimizer, args, loss_fn, writer)

            """ Validate
            """
            validate_binary(args, cur, epoch, model, val_loader, writer, loss_fn)

        """ Evaluate metric
        """
        val_acc, val_auc = eval_summary_binary(model, val_loader, args)
        print('Test error: {:.4f}, test auc: {:.4f}'.format(test_error, test_auc))

        test_acc, test_auc = eval_summary_binary(model, test_loader, args)
        print('Test error: {:.4f}, test auc: {:.4f}'.format(test_error, test_auc))

        """ Save Result
        """ 
        if writer:
            writer.add_scalar('final/test_acc', test_acc, 0)
            writer.add_scalar('final/test_auc', test_auc, 0)
            writer.close()

        return test_auc, val_auc, test_acc, val_acc
    