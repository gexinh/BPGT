from utils.utils import *

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


def convert_label(Y_ori, region_list):
    # region 2 label
    if len(Y_ori.size()) >1:
        for idx in range(len(Y_ori.size()[0])):
            Y_ori[idx,:] =region_list.index(Y_ori[idx,:])
    else:
        Y_ori[:] = torch.tensor(list(region_list).index(Y_ori[:]),dtype=torch.int32)
    return Y_ori

def multi_cls_auc(labels, prob, n_classes):
    aucs = []
    binary_labels = label_binarize(labels, classes=[i for i in range(n_classes)])
    for class_idx in range(n_classes):
        if class_idx in labels:
            fpr, tpr, _ = roc_curve(binary_labels[:, class_idx], prob[:, class_idx])
            aucs.append(calc_auc(fpr, tpr))
        else:
            aucs.append(float('nan'))

    auc = np.nanmean(np.array(aucs))
    if np.isnan(auc):
        auc = 0.000
    return auc, aucs

def metric_per_label(prob, labels, n_label, metric, label_list):
    """
        prob: prediction \in [N, n_label]
        labels: ground truth \in [N, n_label]
        n_label: number of labels
        metric: 'f1' or 'auc'
        label_list: []
    """
    metrics = []
    if metric == 'auc':
        for i in range(n_label):
            cur_label = labels[:,i]
            if sum(cur_label)!=0 :
                auc = roc_auc_score(cur_label, prob[:, i])
            else:
                auc = 0.000
            if np.isnan(auc):
                auc = 0.000
            metrics.append(round(auc,3))
            # if i != n_label:
            #     print('gene {}`s AUC: {}'.format(args.process_gene_list.iloc[i], auc))
            # else:
            #     print('non-mutation`s AUC: {}'.format(auc))
    elif metric == 'f1':
        eps = np.finfo(np.float32).eps
        binarizer = Binarizer(threshold=0.5)
        pred = binarizer.transform(prob)
        tp = labels * pred # true positive

        # (samples, labels)
        if len(labels.shape) == 2:
            no_tp = np.sum(tp, axis=0) + eps
            no_pred = np.sum(pred, axis=0) + eps
            no_pos = np.sum(labels, axis=0) + eps
        # 
        elif len(labels.shape) == 1:
            no_tp = np.sum(tp) + eps
            no_pred = np.sum(pred) + eps
            no_pos = np.sum(labels) + eps

        prec_class = no_tp / no_pred + eps
        rec_class = no_tp / no_pos + eps
        f1_class = 2 * prec_class * rec_class / (prec_class + rec_class)
        metrics = np.around(f1_class,3)

    print('Gene:', end='')
    print(*label_list, sep = " | ")
    print(metric+':', end='') 
    print(*metrics, sep = " | ") 


def calculate_metric(pred, labels):
    positve_num = np.sum(labels,axis=0)
    idx = [ i for i, val in enumerate(positve_num) if val==0] 
    cur_labels = np.delete(labels, idx, axis=1)
    cur_pred = np.delete(pred, idx, axis=1)

    result = calculate_metrics(cur_pred, cur_labels)
    from utils.eval_utils import calculate_mAP
    mAp = calculate_mAP(cur_labels, cur_pred)
    print("metric: "
        "micro f1: {:.3f} "
        "macro f1: {:.3f} "
        "samples f1: {:.3f} "
        "micro precision: {:.3f} "
        "micro recall: {:.3f} "
        "mAP: {:.3f}".format(
                                    result['micro/f1'],
                                    result['macro/f1'],
                                    result['samples/f1'],
                                    result['micro/precision'],
                                    result['micro/recall'],
                                    mAp                                    
                                    ))
    f1_micro = f1_score(cur_pred, cur_labels, average='micro')
    f1_macro = f1_score(cur_pred, cur_labels, average='macro')
    return f1_micro, f1_macro

class Accuracy_Logger(object):
    """Accuracy logger"""
    def __init__(self, n_classes):
        super(Accuracy_Logger, self).__init__()
        self.n_classes = n_classes
        self.initialize()

    def initialize(self):
        self.data = [{"count": 0, "correct": 0} for i in range(self.n_classes)]
    
    def log(self, Y_hat, Y):
        Y_hat = int(Y_hat)
        Y = int(Y)
        self.data[Y]["count"] += 1
        self.data[Y]["correct"] += (Y_hat == Y)
    
    def log_batch(self, Y_hat, Y):
        Y_hat = np.array(Y_hat).astype(int)
        Y = np.array(Y).astype(int)
        for label_class in np.unique(Y):
            cls_mask = Y == label_class
            self.data[label_class]["count"] += cls_mask.sum()
            self.data[label_class]["correct"] += (Y_hat[cls_mask] == Y[cls_mask]).sum()
    
def cal_region_auc(args, dataset):
    label_set, prob_set = dataset
    region_list = label_set.keys()
    for idx, c in enumerate(region_list):
        labels = label_set[c]
        prob = prob_set[c]

        if labels is not None:
            n_label = np.shape(labels)[-1]
            aucs = []
            for i in range(n_label):
                cur_label = labels[:,i]
                cur_prob = prob[:, i]
                if sum(cur_label)!=0 :
                    auc = roc_auc_score(cur_label, cur_prob)
                else:
                    auc = 0.000
                if np.isnan(auc):
                    auc = 0.000
                aucs.append(round(auc,3))

            print('-------------------- Region:[{}]----------------------'.format(c)) 
            if idx == 0:
                print('Gene:', end='')
                print(*args.process_gene_list, sep = " | ")
            print('AUC:', end='') 
            print(*aucs, sep = " | ") 


def region_select_concatenage(region_list, label, label_vector, label_set):
    if len(label_vector.size()) ==1:
        label_vector = label_vector.unsqueeze(0)
    region = region_list[int(label)]
    if label_set[region] is None:
        label_set[region] = label_vector.detach().cpu().numpy()
    else:
        label_set[region] = np.concatenate((label_set[region], label_vector.detach().cpu().numpy()),axis=0)
    return  label_set
    

def train_loop(args, epoch, model, loader, optimizer, 
                n_classes, bag_weight, writer = None, loss_fn = None, 
                num_region=1, region_list=[],category_loss_fn = None):
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.train()
    # acc_logger = Accuracy_Logger(n_classes=n_classes)
    # inst_logger = Accuracy_Logger(n_classes=n_classes)
    
    train_loss = 0.
    train_error = 0.
    train_inst_loss = 0.
    inst_count = 0
    train_lasso_loss = 0. 
    train_category_loss = 0.
    n_label = n_classes

    prob = np.zeros((len(loader), n_label))
    labels = np.zeros((len(loader), n_label))
    binary_logits = np.zeros((len(loader), n_label))
    region_labels = np.zeros((len(loader),1))
    region_prob = np.zeros((len(loader),num_region))

    # region specific auc
        #format
    region_label_set = {i:None for i in region_list}
    region_prob_set = {i:None for i in region_list}

    print('\n')
    print('Epoch: {}'.format(epoch))
    for batch_idx, (data, label) in enumerate(loader):
        optimizer.zero_grad()

        Y_region = label [:, 0]
        Y_region = convert_label(Y_region, region_list) 
        Y_region = Y_region.to(device).to(torch.int64)
        
        label = label[:, 1:]
        data, label = data.to(device), label.to(device)
        # data, label = data.half().to(device), label.to(device)
        # data, label = data.half().cuda(), label.cuda()
        data = torch.squeeze(data)
        
        # print(batch_idx)
        # print(data.size())
        # print(label.size())
        
        #TODO:
        # logits, Y_prob, Y_hat, _, addtional_dict = model(data, label=Y_region, instance_eval=True)
        logits, Y_hat, addtional_dict = model(data, label=Y_region, instance_eval=True)
        lasso_loss, Y_region_logits, Y_region_prob = addtional_dict['lasso_loss'], addtional_dict['Y_region_logits'], addtional_dict['Y_region_prob']
        #TODO: multi-label use Y_hat 
        binary_logits[batch_idx] = logits.detach().cpu().numpy()

        prob[batch_idx] = Y_hat.detach().cpu().numpy()
        labels[batch_idx] = label.detach().cpu().numpy()

        # 
        region_label_set = region_select_concatenage(region_list, Y_region, label, region_label_set)
        region_prob_set = region_select_concatenage(region_list, Y_region, Y_hat, region_prob_set)
        
        if isinstance(Y_region_prob,list):
            region_labels[batch_idx] = np.zeros(1)
            region_prob[batch_idx] = np.zeros(1)
        else:
            region_labels[batch_idx] = Y_region.detach().cpu().numpy()
            region_prob[batch_idx] = Y_region_prob.detach().cpu().numpy()

        # acc_logger.log(Y_hat, label)
        if category_loss_fn is not None and num_region != 1 and args.downstream_model!='mlp' and args.region_cls and args.agg_method=='cor_mil' :
            category_loss = category_loss_fn(Y_region_logits, Y_region)
        else:
            category_loss = torch.zeros(1).to(device)
        loss = loss_fn(logits, label)
        loss_value = loss.item()

        # TODO: instance loss

        if args.region_cls_model == 'wsl' and num_region != 1 and args.downstream_model!='mlp' and args.agg_method=='cor_mil':
            instance_loss = addtional_dict['instance_loss']
            inst_count+=1
            instance_loss_value = instance_loss.item()
            train_inst_loss += instance_loss_value
            
            inst_preds = addtional_dict['inst_preds']
            inst_labels = addtional_dict['inst_labels']
        else:
            instance_loss = torch.tensor(0).to(device)
            instance_loss_value = 0.

        
        """ calculate loss
        """
        # TODO: L1 norm
        if args.l1_norm == True:
            l1_lambda = 5e-5
        else:
            l1_lambda = 0

        l1_norm = sum(p.abs().sum()
                  for p in model.parameters())

        # total_loss = (1-bag_weight) * loss + bag_weight * lasso_loss + l1_lambda*l1_norm

        total_loss = (1-bag_weight) * loss + 0. * lasso_loss + l1_lambda*l1_norm + bag_weight*(category_loss + instance_loss)

        # error = calculate_error(Y_hat, label)
        error = calculate_error_for_ml(Y_hat, label)

        train_error += error
        train_loss += loss_value
        train_lasso_loss += lasso_loss
        train_category_loss += category_loss.item()

        if (batch_idx) % 500 == 0:
            # print('batch {}, loss: {:.4f}, instance_loss: {:.4f}, weighted_loss: {:.4f}, '.format(batch_idx, loss_value, instance_loss_value, total_loss.item()) + 
            #     'label: {}, bag_size: {}'.format(label.item(), data.size(0)))
            
            print('batch {}, loss: {:.4f}, lasso_loss: {:.4f}, category_loss: {:.4f}, instance_loss: {:.4f}, bag_size: {}'
                    .format(batch_idx, loss_value, lasso_loss, category_loss.item(), instance_loss_value, data.size(0)))

        # error = calculate_error(Y_hat, label)
        # train_error += error
        
        # backward pass
        total_loss.backward()
        # step
        optimizer.step()
        

    # calculate loss and error for epoch
    train_loss /= len(loader)
    train_error /= len(loader)
    train_lasso_loss /= len(loader)
    train_category_loss /= len(loader)

    if writer:
        writer.add_scalar('train/loss', train_loss, epoch)
        writer.add_scalar('train/error', train_error, epoch)
        writer.add_scalar('train/clustering_loss', train_inst_loss, epoch)

    #TODO: 
    # binarizer = Binarizer(threshold=0.)
    # pred = binarizer.transform(binary_logits)
    binarizer = Binarizer(threshold=0.5)
    pred = binarizer.transform(prob)
    

    print('||----------------- training performance ---------------------------||')
    f1_micro, f1_macro = calculate_metric(pred, labels)
    print('Epoch: {}, train_loss: {:.4f}, train_category_loss: {:.4f}, train_error: {:.4f}, f1_micro:  {:.4f}, f1_macro: {:.4f}'.format(epoch, train_loss, train_category_loss, train_error, f1_micro, f1_macro))
    

    metric_per_label(prob, labels, n_label, metric = 'f1', label_list = args.process_gene_list)

    print('-------------------------------- region classification -------------------------------')
    if args.region_cls == True:
        region_auc, auc_list = multi_cls_auc(labels=region_labels, prob=region_prob, n_classes=num_region)
        print('average region auc is {}'.format(region_auc))
        print('Region:', end='') 
        print(*[round(auc_list.index(i),3) for i in auc_list], sep = " | ")         
        print('AUC:', end='') 
        print(*[round(i,3) for i in auc_list], sep = " | ") 

    print('-------------------------------- train Region mAUC -------------------------------')
    region_dataset = [region_label_set, region_prob_set]
    # cal_region_auc(args, region_dataset)
    


def validate(args, cur, epoch, model, loader, n_classes, 
            early_stopping = None, writer = None, loss_fn = None, 
            results_dir = None, 
            num_region=1, region_list=[],category_loss_fn = None):
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    # acc_logger = Accuracy_Logger(n_classes=n_classes)
    # inst_logger = Accuracy_Logger(n_classes=n_classes)
    val_loss = 0.
    val_error = 0.
    val_lasso_loss = 0.

    val_category_loss = 0.
    val_inst_loss = 0.
    # val_inst_acc = 0.
    inst_count=0
    
    n_label = n_classes
    binary_logits = np.zeros((len(loader), n_label))
    prob = np.zeros((len(loader), n_label))
    labels = np.zeros((len(loader), n_label))
    region_labels = np.zeros((len(loader),1))
    region_prob = np.zeros((len(loader),num_region))

    # region specific 
    region_label_set = {i:None for i in region_list}
    region_prob_set = {i:None for i in region_list}
    

    with torch.no_grad():
        for batch_idx, (data, label) in enumerate(loader):
            
            Y_region = label [:, 0]
            Y_region = convert_label(Y_region, region_list) 
            Y_region = Y_region.to(device).to(torch.int64)
            label = label[:, 1:]
            data, label = data.to(device), label.to(device)     
            # data, label = data.half().cuda(), label.cuda()      
            data = torch.squeeze(data)

            logits, Y_hat, addtional_dict = model(data, label=Y_region, instance_eval=True)
            lasso_loss, Y_region_logits, Y_region_prob = addtional_dict['lasso_loss'], addtional_dict['Y_region_logits'], addtional_dict['Y_region_prob']

            # acc_logger.log(Y_hat, label)
            if category_loss_fn is not None and num_region != 1 and args.downstream_model!='mlp' and args.region_cls and args.agg_method=='cor_mil' :
                category_loss = category_loss_fn(Y_region_logits, Y_region)
            else:
                category_loss = torch.zeros(1).to(device)
            loss = loss_fn(logits, label)
            error = calculate_error_for_ml(Y_hat, label)
            val_category_loss += category_loss.item()

            if args.region_cls_model == 'wsl' and num_region != 1 and args.downstream_model!='mlp' and args.agg_method=='cor_mil':
                instance_loss = addtional_dict['instance_loss']
                inst_count+=1
                instance_loss_value = instance_loss.item()
                val_inst_loss += instance_loss_value
                
                inst_preds = addtional_dict['inst_preds']
                inst_labels = addtional_dict['inst_labels']
            else:
                instance_loss = torch.tensor(0).to(device)
                instance_loss_value = 0.

            binary_logits[batch_idx] = logits.detach().cpu().numpy()
            prob[batch_idx] = Y_hat.detach().cpu().numpy()
            labels[batch_idx] = label.detach().cpu().numpy()
            if isinstance(Y_region_prob,list):
                region_labels[batch_idx] = np.zeros(1)
                region_prob[batch_idx] = np.zeros(1)
            else:
                region_labels[batch_idx] = Y_region.detach().cpu().numpy()
                region_prob[batch_idx] = Y_region_prob.detach().cpu().numpy()

            # region-sepcific auc
            region_label_set = region_select_concatenage(region_list, Y_region, label, region_label_set)
            region_prob_set = region_select_concatenage(region_list, Y_region, Y_hat, region_prob_set)

            if (batch_idx) % 1000 == 0:
                print('max prediction probability is: {:.3f}'.format(np.max(prob[batch_idx,:])))

            val_loss += loss.item()
            val_error += error
            val_lasso_loss += lasso_loss

    val_error /= len(loader)
    val_loss /= len(loader)
    val_lasso_loss /= len(loader)
    val_category_loss /= len(loader)

    binarizer = Binarizer(threshold=0.)
    pred = binarizer.transform(binary_logits)

    print('||------------------------------- Validation set -----------------------------||')
    f1_micro, f1_macro = calculate_metric(pred, labels)
    print('Val Set, val_loss: {:.4f}, val_lasso_loss: {:.4f}, val_category_loss: {:.4f}, val_error: {:.4f}, micro-f1: {:.4f}, macro-f1: {:.4f}'.format(val_loss, val_lasso_loss, val_category_loss, val_error, f1_micro, f1_macro))
    
    # AUC per label
    metric_per_label(prob, labels, n_label, metric = 'f1', label_list = args.process_gene_list)

    print('-------------------------------- region classification ------------------------')
    if args.region_cls == True:
        region_auc, auc_list = multi_cls_auc(labels=region_labels, prob=region_prob, n_classes=num_region)
        print('average region auc is {}'.format(region_auc))
        print('Region:', end='') 
        print(*[round(auc_list.index(i),3) for i in auc_list], sep = " | ")         
        print('AUC:', end='') 
        print(*[round(i,3) for i in auc_list], sep = " | ") 
    
    print('-------------------------------- Val Region mAUC -------------------------------')
    region_dataset = [region_label_set, region_prob_set]


    if writer:
        writer.add_scalar('val/loss', val_loss, epoch)
        # writer.add_scalar('val/auc', auc, epoch)
        writer.add_scalar('val/error', val_error, epoch)
        # writer.add_scalar('val/inst_loss', val_inst_loss, epoch)

    if early_stopping:
        assert results_dir
        early_stopping(epoch, val_loss, model, ckpt_name = os.path.join(results_dir, "s_{}_checkpoint.pt".format(cur)))
        
        if early_stopping.early_stop:
            print("Early stopping")
            return True

    return False


def summary(args, model, loader, n_classes,region_list=[], if_print=False):
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    acc_logger = Accuracy_Logger(n_classes=n_classes)
    model.eval()
    test_loss = 0.
    test_error = 0.

    n_label = n_classes
    binary_logits = np.zeros((len(loader), n_label))
    prob = np.zeros((len(loader), n_label))
    labels = np.zeros((len(loader), n_label))
    
    # region specific 
    region_label_set = {i:None for i in region_list}
    region_prob_set = {i:None for i in region_list}

    slide_ids = loader.dataset.slide_data['slide_id']
    patient_results = {}

    for batch_idx, (data, label) in enumerate(loader):
        Y_region = label [:, 0]
        Y_region = convert_label(Y_region, region_list) 
        Y_region = Y_region.to(device).to(torch.int64)
        label = label[:, 1:]
        data, label = data.to(device), label.to(device)     
        # data, label = data.half().cuda(), label.cuda()
        data = torch.squeeze(data)

        slide_id = slide_ids.iloc[batch_idx]
        with torch.no_grad():
            logits, Y_hat, addtional_dict = model(data, label=Y_region, instance_eval=False)
            lasso_loss, Y_region_logits, Y_region_prob = addtional_dict['lasso_loss'], addtional_dict['Y_region_logits'], addtional_dict['Y_region_prob']
        # acc_logger.log(Y_hat, label)
        error = calculate_error_for_ml(Y_hat, label)
        test_error += error

        # convert to cpu as numpy.array
        binary_logits[batch_idx] = logits.detach().cpu().numpy()
        prob[batch_idx] = Y_hat.detach().cpu().numpy()
        labels[batch_idx] = label.detach().cpu().numpy()

        # region-sepcific auc
        region_label_set = region_select_concatenage(region_list, Y_region, label, region_label_set)
        region_prob_set = region_select_concatenage(region_list, Y_region, Y_hat, region_prob_set)

        # patient_results.update({slide_id: {'slide_id': np.array(slide_id), 'label': label.item()}})
        patient_results.update({slide_id: {'slide_id': np.array(slide_id), }})
        error = calculate_error(Y_hat, label)

    test_error /= len(loader)


    
    binarizer = Binarizer(threshold=0.)
    pred = binarizer.transform(binary_logits)
    # binarizer = Binarizer(threshold=0.5)
    # pred = binarizer.transform(prob)

    if if_print:
        print('||-------------------------------- TestSet -------------------------------||')
        metric_per_label(prob, labels, n_label, metric = 'f1', label_list = args.process_gene_list)

        print('-------------------------------- test Region mAUC -------------------------')
        region_dataset = [region_label_set, region_prob_set]
        # cal_region_auc(args, region_dataset)

    f1_micro, f1_macro = calculate_metric(pred, labels)
    # from utils.eval_utils import calculate_mAP
    # mAp = calculate_mAP(labels, pred)

    return patient_results, test_error, f1_micro, f1_macro, acc_logger
