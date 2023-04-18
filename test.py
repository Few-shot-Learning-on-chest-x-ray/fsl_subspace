# -*- coding: utf-8 -*-
import argparse

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from torch.autograd import Variable

from tqdm import tqdm

from models.protonet_embedding import ProtoNetEmbedding
from models.R2D2_embedding import R2D2Embedding
from models.ResNet12_embedding import resnet12

from models.classification_heads import ClassificationHead

from utils import pprint, set_gpu, Timer, count_accuracy, log
from sklearn.metrics import confusion_matrix, f1_score, roc_curve, auc
import scikitplot as skplt
import matplotlib.pyplot as plt


import numpy as np
import os
import random

import pickle 

from dataloader.chest import label_dict


import pandas as pd

def multiclass_roc(y_test, y_score,n_classes = 3):
    

    # structures
    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    # calculate dummies once
    y_test_dummies = pd.get_dummies(y_test, drop_first=False).values
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test_dummies[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    return fpr,tpr,roc_auc

# os.environ['CUDA_VISIBLE_DEVICES'] = "0"

def seed_everything(seed: int):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

def euclidean_dist(x, y):

    # x: N x D
    # y: M x D
    n = x.size(0)
    m = y.size(0)
    d = x.size(1)

    assert d == y.size(1)

    x = x.unsqueeze(1).expand(n, m, d)
    y = y.unsqueeze(0).expand(n, m, d)

    return torch.pow(x - y, 2).sum(2)

def flip(x, dim):
    xsize = x.size()
    dim = x.dim() + dim if dim < 0 else dim
    x = x.view(-1, *xsize[dim:])
    x = x.view(x.size(0), x.size(1), -1)[:, getattr(torch.arange(x.size(1)-1,
                      -1, -1), ('cpu','cuda')[x.is_cuda])().long(), :]
    return x.view(xsize)


def get_model(options):
    # Choose the embedding network
    if options.network == 'ProtoNet':
        network = ProtoNetEmbedding().cuda()
    elif options.network == 'R2D2':
        network = R2D2Embedding().cuda()
    elif options.network == 'ResNet':
        if options.dataset == 'miniImageNet' or options.dataset == 'tieredImageNet':
            network = resnet12(avg_pool=False, drop_rate=0.1, dropblock_size=5,num_layer=options.num_layer).cuda()
            network = torch.nn.DataParallel(network)
        else:
            network = resnet12(avg_pool=False, drop_rate=0.1, dropblock_size=2,num_layer=options.num_layer).cuda()
    else:
        print ("Cannot recognize the network type")
        assert(False)

    # Choose the classification head
    if opt.head == 'ProtoNet':
        cls_head = ClassificationHead(base_learner='ProtoNet').cuda()
    elif options.head == 'SubspaceTrans':
        cls_head = ClassificationHead(base_learner='SubspaceTrans').cuda()
    elif options.head == 'Subspace':
        cls_head = ClassificationHead(base_learner='Subspace').cuda()
    elif options.head == 'SubspaceFast':
        cls_head = ClassificationHead(base_learner='SubspaceFast').cuda()
    elif opt.head == 'Ridge':
        cls_head = ClassificationHead(base_learner='Ridge').cuda()
    elif opt.head == 'R2D2':
        cls_head = ClassificationHead(base_learner='R2D2').cuda()
    elif opt.head == 'SVM':
        cls_head = ClassificationHead(base_learner='SVM-CS').cuda()
    else:
        print ("Cannot recognize the classification head type")
        assert(False)

    return (network, cls_head)

def get_dataset(options):
    # Choose the embedding network
    if options.dataset == 'miniImageNet':
        from dataloader.mini_imagenet import MiniImageNet, FewShotDataloader
        dataset_test = MiniImageNet(phase='test')
        data_loader = FewShotDataloader
    elif options.dataset == 'tieredImageNet':
        from dataloader.tiered_imagenet import tieredImageNet, FewShotDataloader
        dataset_test = tieredImageNet(phase='test')
        data_loader = FewShotDataloader
    elif options.dataset == 'CIFAR_FS':
        from dataloader.CIFAR_FS import CIFAR_FS, FewShotDataloader
        dataset_test = CIFAR_FS(phase='test')
        data_loader = FewShotDataloader
    elif options.dataset == 'FC100':
        from dataloader.FC100 import FC100, FewShotDataloader
        dataset_test = FC100(phase='test')
        data_loader = FewShotDataloader
    elif options.dataset == 'Chest':
        from dataloader.chest import Chest, FewShotDataloader
        dataset_test = Chest(phase='test')
        data_loader = FewShotDataloader
    else:
        print ("Cannot recognize the dataset type")
        assert(False)

    return (dataset_test, data_loader)

#
if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    #Changes
    parser.add_argument('--gpu', default='3')
    #Changes    
    parser.add_argument('--load', 
                        default='experiments/group2_subspace30_CE_train/best_model.pth', ## your best model
                        help='path of the checkpoint file')
    #Changes
    parser.add_argument('--num_layer', type=int, default=30,
                            help='num of layer')
    
    parser.add_argument('--episode', type=int, default=1000,
                            help='number of episodes to test')
    parser.add_argument('--way', type=int, default=3,
                            help='number of classes in one test episode')
    parser.add_argument('--shot', type=int, default=5,
                            help='number of support examples per training class')
    parser.add_argument('--query', type=int, default=5,
                            help='number of query examples per training class')
    parser.add_argument('--network', type=str, default='ResNet',
                            help='choose which embedding network to use. ProtoNet, R2D2, ResNet')
    parser.add_argument('--head', type=str, default='Subspace',
                            help='choose which embedding network to use. ProtoNet, Ridge, R2D2, SVM')
    parser.add_argument('--dataset', type=str, default='Chest',
                            help='choose which classification head to use. miniImageNet, tieredImageNet, CIFAR_FS, FC100')


    opt = parser.parse_args()

    seed_everything(42)

    (dataset_test, data_loader) = get_dataset(opt)

    set_gpu(opt.gpu)

    # Define the models
    (embedding_net, cls_head) = get_model(opt)

    # Load saved model checkpoints
    saved_models = torch.load(opt.load)
    embedding_net.load_state_dict(saved_models['embedding'])
    embedding_net.eval()
    cls_head.load_state_dict(saved_models['head'])
    cls_head.eval()


    aug=False

    label_dict_inv =  {v:k for k,v in label_dict.items()}

    test_accuracies = []
    per_class_accuracies = []
    y_pred_list = []
    y_list = []
    dloader_test = data_loader(
        dataset=dataset_test,
        nKnovel=opt.way,
        nKbase=0,
        nExemplars=opt.shot,  # num training examples per novel category
        nTestNovel=opt.query * opt.way,  # num test examples for all the novel categories
        nTestBase=0,  # num test examples for all the base categories
        batch_size=1,
        num_workers=1,
        epoch_size=opt.episode,  # num of batches per epoch
    )

    #print("epp:  ", epp)

    with torch.no_grad():
        for i, batch in enumerate(tqdm(dloader_test()), 1):
            data_support, labels_support, data_query, labels_query, _, _ = [x.cuda() for x in batch]

            n_support = opt.way * opt.shot
            n_query = opt.way * opt.query

            if opt.shot == 1 and aug:
                flipped_data_support = flip(data_support, 3)
                data_support = torch.cat((data_support, flipped_data_support), dim=0)
                labels_support = torch.cat((labels_support, labels_support), dim=0)

            list_emb_support = embedding_net(data_support.reshape([-1] + list(data_support.shape[-3:])))
            list_emb_query = embedding_net(data_query.reshape([-1] + list(data_query.shape[-3:])))

            logits = torch.zeros(n_query, opt.way).cuda()

            for emb_support, emb_query in zip(list_emb_support, list_emb_query):


                emb_support = emb_support.view(1, opt.way, opt.shot, -1).mean(2)

                emb_query = emb_query.reshape(1, n_query, -1)

                dists = euclidean_dist(emb_query[0], emb_support[0])


                logits += F.softmax(-dists, dim=1).view(1 * opt.way * opt.query, -1)



            logits /= opt.num_layer

            logits = logits.reshape(-1, opt.way)
            labels_query = labels_query.reshape(-1)


            acc,pca = count_accuracy(logits, labels_query)
            test_accuracies.append(acc.item())
            per_class_accuracies.append(pca)

            y_pred_list.append(logits.detach().cpu().numpy())
            y_list.append(labels_query.detach().cpu().numpy())

            avg = np.mean(np.array(test_accuracies))
            std = np.std(np.array(test_accuracies))
            ci95 = 1.96 * std / np.sqrt(i + 1)

            if i % 10 == 0:
                
                # print(logits.detach().cpu().numpy())
                # print(torch.argmax(logits, dim=1).view(-1))
                # print(labels_query.detach().cpu().numpy())

                pca = np.array(per_class_accuracies).mean(0)
                pcs = np.array(per_class_accuracies).std(0)

                print('Episode [{}/{}]:\t\t\tAccuracy: {:.2f} ± {:.2f} ({:.2f}) % ({:.2f} %)'\
                    .format(i, opt.episode, avg, ci95,std, acc))
                print(f'{label_dict_inv[9]}: {pca[0]:.2f} ± {pcs[0]:.2f} % | {label_dict_inv[10]}: {pca[1]:.2f} ± {pcs[1]:.2f} % | {label_dict_inv[11]}: {pca[2]:.2f} ± {pcs[2]:.2f}%')

        

    pca = np.array(per_class_accuracies).mean(0)
    pcs = np.array(per_class_accuracies).std(0)

    print("Mean")
    print(pca)
    print('Standard Deviation')
    print(pcs)


    y_pred_proba = np.array(
        y_pred_list).reshape(-1, 3)

    y_pred = np.argmax(y_pred_proba, axis=1)

    y_true = np.array(y_list).reshape(-1)

    f1 = f1_score(y_true, y_pred, average=None)

    print('F1 Score')
    print(f1)

    fpr,tpr, auc = multiclass_roc(y_true,y_pred_proba)
    save_tuple = (fpr,tpr,auc)

    print(auc)

    # Plots

    #Changes
    # with open('plot/group5_subspace25.pickle', 'wb') as f:
    #     pickle.dump(save_tuple, f)

    #Changes
    class_dict = {'Fibrosis': 0, 'Hernia': 1, 'Pneumonia': 2}
    # class_dict = {'Mass': 0, 'Nodule': 1, 'Pleural_Thickening': 2}
    # class_dict = {'Cardiomegaly': 0, 'Edema': 1, 'Emphysema': 2}
    # class_dict = {'Consolidation': 0, 'Effusion': 1, 'Pneumothorax': 2}
    # class_dict = {'Atelectasis': 0, 'Infiltration': 1, 'No Finding': 2}

    class_dict_inv = {v: k for k, v in class_dict.items()}

    y_true = np.array([class_dict_inv[i]
                      for i in np.array(y_list).reshape(-1)])

    # print(np.array(y_pred_list).reshape(-1, 3).shape)
    # print(np.array(y_list).reshape(-1).shape)
    # print(y_list)
    # print(np.array(y_pred_list).reshape(-1, 3))


    # skplt.metrics.plot_roc(y_true, y_pred_proba,plot_micro=False, plot_macro=False)

    #Changes
    # plt.savefig('plot/group5_subspace25.png', dpi=1000)
    # plt.show()




# python test_ortho_bcs.py --gpu 2 --load experiments/chest_exp1/best_model.pth --way 3 --dataset Chest
# python test_ortho_bcs.py --gpu 2 --load experiments/chest_exp1/best_model.pth --way 3 --dataset Chest