# -*- coding: utf-8 -*-
import timm
import os
import sys
import argparse
import random
import numpy as np
from tqdm import tqdm
import torch
import torch.nn.functional as F
from torch import linalg as LA
from models.classification_heads import ClassificationHead
from models.R2D2_embedding import R2D2Embedding
from models.protonet_embedding import ProtoNetEmbedding
from models.ResNet12_embedding import resnet12
import torch.nn as nn
from utils import set_gpu, Timer, count_accuracy, check_dir, log
import warnings
import wandb
from itertools import combinations

from torchsummary import summary
warnings.filterwarnings("ignore")


def one_hot(indices, depth):
    """
    Returns a one-hot tensor.
    This is a PyTorch equivalent of Tensorflow's tf.one_hot.

    Parameters:
      indices:  a (n_batch, m) Tensor or (m) Tensor.
      depth: a scalar. Represents the depth of the one hot dimension.
    Returns: a (n_batch, m, depth) Tensor or (m, depth) Tensor.
    """

    encoded_indicies = torch.zeros(indices.size() + torch.Size([depth])).cuda()
    index = indices.view(indices.size()+torch.Size([1]))
    encoded_indicies = encoded_indicies.scatter_(1, index, 1)

    return encoded_indicies

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

def cosine_dist(x, y):

    # x: N x D
    # y: M x D
    n = x.size(0)
    m = y.size(0)
    d = x.size(1)

    assert d == y.size(1)

    x = x.unsqueeze(1).expand(n, m, d)
    y = y.unsqueeze(0).expand(n, m, d)



    cos = nn.CosineSimilarity(dim=2, eps=1e-6)
    out = 1 - cos(x,y)


    return out


def get_model(options):
    # Choose the embedding network
    if options.network == 'ProtoNet':
        network = ProtoNetEmbedding().cuda()
    elif options.network == 'R2D2':
        network = R2D2Embedding().cuda()
    elif options.network == 'ResNet':
        if options.dataset == 'miniImageNet' or options.dataset == 'tieredImageNet':
            network = resnet12(avg_pool=False, drop_rate=0.1,
                               dropblock_size=5,num_layer=options.num_layer).cuda()
            network = torch.nn.DataParallel(network)  # , device_ids=[1, 2])
        else:
            network = resnet12(avg_pool=False, drop_rate=0.1,
                               dropblock_size=2,num_layer=options.num_layer).cuda()
    else:
        print("Cannot recognize the network type")
        assert(False)

    # Choose the classification head
    if options.head == 'Subspace':
        cls_head = ClassificationHead(base_learner='Subspace').cuda()
    elif options.head == 'ProtoNet':
        cls_head = ClassificationHead(base_learner='ProtoNet').cuda()
    elif options.head == 'Ridge':
        cls_head = ClassificationHead(base_learner='Ridge').cuda()
    elif options.head == 'R2D2':
        cls_head = ClassificationHead(base_learner='R2D2').cuda()
    elif options.head == 'SVM':
        cls_head = ClassificationHead(base_learner='SVM-CS').cuda()
    else:
        print("Cannot recognize the dataset type")
        assert(False)

    return (network, cls_head)

def get_dataset(options):
    # Choose the embedding network
    if options.dataset == 'miniImageNet':
        from dataloader.mini_imagenet import MiniImageNet, FewShotDataloader
        # change it to train only, this is including the validation set
        dataset_train = MiniImageNet(phase='trainval')
        dataset_val = MiniImageNet(phase='test')
        data_loader = FewShotDataloader
    elif options.dataset == 'tieredImageNet':
        from dataloader.tiered_imagenet import tieredImageNet, FewShotDataloader
        dataset_train = tieredImageNet(phase='train')
        dataset_val = tieredImageNet(phase='test')
        data_loader = FewShotDataloader
    elif options.dataset == 'CIFAR_FS':
        from dataloader.CIFAR_FS import CIFAR_FS, FewShotDataloader
        dataset_train = CIFAR_FS(phase='train')
        dataset_val = CIFAR_FS(phase='test')
        data_loader = FewShotDataloader
    elif options.dataset == 'Chest':
        from dataloader.chest import Chest, FewShotDataloader
        dataset_train = Chest(phase='train')
        dataset_val = Chest(phase='val')
        data_loader = FewShotDataloader
    else:
        print("Cannot recognize the dataset type")
        assert(False)

    return (dataset_train, dataset_val, data_loader)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num-epoch', type=int, default=80,
                        help='number of training epochs')
    parser.add_argument('--save-epoch', type=int, default=5,
                        help='frequency of model saving')
    parser.add_argument('--train-shot', type=int, default=5,
                        help='number of support examples per training class')
    parser.add_argument('--val-shot', type=int, default=5,
                        help='number of support examples per validation class')
    parser.add_argument('--train-query', type=int, default=5,
                        help='number of query examples per training class')
    parser.add_argument('--val-episode', type=int, default=600,
                        help='number of episodes per validation')
    parser.add_argument('--val-query', type=int, default=5,
                        help='number of query examples per validation class')
    parser.add_argument('--train-way', type=int, default=3,
                        help='number of classes in one training episode')
    parser.add_argument('--test-way', type=int, default=3,
                        help='number of classes in one test (or validation) episode')
    parser.add_argument('--save-path', default='experiments')

    parser.add_argument('--wandbexperiment', default="group5_subspace30",type=str)
    parser.add_argument('--gpu', default='0')  # using 4 gpus
    parser.add_argument('--num_layer', type=int, default=30,
                        help='number of linear layer')

    # parser.add_argument('--gpu', default='0,1,2,3')  # using 4 gpus
    parser.add_argument('--network', type=str, default='ResNet',
                        help='choose which embedding network to use. ResNet')
    parser.add_argument('--head', type=str, default='Subspace',
                        help='choose which classification head to use. Subspace, ProtoNet, R2D2, SVM')
    parser.add_argument('--dataset', type=str, default='Chest',
                        help='choose which classification head to use. miniImageNet, tieredImageNet, CIFAR_FS, FC100')
    parser.add_argument('--episodes-per-batch', type=int, default=1,
                        help='number of episodes per batch')
    parser.add_argument('--eps', type=float, default=0.0,
                        help='epsilon of label smoothing')
    parser.add_argument('--wandb', action="store_true")
    parser.add_argument("--wandbkey", type=str,
                        default='db1158429a436f94565ac9eadecc6afe9e5a0b8f',
                        help='Wandb project key')


# python train_my.py --gpu 2 --dataset Chest --num_layer 5


    opt = parser.parse_args()
    seed_everything(42)
    print(opt)
    opt.save_path = os.path.join(opt.save_path,opt.wandbexperiment)


    if opt.wandb:
        os.system('wandb login {}'.format(opt.wandbkey))
        wandb.init(name=opt.wandbexperiment,
                   project='chest-few-shot-final')
        wandb.config.update(opt)

    (dataset_train, dataset_val, data_loader) = get_dataset(opt)

    # Dataloader of Gidaris & Komodakis (CVPR 2018)
    dloader_train = data_loader(
        dataset=dataset_train,
        nKnovel=opt.train_way,
        nKbase=0,
        nExemplars=opt.train_shot,  # num training examples per novel category
        # num test examples for all the novel categories
        nTestNovel=opt.train_way * opt.train_query,
        nTestBase=0,  # num test examples for all the base categories
        batch_size=opt.episodes_per_batch,
        num_workers=15,
        epoch_size=opt.episodes_per_batch * 1000,  # num of batches per epoch
    )

    dloader_val = data_loader(
        dataset=dataset_val,
        nKnovel=opt.test_way,
        nKbase=0,
        nExemplars=opt.val_shot,  # num training examples per novel category
        # num test examples for all the novel categories
        nTestNovel=opt.val_query * opt.test_way,
        nTestBase=0,  # num test examples for all the base categories
        batch_size=1,
        num_workers=15,
        epoch_size=1 * opt.val_episode,  # num of batches per epoch
    )

    set_gpu(opt.gpu)
    check_dir('./experiments/')
    check_dir(opt.save_path)

    log_file_path = os.path.join(opt.save_path, "train_log.txt")
    log(log_file_path, str(vars(opt)))

    (embedding_net, cls_head) = get_model(opt)

    optimizer = torch.optim.SGD(embedding_net.parameters(),lr=3e-3)


    def lambda_epoch(e): return 1.0 if e < 12 else (
        0.025 if e < 30 else 0.0032 if e < 45 else (0.0014 if e < 57 else (0.00052)))

    ## tieredimagenet###
    # lambda_epoch = lambda e: 1.0 if e < 20 else (
    #     0.012 if e < 45 else 0.0052 if e < 59 else (0.00054 if e < 68 else (0.00012)))

    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda=lambda_epoch, last_epoch=-1)

    max_val_acc = 0.0

    timer = Timer()
    x_entropy = torch.nn.CrossEntropyLoss()


    index = list(combinations([i for i in range(opt.num_layer)], 2))

    for epoch in range(1, opt.num_epoch + 1):


        for param_group in optimizer.param_groups:
            epoch_learning_rate = param_group['lr']

        log(log_file_path, 'Train Epoch: {}\tLearning Rate: {:.4f}'.format(
            epoch, epoch_learning_rate))

        _, _ = [x.train() for x in (embedding_net, cls_head)]

        train_accuracies = []
        train_losses = []

        train_n_support = opt.train_way * opt.train_shot
        train_n_query = opt.train_way * opt.train_query




        for i, batch in enumerate(tqdm(dloader_train(epoch)), 1):

            data_support, labels_support, data_query, labels_query, _, _ = [
                x.cuda() for x in batch]

            list_emb_query = embedding_net(data_query.view(
                [-1] + list(data_query.shape[-3:])))  # [100, 2560]
            list_emb_support  = embedding_net(data_support.view(
                [-1] + list(data_support.shape[-3:])))  # [100, 3, 32, 32] -> [100, 2560]


            loss_weights = 0.
            for ind in index:

                loss_weights += torch.abs(F.cosine_similarity(getattr(embedding_net,f'linear{ind[0]}_1').weight.view(-1),getattr(embedding_net,f'linear{ind[1]}_1').weight.view(-1),dim=0))


            log_p_y = torch.zeros(
                opt.episodes_per_batch * opt.train_way * opt.train_query, opt.train_way).cuda()

            for emb_support,emb_query in zip(list_emb_support, list_emb_query):
                # emb_support = emb_support.view(
                #     opt.episodes_per_batch, train_n_support, -1)  # [4, 25, 2560]
                if opt.train_shot == 1:
                    emb_support = emb_support.view(
                        opt.episodes_per_batch, opt.train_way, -1)  # [4,5,5,2560] --> [4, 5, 20]
                else:
                    emb_support = emb_support.view(
                        opt.episodes_per_batch, opt.train_way, opt.train_shot, -1).mean(2)  # [4,5,5,2560] --> [4, 5, 20]

                emb_query = emb_query.view(
                    opt.episodes_per_batch, train_n_query, -1)  # [4, 25, 2560]


                dists = torch.stack(
                    [euclidean_dist(emb_query[i], emb_support[i]) for i in range(opt.episodes_per_batch)])  # [4,25,5]



                log_p_y += F.softmax(-dists,
                                     dim=2).view(opt.episodes_per_batch* opt.train_way* opt.train_query, -1)  # [100,5]


            log_p_y /= opt.num_layer


            smoothed_one_hot = one_hot(
                labels_query.view(-1), opt.train_way)  # [100,5]

            loss = x_entropy(
                log_p_y.view(-1, opt.train_way), labels_query.view(-1))


            acc, _ = count_accuracy(
                log_p_y.view(-1, opt.train_way), labels_query.view(-1))

            train_accuracies.append(acc.item())
            train_losses.append(loss.item())

            if (i % 100 == 0):
                train_acc_avg = np.mean(np.array(train_accuracies))
                log(log_file_path, 'Train Epoch: {}\tBatch: [{}/{}]\tLoss: {:.4f}\tAccuracy: {:.2f} % ({:.2f} %)'.format(
                    epoch, i, len(dloader_train), loss.item(), train_acc_avg, acc))
                if opt.wandb:

                    wandb.log({'Epoch': epoch,
                           'lr': optimizer.param_groups[0]['lr'],"Loss":loss.item(),"Avg Accuracy":train_acc_avg,'Accuracy':acc,
                           'cosine loss':loss_weights})


            optimizer.zero_grad()

            loss += loss_weights
            loss.backward()

            optimizer.step()

        # Evaluate on the validation split
        _, _ = [x.eval() for x in (embedding_net, cls_head)]

        val_accuracies = []
        val_losses = []


        with torch.no_grad():

            for i, batch in enumerate(tqdm(dloader_val(epoch)), 1):
                data_support, labels_support, data_query, labels_query, _, _ = [
                    x.cuda() for x in batch]

                test_n_support = opt.test_way * opt.val_shot
                test_n_query = opt.test_way * opt.val_query


                list_emb_support = embedding_net(data_support.view(
                    [-1] + list(data_support.shape[-3:])))
                list_emb_query = embedding_net(data_query.view(
                    [-1] + list(data_query.shape[-3:])))


                logit_query = torch.zeros(test_n_query, opt.test_way).cuda()

                for emb_support, emb_query in zip(list_emb_support, list_emb_query):

                    # print(emb_support.size())
                    emb_support = emb_support.view(1, test_n_support, -1)
                    # print(emb_support.size())

                    emb_support = emb_support.view(
                        1, opt.train_way, opt.train_shot, -1).mean(2)  # [4, 5, 20]

                    emb_query = emb_query.view(1, test_n_query, -1)

                    # print(emb_support.size(),emb_query.size())

                    dists = torch.stack(
                        [euclidean_dist(emb_query[i], emb_support[i]) for i in range(emb_query.size(0))])

                    logit_query += F.softmax(-dists, dim=2).view(1 *
                                                                opt.test_way * opt.val_query, -1)  # []

                logit_query /= opt.num_layer


                loss = x_entropy(
                    logit_query.view(-1, opt.test_way), labels_query.view(-1))
                acc, _ = count_accuracy(
                    logit_query.view(-1, opt.test_way), labels_query.view(-1))

                val_accuracies.append(acc.item())
                val_losses.append(loss.item())

        val_acc_avg = np.mean(np.array(val_accuracies))
        val_acc_ci95 = 1.96 * \
            np.std(np.array(val_accuracies)) / np.sqrt(opt.val_episode)

        val_loss_avg = np.mean(np.array(val_losses))

        if val_acc_avg > max_val_acc:
            max_val_acc = val_acc_avg
            torch.save({'embedding': embedding_net.state_dict(), 'head': cls_head.state_dict()},
                       os.path.join(opt.save_path, 'best_model.pth'))



            log(log_file_path, 'Validation Epoch: {}\t\t\tLoss: {:.4f}\tAccuracy: {:.2f} ± {:.2f} % (Best)'
                .format(epoch, val_loss_avg, val_acc_avg, val_acc_ci95))
        else:
            log(log_file_path, 'Validation Epoch: {}\t\t\tLoss: {:.4f}\tAccuracy: {:.2f} ± {:.2f} %'
                .format(epoch, val_loss_avg, val_acc_avg, val_acc_ci95))

        if opt.wandb:
            wandb.log({"Validation Loss":val_loss_avg,"Val Avg Accuracy":val_acc_avg})

        torch.save({'embedding': embedding_net.state_dict(
        ), 'head': cls_head.state_dict()}, os.path.join(opt.save_path, 'last_epoch.pth'))

        if epoch % opt.save_epoch == 0:
            torch.save({'embedding': embedding_net.state_dict(), 'head': cls_head.state_dict(
            )}, os.path.join(opt.save_path, 'epoch_{}.pth'.format(epoch)))

        log(log_file_path, 'Elapsed Time: {}/{}\n'.format(timer.measure(),
                                                          timer.measure(epoch / float(opt.num_epoch))))

    # lr_scheduler.step()
