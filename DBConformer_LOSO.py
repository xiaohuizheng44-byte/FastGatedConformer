'''
=================================================
coding:utf-8
@Time:      2025/5/6 10:58
@File:      DBConformer_LOSO.py
@Author:    Ziwei Wang
@Function: Leave-One-Subject-Out (LOSO) scenario
=================================================
'''
import math
import numpy as np
import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from utils.network import backbone_net, backbone_net_deep, backbone_net_shallow, backbone_net_ifnet, \
    backbone_net_fbcnet, backbone_net_adfcnn, backbone_net_conformer, backbone_net_fbmsnet, backbone_net_dbconformer, \
    backbone_net_dgmambaconformer, backbone_net_fastgatedconformer
from utils.LogRecord import LogRecord
from utils.dataloader import read_mi_combine_tar
from utils.utils import fix_random_seed, cal_acc_comb, data_loader
import gc
import sys
import warnings
warnings.filterwarnings('ignore')


def infer_feature_deep_dim(netF, args):
    was_training = netF.training
    netF.eval()
    with torch.no_grad():
        if args.backbone in ['EEGNet', 'deep', 'shallow']:
            dummy = torch.zeros(2, 1, args.chn, args.time_sample_num)
        else:
            dummy = torch.zeros(2, args.chn, args.time_sample_num)
        features = netF(dummy)
        if isinstance(features, tuple):
            features = features[0]
        feature_dim = int(features.reshape(features.size(0), -1).shape[1])
    netF.train(was_training)
    return feature_dim


def parse_cli_args():
    parser = argparse.ArgumentParser(description='LOSO evaluation for EEG decoding models.')
    parser.add_argument('device_id', nargs='?', default=None,
                        help='Optional CUDA device id, e.g. 0. If omitted, CUDA is used when available.')
    parser.add_argument('--backbone', default='FastGatedConformer',
                        choices=['EEGNet', 'deep', 'shallow', 'IFNet', 'FBCNet', 'ADFCNN',
                                 'Conformer', 'FBMSNet', 'DBConformer', 'DGMambaConformer',
                                 'FastGatedConformer'],
                        help='Backbone model to evaluate.')
    parser.add_argument('--data-names', nargs='+', default=['BNCI2014002'],
                        help='Dataset names to evaluate.')
    parser.add_argument('--seeds', nargs='+', type=int, default=[1, 2, 3, 4, 5],
                        help='Random seeds for repeated runs.')
    parser.add_argument('--max-epoch', type=int, default=100,
                        help='Training epochs for each subject split.')
    parser.add_argument('--subject-adv-weight', type=float, default=0.0,
                        help='GRL subject adversarial loss weight. Used only by DGMambaConformer.')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Training batch size.')
    parser.add_argument('--branch', default='all', choices=['all', 'temporal', 'spatial'],
                        help='Feature branch to use. all = temporal + spatial fusion.')
    parser.add_argument('--no-channel-attention', action='store_true',
                        help='Disable channel attention pooling and use mean pooling for spatial features.')
    return parser.parse_args()


class GradientReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, coeff):
        ctx.coeff = coeff
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.coeff * grad_output, None


class SubjectDiscriminator(nn.Module):
    def __init__(self, in_dim, num_subjects, hidden_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ELU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_dim, num_subjects),
        )

    def forward(self, x, coeff=1.0):
        x = GradientReverse.apply(x, coeff)
        return self.net(x)


def make_source_subject_labels(args, sample_num):
    if args.trials_arr:
        counts = [int(args.trials_arr[i]) for i in range(args.N) if i != args.idt]
    else:
        counts = [int(sample_num // (args.N - 1))] * (args.N - 1)
    labels = []
    domain_id = 0
    for count in counts:
        labels.extend([domain_id] * count)
        domain_id += 1
    if len(labels) != sample_num:
        labels = (labels + [domain_id - 1] * sample_num)[:sample_num]
    return np.asarray(labels, dtype=np.int64)


def train_target(args):
    X_src, y_src, X_tar, y_tar = read_mi_combine_tar(args)
    print('X_src, y_src, X_tar, y_tar:', X_src.shape, y_src.shape, X_tar.shape, y_tar.shape)
    args.time_sample_num = int(X_src.shape[-1])
    args.chn = int(X_src.shape[1])
    source_subject_labels = make_source_subject_labels(args, X_src.shape[0])
    args.return_source_index = args.backbone == 'DGMambaConformer' and args.subject_adv_weight > 0
    dset_loaders = data_loader(X_src, y_src, X_tar, y_tar, args)
    # network selection
    netC = None
    if args.backbone == 'EEGNet':
        netF, netC = backbone_net(args, return_type='xy')
    elif args.backbone == 'deep':
        netF, netC = backbone_net_deep(args, return_type='xy')
    elif args.backbone == 'shallow':
        netF, netC = backbone_net_shallow(args, return_type='xy')
    elif args.backbone == 'IFNet':
        netF, netC = backbone_net_ifnet(args, return_type='xy')
    elif args.backbone == 'FBCNet':
        netF = backbone_net_fbcnet(args, return_type='xy')
    elif args.backbone == 'ADFCNN':
        netF, netC = backbone_net_adfcnn(args, return_type='xy')
    elif args.backbone == 'Conformer':
        netF = backbone_net_conformer(args, return_type='xy')
    elif args.backbone == 'FBMSNet':
        netF, netC = backbone_net_fbmsnet(args, return_type='xy')
    elif args.backbone == 'DBConformer':
        netF = backbone_net_dbconformer(args)
    elif args.backbone == 'DGMambaConformer':
        netF = backbone_net_dgmambaconformer(args)
    elif args.backbone == 'FastGatedConformer':
        netF = backbone_net_fastgatedconformer(args)
    model_modules = [netF] if netC is None else [netF, netC]
    model_params = sum(p.numel() for module in model_modules for p in module.parameters())
    trainable_params = sum(p.numel() for module in model_modules for p in module.parameters() if p.requires_grad)
    param_str = 'Model params: total={}, trainable={}'.format(model_params, trainable_params)
    print(param_str)
    args.log.record(param_str)
    if args.data_env != 'local':
        if args.backbone == 'FBCNet' or args.backbone == 'Conformer' or args.backbone == 'DBConformer' or args.backbone == 'DGMambaConformer' or args.backbone == 'FastGatedConformer':
            netF = netF.cuda()
            base_network = netF
            optimizer_f = optim.Adam(netF.parameters(), lr=args.lr)
        else:
            netF, netC = netF.cuda(), netC.cuda()
            base_network = nn.Sequential(netF, netC)
            optimizer_f = optim.Adam(netF.parameters(), lr=args.lr)
            optimizer_c = optim.Adam(netC.parameters(), lr=args.lr)
    if args.data_env == 'local':
        if args.backbone == 'FBCNet' or args.backbone == 'Conformer' or args.backbone == 'DBConformer' or args.backbone == 'DGMambaConformer' or args.backbone == 'FastGatedConformer':
            base_network = netF
            optimizer_f = optim.Adam(netF.parameters(), lr=args.lr)
        else:
            base_network = nn.Sequential(netF, netC)
            optimizer_f = optim.Adam(netF.parameters(), lr=args.lr)
            optimizer_c = optim.Adam(netC.parameters(), lr=args.lr)

    use_subject_adv = args.backbone == 'DGMambaConformer' and args.subject_adv_weight > 0
    if use_subject_adv:
        adv_in_dim = args.emb_size if args.branch != 'all' or args.gate_flag else args.emb_size * 2
        subject_discriminator = SubjectDiscriminator(adv_in_dim, args.N - 1)
        if args.data_env != 'local':
            subject_discriminator = subject_discriminator.cuda()
        optimizer_d = optim.Adam(subject_discriminator.parameters(), lr=args.lr)
        source_subject_labels = torch.from_numpy(source_subject_labels).long()
        if args.data_env != 'local':
            source_subject_labels = source_subject_labels.cuda()
    device = torch.device('cuda' if args.data_env != 'local' else 'cpu')
    if args.class_num == 2:
        class_weight = torch.tensor([1., args.weight], dtype=torch.float32, device=device)  # class imbalance
        criterion = nn.CrossEntropyLoss(weight=class_weight)
    else:
        criterion = nn.CrossEntropyLoss()
    domain_criterion = nn.CrossEntropyLoss()
    max_iter = args.max_epoch * len(dset_loaders["source"])
    interval_iter = max_iter // args.max_epoch
    args.max_iter = max_iter
    iter_num = 0
    base_network.train()
    if use_subject_adv:
        subject_discriminator.train()
    while iter_num < max_iter:
        try:
            source_batch = next(iter_source)
        except:
            iter_source = iter(dset_loaders["source"])
            source_batch = next(iter_source)
        if use_subject_adv:
            inputs_source, labels_source, batch_indices = source_batch
        else:
            inputs_source, labels_source = source_batch
            batch_indices = None
        if inputs_source.size(0) == 1:
            continue
        iter_num += 1
        if 'ADFCNN' in args.backbone or 'Conformer' in args.backbone or args.backbone == 'DBConformer' or args.backbone == 'DGMambaConformer' or args.backbone == 'FastGatedConformer':
            inputs_source = inputs_source.unsqueeze_(3)
            inputs_source = inputs_source.permute(0, 3, 1, 2)
        features_source, outputs_source = base_network(inputs_source)
        classifier_loss = criterion(outputs_source, labels_source)
        total_loss = classifier_loss
        if use_subject_adv:
            domain_labels = source_subject_labels[batch_indices]
            coeff = args.subject_adv_weight * (2.0 / (1.0 + math.exp(-10 * iter_num / max_iter)) - 1.0)
            domain_outputs = subject_discriminator(features_source, coeff)
            domain_loss = domain_criterion(domain_outputs, domain_labels)
            total_loss = total_loss + domain_loss
        optimizer_f.zero_grad()
        if args.backbone != 'FBCNet' and args.backbone != 'Conformer' and args.backbone != 'DBConformer' and args.backbone != 'DGMambaConformer' and args.backbone != 'FastGatedConformer':
            optimizer_c.zero_grad()
        if use_subject_adv:
            optimizer_d.zero_grad()
        total_loss.backward()
        optimizer_f.step()
        if args.backbone != 'FBCNet' and args.backbone != 'Conformer' and args.backbone != 'DBConformer' and args.backbone != 'DGMambaConformer' and args.backbone != 'FastGatedConformer':
            optimizer_c.step()
        if use_subject_adv:
            optimizer_d.step()
        if iter_num % interval_iter == 0 or iter_num == max_iter:
            base_network.eval()
            acc_t_te, _ = cal_acc_comb(dset_loaders["target-online"], base_network, args=args)  # TODO target-online
            log_str = 'Task: {}, Iter:{}/{}; Acc = {:.2f}%'.format(args.task_str, int(iter_num // len(dset_loaders["source"])), int(max_iter // len(dset_loaders["source"])), acc_t_te)
            args.log.record(log_str)
            base_network.train()
            if use_subject_adv:
                subject_discriminator.train()
    print('Test Acc = {:.2f}%'.format(acc_t_te))
    print('saving model...')
    if not os.path.exists('./runs/' + str(args.data_name) + '/'):
        os.makedirs('./runs/' + str(args.data_name) + '/')
    if args.align:
        torch.save(base_network.state_dict(),
                   './runs/' + str(args.data_name) + '/' + str(args.backbone) + '_S' + str(args.idt) + '_seed' + str(args.SEED) + '.ckpt')
    else:
        torch.save(base_network.state_dict(),
                   './runs/' + str(args.data_name) + '/' + str(args.backbone) + '_S' + str(args.idt) + '_seed' + str(args.SEED) + '_noEA' + '.ckpt')
    gc.collect()
    if args.data_env != 'local':
        torch.cuda.empty_cache()
    return acc_t_te


if __name__ == '__main__':
    cli_args = parse_cli_args()
    cpu_num = 8
    torch.set_num_threads(cpu_num)
    data_name_list = cli_args.data_names  # 'BNCI2014001', BNCI2014004, 'Zhou2016', 'MI1-7' (Blankertz2007), 'BNCI2014002'
    dct = pd.DataFrame(columns=['dataset', 'avg', 'std', 's0', 's1', 's2', 's3', 's4', 's5', 's6', 's7', 's8', 's9', 's10', 's11', 's12', 's13'])

    for data_name in data_name_list:
        weight = 1
        if data_name == 'BNCI2014001': paradigm, N, chn, class_num, time_sample_num, sample_rate, trial_num = 'MI', 9, 22, 2, 1001, 250, 144
        if data_name == 'BNCI2014002': paradigm, N, chn, class_num, time_sample_num, sample_rate, trial_num = 'MI', 14, 15, 2, 2561, 512, 100
        if data_name == 'BNCI2014004': paradigm, N, chn, class_num, time_sample_num, sample_rate, trial_num = 'MI', 9, 3, 2, 1126, 250, 120
        if data_name == 'BNCI2015001': paradigm, N, chn, class_num, time_sample_num, sample_rate, trial_num = 'MI', 12, 13, 2, 2561, 512, 200
        if data_name == 'BNCI2014001-4': paradigm, N, chn, class_num, time_sample_num, sample_rate, trial_num = 'MI', 9, 22, 4, 1001, 250, 288
        if data_name == 'MI1-7': paradigm, N, chn, class_num, time_sample_num, sample_rate, trial_num = 'MI', 7, 59, 2, 750, 250, 200
        if data_name == 'MI1': paradigm, N, chn, class_num, time_sample_num, sample_rate, trial_num, dim_e, dim_p = 'MI', 5, 59, 2, 750, 250, 200, 184, 750
        if data_name == 'BNCI2014008': paradigm, N, chn, class_num, time_sample_num, sample_rate, trial_num, weight = 'ERP', 8, 8, 2, 256, 256, 4200, 64,
        if data_name == 'BNCI2015003': paradigm, N, chn, class_num, time_sample_num, sample_rate, trial_num, weight = 'ERP', 10, 8, 2, 206, 256, 2520, 64
        if data_name == 'Zhou2016': paradigm, N, chn, class_num, time_sample_num, sample_rate, trial_num = 'MI', 4, 14, 2, 1251, 250, -1
        if data_name == 'Zhou2016_3': paradigm, N, chn, class_num, time_sample_num, sample_rate, trial_num = 'MI', 4, 14, 3, 1251, 250, -1
        F1, D, F2 = 4, 2, 8

        if 'BNCI2014008' in data_name:
            F1, D, F2 = 8, 4, 16
        args = argparse.Namespace(trial_num=trial_num,
                                  time_sample_num=time_sample_num, sample_rate=sample_rate,
                                  N=N, chn=chn, class_num=class_num, paradigm=paradigm, data_name=data_name,
                                  F1=F1, D=D, F2=F2, weight=weight)

        args.backbone = cli_args.backbone  # FastGatedConformer, DGMambaConformer, DBConformer
        args.method = args.backbone + '_' + data_name
        # DBConformer parameters
        args.gate_flag = False
        args.posemb_flag = True
        args.chn_atten_flag = not cli_args.no_channel_attention
        args.branch = cli_args.branch  # [all, temporal, spatial]
        if args.backbone == 'DBConformer' or args.backbone == 'DGMambaConformer' or args.backbone == 'FastGatedConformer':
            args.emb_size = 40
            args.spa_dim = 16
            if data_name == 'BNCI2014001' or data_name == 'BNCI2014004':
                args.transformer_depth_tem = 2
                args.transformer_depth_chn = 2
            else:
                args.transformer_depth_tem = 6
                args.transformer_depth_chn = 6
            if data_name == 'BNCI2015001' or data_name == 'BNCI2014002':
                args.patch_size = 128
            else:
                args.patch_size = 125
        # Domain generalization: adversarially remove source-subject identity from features.
        args.subject_adv_weight = cli_args.subject_adv_weight
        # whether to use EA
        args.align = True
        args.dropoutRate = 0.25
        # learning rate
        args.lr = 0.001
        # train batch size
        args.batch_size = cli_args.batch_size
        # training epochs
        args.max_epoch = cli_args.max_epoch
        # GPU device id
        if cli_args.device_id is not None:
            device_id = str(cli_args.device_id)
            os.environ["CUDA_VISIBLE_DEVICES"] = device_id
            args.data_env = 'gpu' if torch.cuda.device_count() != 0 else 'local'
        else:
            args.data_env = 'gpu' if torch.cuda.is_available() else 'local'
        total_acc = []
        for s in cli_args.seeds:
            args.SEED = s
            fix_random_seed(args.SEED)
            torch.backends.cudnn.deterministic = True
            args.data = data_name
            print(args.data)
            print(args.method)
            print(args.SEED)
            print(args)
            args.local_dir = './data/' + str(data_name) + '/'
            args.result_dir = './logs/'
            my_log = LogRecord(args)
            my_log.log_init()
            my_log.record('=' * 50 + '\n' + os.path.basename(__file__) + '\n' + '=' * 50)
            sub_acc_all = np.zeros(N)
            for idt in range(N):
                args.idt = idt
                source_str = 'Except_S' + str(idt)
                target_str = 'S' + str(idt)
                args.task_str = source_str + '_2_' + target_str
                info_str = '\n========================== Transfer to ' + target_str + ' =========================='
                print(info_str)
                my_log.record(info_str)
                args.log = my_log
                sub_acc_all[idt] = train_target(args)
            print('Sub acc: ', np.round(sub_acc_all, 3))
            print('Avg acc: ', np.round(np.mean(sub_acc_all), 3))
            total_acc.append(sub_acc_all)
            acc_sub_str = str(np.round(sub_acc_all, 3).tolist())
            acc_mean_str = str(np.round(np.mean(sub_acc_all), 3).tolist())
            args.log.record("\n==========================================")
            args.log.record(acc_sub_str)
            args.log.record(acc_mean_str)
        args.log.record('\n' + '#' * 20 + 'final results' + '#' * 20)
        print(str(total_acc))
        args.log.record(str(total_acc))
        subject_mean = np.round(np.average(total_acc, axis=0), 5)
        total_mean = np.round(np.average(np.average(total_acc)), 5)
        total_std = np.round(np.std(np.average(total_acc, axis=1)), 5)
        print(subject_mean)
        print(args.method)
        print(total_mean)
        print(total_std)
        args.log.record(str(subject_mean))
        args.log.record(str(total_mean))
        args.log.record(str(total_std))
