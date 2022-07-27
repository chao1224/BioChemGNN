import argparse
import random
import os
import numpy as np
import math
import tqdm
import warnings
warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
import torch.optim as optim

from BioChemGNN import data, datasets, utils, models
from torch.utils.data import SubsetRandomSampler


parser = argparse.ArgumentParser()
parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--weight_decay', type=float, default=0)
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--num_workers', type=int, default=0)
parser.add_argument('--epochs', type=int, default=200)

parser.add_argument('--dataset', type=str, default='delaney')
parser.add_argument('--model', type=str, default='gin')
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--split_seed', type=int, default=0)
parser.add_argument('--split_function', type=str, default='random_split',
                    choices=['random_split', 'scaffold_split'])
parser.add_argument('--print_every', type=int, default=10)

parser.add_argument('--gnn_hidden_dim', type=int, default=256)
parser.add_argument('--gnn_layer_size', type=int, default=6)
args = parser.parse_args()


config2dataset = {
    'bace': datasets.BACE,
    'bbbp': datasets.BBBP,
    'cep': datasets.CEP,
    'clintox': datasets.ClinTox,
    'delaney': datasets.Delaney,
    'freesolv': datasets.FreeSolv,
    'hiv': datasets.HIV,
    'lipophilicity': datasets.Lipophilicity,
    'malaria': datasets.Malaria,
    'muv': datasets.MUV,
    'sider': datasets.SIDER,
    'tox21': datasets.Tox21,
    'toxcast': datasets.ToxCast,
}


def train(train_dataloader, valid_dataloader, test_dataloader):
    for epoch in range(1, 1+args.epochs):
        model.train()
        loss_train = 0
        for batch in train_dataloader:
            batch = utils.cuda(batch)
            graph = batch['graph']
            y_actual = batch['label'].float()

            optimizer.zero_grad()
            y_pred = model(graph)['y']

            if mode == 'classification':
                y_actual = y_actual.long()
                is_valid = y_actual ** 2 > 0
                y_actual = (y_actual + 1) / 2
                y_actual = y_actual.float()
                loss = criterion(y_pred, y_actual)
                loss = torch.where(is_valid, loss, torch.zeros_like(loss))
                loss = torch.sum(loss) / torch.sum(is_valid)
            else:
                loss = criterion(y_pred, y_actual)

            loss.backward()
            optimizer.step()
            loss_train += loss.detach().item()
        print('Epoch: {}'.format(epoch), loss_train / len(train_dataloader))

        if epoch % args.print_every == 0:
            test(valid_dataloader, 'valid_data', metrics)
            test(test_dataloader, 'test_data', metrics)
        print()
    print('\n\n\n')
    return


def test(dataloader, eval_mode, metrics):
    model.eval()
    y_actual_list, y_predicted_list = [], []
    with torch.no_grad():
        for batch in dataloader:
            batch = utils.cuda(batch)
            graph = batch['graph']
            y_actual = batch['label']
            y_pred = model(graph)['y']
            if mode == 'classification':
                y_pred = sigmoid(y_pred)
            y_actual_list.append(y_actual.cpu().detach().numpy())
            y_predicted_list.append(y_pred.cpu().detach().numpy())

    y_actual_list = np.concatenate(y_actual_list, axis=0)
    y_predicted_list = np.concatenate(y_predicted_list, axis=0)

    for metric_name, metric_func in metrics.items():
        value_list = []
        for i in range(num_tasks):
            if mode == 'classification':
                if np.sum(y_actual_list[:, i] == 1) > 0 and np.sum(y_actual_list[:, i] == -1) > 0:
                    is_valid = y_actual_list[:, i] ** 2 > 0
                    value = metric_func(y_actual_list[is_valid, i], y_predicted_list[is_valid, i])
                    value_list.append(value)
            else:
                value = metric_func(y_actual_list[:, i], y_predicted_list[:, i])
                value_list.append(value)
        print('{}\t{}\t{}'.format(eval_mode, metric_name, np.mean(value_list)))
    return


if __name__ == '__main__':
    os.environ['PYTHONHASHargs.seed'] = str(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True

    if args.dataset in ['bace', 'bbbp', 'clintox', 'hiv', 'muv', 'sider', 'tox21', 'toxcast']:
        mode = 'classification'
        criterion = nn.BCEWithLogitsLoss(reduction="none")
        metrics = {'ROC': utils.roc_auc_score, 'PRC': utils.average_precision_score}
    else:
        mode = 'regression'
        criterion = nn.MSELoss()
        metrics = {'RMSE': utils.root_mean_squared_error, 'MAE': utils.mean_absolute_error}
    sigmoid = nn.Sigmoid()

    kwargs = {'node_feature': 'default', 'edge_feature': 'default'}
    dataset = config2dataset[args.dataset]('../datasets', **kwargs)

    if args.split_function == 'random_split':
        train_indices, valid_indices, test_indices = utils.random_split(dataset=dataset, split_seed=args.split_seed)
    elif args.split_function == 'scaffold_split':
        train_indices, valid_indices, test_indices = utils.scaffold_split(dataset=dataset, include_chirality=True)
    else:
        raise ValueError('Split function {} not included.'.format(args.split_function))
    if args.model == 'ecfp':
        dataset = datasets.MoleculeECFPDataset(dataset)
    num_tasks = len(dataset.target_fields)

    train_sampler = SubsetRandomSampler(train_indices)
    valid_sampler = SubsetRandomSampler(valid_indices)
    test_sampler = SubsetRandomSampler(test_indices)
    train_dataloader = data.DataLoader(dataset, sampler=train_sampler, batch_size=args.batch_size, num_workers=args.num_workers)
    valid_dataloader = data.DataLoader(dataset, sampler=valid_sampler, batch_size=args.batch_size, num_workers=args.num_workers)
    test_dataloader = data.DataLoader(dataset, sampler=test_sampler, batch_size=args.batch_size, num_workers=args.num_workers)
    print(len(train_indices), len(valid_indices), len(test_indices))

    if args.model == 'gin':
        model = models.GIN(
            node_feature_dim=dataset.node_feature_dim,
            hidden_dim=args.gnn_hidden_dim,
            layer_size=args.gnn_layer_size, output_dim=num_tasks
        )
    elif args.model == 'gin_enn':
        model = models.GIN(
            node_feature_dim=dataset.node_feature_dim,
            edge_feature_dim=dataset.edge_feature_dim,
            hidden_dim=args.gnn_hidden_dim,
            layer_size=args.gnn_layer_size, output_dim=num_tasks
        )
    elif args.model == 'gat':
        model = models.GAT(
            node_feature_dim=dataset.node_feature_dim,
            # hidden_dim=300, head_num=8, dropout=0.1, alpha=0.2,
            # layer_size=3, output_dim=1
            hidden_dim=args.gnn_hidden_dim, head_num=8, dropout=0., alpha=0.,
            layer_size=args.gnn_layer_size, output_dim=num_tasks
        )
    elif args.model == 'dmpnn':
        model = models.DMPNN(
            node_feature_dim=dataset.node_feature_dim,
            edge_feature_dim=dataset.edge_feature_dim,
            hidden_dim=300, layer_size=3, output_dim=num_tasks
        )
    elif args.model == 'ecfp':
        model = MLP(output_dim=num_tasks)
    else:
        raise ValueError('Model {} not included.'.format(args.model))

    model.cuda()
    print(model)

    parameters = list(model.parameters())
    optimizer = optim.Adam(parameters, lr=args.lr, weight_decay=args.weight_decay)

    train(train_dataloader, valid_dataloader, test_dataloader)

    test(train_dataloader, 'train_data', metrics)
    test(valid_dataloader, 'valid_data', metrics)
    test(test_dataloader, 'test_data', metrics)