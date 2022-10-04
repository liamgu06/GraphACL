import os.path as osp
import random

import torch
from torch_geometric.data import DataLoader
from torch_geometric.datasets import TUDataset
from encoder import GNN
import time
import numpy as np
from sklearn.model_selection import GridSearchCV, StratifiedKFold

from graphacl import GraphACL, Adversary_Negatives
from tqdm import tqdm
import sys
import os
import logging
from aug import TUDataset_aug
from arguments import arg_parse


sys.path.insert(0, '../..')
args = arg_parse()


def create_mkdir(path):
    isExists = os.path.exists(path)
    if not isExists:
        os.makedirs(path)


def train(model, device, loader, optimizer, args, memory_bank, optimizer_mem):
    loss_all = 0
    loss_list_np = np.zeros(3)
    model.train()
    for step, batch in tqdm(enumerate(loader)):
        data, data_aug = batch

        data = data.to(device)
        data_aug = data_aug.to(device)
        if data.x.shape[0] == 1 or data.batch[-1] == 0:
            pass
        else:
            optimizer.zero_grad()
            optimizer_mem.zero_grad()

            loss_sim, loss_so, loss_sd, loss_list = model(data, data_aug, memory_bank)

            loss_sim.backward(retain_graph=True)
            # update memory bank
            if args.adversarial == 1:
                for para in memory_bank.parameters():
                    para.grad = -para.grad
            else:
                optimizer_mem.zero_grad()

            # loss_neg = 1 * (loss_so + loss_sd)
            loss_neg = args.lambda1 * loss_sd + args.lambda2 * loss_so

            loss_neg.backward()
            optimizer.step()
            optimizer_mem.step()

            model.update_moving_average()

            loss_all += (loss_sim.item()+loss_neg.item()) * data.num_graphs
            loss_list_np += np.asarray(loss_list) * data.num_graphs

    return loss_all / len(loader.dataset), loss_list_np / len(loader.dataset)


def eval(model, device, loader):
    model.eval()
    all_embed = []
    y_true = []

    for step, batch_tuple in enumerate(loader):
        batch, data_aug = batch_tuple
        batch = batch.to(device)

        if batch.x.shape[0] == 1:
            pass
        else:
            with torch.no_grad():
                batch_embed = model.embed(batch)
                all_embed.append(batch_embed)
                y_true.append(batch.y.detach().cpu())

    all_embed = torch.cat(all_embed, dim=0)
    y_true = torch.cat(y_true, dim=0)
    x = all_embed.cpu().detach().numpy()
    y = y_true.cpu().detach().numpy()

    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    x = scaler.fit_transform(x)
    from sklearn.svm import LinearSVC
    from sklearn.metrics import accuracy_score
    # params = {'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000]}
    params = {'C': [0.01, 0.1, 1, 10]}
    kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=args.seed)
    accuracies = []
    for train_index, test_index in tqdm(kf.split(x, y)):
        x_train, x_test = x[train_index], x[test_index]
        y_train, y_test = y[train_index], y[test_index]
        classifier = GridSearchCV(LinearSVC(dual=False), params, cv=5, scoring='accuracy', verbose=0, n_jobs=-1)
        classifier.fit(x_train, y_train)
        accuracies.append(accuracy_score(y_test, classifier.predict(x_test)))
    return np.mean(accuracies), np.std(accuracies), x, y


def get_one_hot(dataset):
    g_idx = 0
    total_node = 0
    for i in dataset.data.num_nodes:
        total_node += i
    total_degree = np.zeros(total_node)
    node_start = 0
    node_end = 0
    for i in dataset.data.num_nodes:
        node_end += i
        edge_start = dataset.slices['edge_index'][g_idx]
        edge_end = dataset.slices['edge_index'][g_idx+1]
        edges = dataset.data.edge_index[:, edge_start:edge_end]
        in_degree = out_degree = np.zeros(i)

        for ee in edges:
            in_degree[ee] += 1
            out_degree[ee] += 1

        tot_degree = in_degree + out_degree
        total_degree[node_start:node_end] = tot_degree
        node_start = node_end
        g_idx += 1

    total_degree = total_degree.astype(np.int64)
    return torch.nn.functional.one_hot(torch.tensor(torch.from_numpy(total_degree))).float()


def seed_everything(seed=42):
    """"
    Seed everything.
    """
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def main(logger):
    # print key configurations
    print('########################################################################')
    print('########################################################################')
    print(f'dataset:                                  {args.dataset}')
    print(f'num_layer:                                {args.num_layer}')
    print(f'number of epochs:                         {args.epochs}')
    print(f'emb_dim:                                  {args.emb_dim}')
    print(f'batch_size:                               {args.batch_size}')
    print(f'lr:                                       {args.lr}')
    print('########################################################################')
    print('########################################################################')

    best_result = -1
    all_results = []
    seeds = [args.seed]
    for run in range(len(seeds)):
        seed = seeds[run]
        seed_everything(seed)

        device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")

        path = osp.join(osp.dirname(osp.realpath(__file__)), 'data', args.dataset)
        dataset = TUDataset_aug(path, name=args.dataset, aug1=args.aug1, aug_ratio=args.aug_ratio).shuffle()
        dataset.data.edge_attr = None
        dataset_eval = TUDataset(path, name=args.dataset).shuffle()
        dataset_eval.data.edge_attr = None
        dataset_num_features = max(dataset_eval.num_features, 1)

        print('================')
        print('num_graphs: {}'.format(len(dataset)))
        print('lr: {}'.format(args.lr))
        print('num_features: {}'.format(dataset_num_features))
        print('================')

        max_node_num = max(dataset_eval.data.num_nodes)
        min_node_num = min(dataset_eval.data.num_nodes)
        print('max_node_num:', max_node_num, 'min_node_num:', min_node_num)

        if dataset.data.x is None:
            dataset.data.x = get_one_hot(dataset)
        else:
            dataset.data.x = dataset.data.x.float()

        if dataset.data.x is not None:
            feat_dim = dataset.data.x.shape[-1]

        dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                                num_workers=args.num_workers)

        if 'x' not in dataloader.dataset.slices:
            tmp = torch.LongTensor(len(dataset.data.num_nodes) + 1)
            accum_node = 0
            tmp[0] = 0
            for i in range(len(dataset.data.num_nodes)):
                accum_node += dataset.data.num_nodes[i]
                tmp[i + 1] = accum_node
            dataloader.dataset.slices['x'] = tmp

        best_acc = -1
        best_std = -1
        results = []

        if args.gnn == 'gin':
            gnnmodel = GNN(gnn_type='gin', num_layer=args.num_layer, emb_dim=args.emb_dim, graph_pooling=args.pooling_global,
                           drop_ratio=args.drop_ratio, virtual_node=False, feat_dim=feat_dim).to(device)
        elif args.gnn == 'gcn':
            gnnmodel = GNN(gnn_type='gcn', num_layer=args.num_layer, emb_dim=args.emb_dim, graph_pooling=args.pooling_global,
                           drop_ratio=args.drop_ratio, virtual_node=False, feat_dim=feat_dim).to(device)
        elif args.gnn == 'sage':
            gnnmodel = GNN(gnn_type='sage', num_layer=args.num_layer, emb_dim=args.emb_dim,
                           graph_pooling=args.pooling_global,
                           drop_ratio=args.drop_ratio, virtual_node=False, feat_dim=feat_dim).to(device)
        else:
            raise ValueError('Invalid GNN type')

        model = GraphACL(gnnmodel, feat_dim=feat_dim, args=args, beta=args.beta, emb_dim=args.emb_dim,
                         projection_size=args.projection_size, prediction_size=args.prediction_size,
                         projection_hidden_size=args.projection_hidden_size, moving_average_decay=args.decay,
                         alpha=args.alpha)
        model.to(device)

        memory_bank = Adversary_Negatives(args.cluster, args.bank_dim).to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        optimizer_mem = torch.optim.Adam(memory_bank.parameters(), lr=args.lr)
        cnt_wait = 0
        best = 1e9
        patience = 50
        train_loss = np.zeros(args.epochs+1)
        for epoch in tqdm(range(1, args.epochs + 1)):
            loss, loss_list_np = train(model, device, dataloader, optimizer, args, memory_bank, optimizer_mem)
            train_loss[epoch] = loss
            logger.info("loss_total={:.4f},loss_sim={:.4e},loss_so={:.4e},loss_sd={:.4e}".format(loss,
                                                                                                 loss_list_np[0],
                                                                                                 loss_list_np[1],
                                                                                                 loss_list_np[2]))
            
            if loss < best:
                best = loss
                cnt_wait = 0
            else:
                cnt_wait += 1

            if cnt_wait > patience:
                break

            if epoch % args.test_freq == 0:
                acc, std, x, y = eval(model, device, dataloader)

                results.append([seed, epoch, acc, std])
                if acc > best_acc:
                    best_acc, best_std = acc, std

                logger.info(f'acc mean {acc:.5f}, std {std:.5f}, best acc mean {best_acc:.5f}, std {best_std:.5f}, loss {loss:.5f}')
        for r in results:
            print(f'seed{r[0]}, epoch{r[1]} acc:{r[2]:.5f} std:{r[3]:.5f}')
            best_result = r[2] if r[2] > best_result else best_result
        all_results.append(results)

    logger.info(f'best acc = {best_result}')


if __name__ == "__main__":
    create_mkdir(args.log_dir)
    args.prediction_size = args.emb_dim
    args.projection_size = args.emb_dim
    log_path = os.path.join(args.log_dir, args.log_file)
    print('logging into %s' % log_path)

    logger = logging.getLogger(__name__)
    logger.setLevel(level=logging.INFO)
    handler = logging.FileHandler(log_path)
    handler.setLevel(logging.INFO)
    logger.addHandler(handler)

    logger.info('#'*20)
    localtime = time.asctime(time.localtime(time.time()))
    logger.info("%s" % localtime)

    # record arguments
    logger.info("%s" % args.dataset)
    args_str = ""
    for k, v in sorted(vars(args).items()):
        args_str += "%s" % k + "=" + "%s" % v + "; "
    logger.info(args_str)
    # print(args_str)
    logger.info("args.dataset: %s" % args.dataset)
    logger.info("args.num_layer: %s" % args.num_layer)
    logger.info("args.emb_dim: %s" % args.emb_dim)
    logger.info("args.batch_size: %s" % args.batch_size)

    main(logger)

    logger.info("\n")

