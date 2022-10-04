import torch
from torch import nn
import torch.nn.functional as F
import copy
from functools import wraps
import numpy as np


class EMA():
    def __init__(self, beta):
        super().__init__()
        self.beta = beta

    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new


class Adversary_Negatives(nn.Module):
    def __init__(self, bank_size, dim):
        super(Adversary_Negatives, self).__init__()
        self.W = nn.Parameter(torch.randn(dim, bank_size))
        torch.nn.init.xavier_uniform_(self.W.data)

    def forward(self, q):
        memory_bank = self.W
        memory_bank = nn.functional.normalize(memory_bank, dim=0)
        logit = torch.einsum('nc,ck->nk', [q, memory_bank])
        return memory_bank, self.W, logit

    def print_weight(self):
        print(torch.sum(self.W).item())


def update_moving_average(ema_updater, ma_model, current_model):
    for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
        old_weight, up_weight = ma_params.data, current_params.data
        ma_params.data = ema_updater.update_average(old_weight, up_weight)


def contrastive_loss_wo_aug(x, T=0.2):
    batch_size, _ = x.size()
    x_norm = F.normalize(x)
    sim_matrix = torch.exp(torch.mm(x_norm, x_norm.t().contiguous()) / T)
    mask = (torch.ones_like(sim_matrix) - torch.eye(batch_size, device=sim_matrix.device)).bool()
    sim_matrix = sim_matrix.masked_select(mask).view(batch_size, -1)
    pos_sim = torch.exp(torch.sum(x_norm**2, dim=-1) / T)
    loss = (-torch.log(pos_sim / sim_matrix.sum(dim=-1))).mean()
    return loss


class MLP(nn.Module):
    def __init__(self, dim, hidden_size, projection_size):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(inplace=True),
            #nn.PReLU(),
            nn.Linear(hidden_size, projection_size)
        )

    def forward(self, x):
        return self.net(x)


def singleton(cache_key):
    def inner_fn(fn):
        @wraps(fn)
        def wrapper(self, *args, **kwargs):
            instance = getattr(self, cache_key)
            if instance is not None:
                return instance

            instance = fn(self, *args, **kwargs)
            setattr(self, cache_key, instance)
            return instance
        return wrapper
    return inner_fn


class GraphACL(nn.Module):
    def __init__(self, net, feat_dim, args, beta, emb_dim=512, projection_hidden_size=64, projection_size=512,
                 prediction_size=512, moving_average_decay=0.99, alpha=1.0):
        super().__init__()

        self.projection_hidden_size = projection_hidden_size

        self.online_encoder = net
        self.target_encoder = None
        self.target_ema_updater = EMA(moving_average_decay)
        self.online_predictor = MLP(emb_dim, projection_hidden_size, prediction_size)
        self.beta = beta
        self.alpha = alpha
        self.device = args.device
        self.model = args.model
        self.temperature = args.bank_t

    @singleton('target_encoder')
    def _get_target_encoder(self):
        target_encoder = copy.deepcopy(self.online_encoder)
        return target_encoder

    def reset_moving_average(self):
        del self.target_encoder
        self.target_encoder = None

    def update_moving_average(self):
        assert self.target_encoder is not None, 'target encoder has not been created yet'
        update_moving_average(self.target_ema_updater, self.target_encoder, self.online_encoder)

    @staticmethod
    def infoNCE_loss(x, y, memory_bank, temperature):
        x = F.normalize(x, dim=-1, p=2)
        y = F.normalize(y, dim=-1, p=2)
        d_norm, d, l_neg = memory_bank(x)
        pos_sim = torch.exp(torch.sum(x * y, dim=-1) / temperature)
        sim_matrix = torch.exp(l_neg / temperature)
        loss_sim = -(torch.log(pos_sim / (pos_sim + sim_matrix.sum(dim=-1)))).mean()
        return loss_sim

    @staticmethod
    def BYOL_like_loss(x, y, memory_bank, alpha):
        x = F.normalize(x, dim=-1, p=2)
        y = F.normalize(y, dim=-1, p=2)
        d_norm, d, l_neg = memory_bank(x)
        pos_sim = torch.sum(x * y, dim=-1)
        neg_sim = l_neg.mean(dim=-1)
        loss_sim = (2 * alpha * neg_sim - 2 * pos_sim + 2 - 2 * alpha).mean()
        return loss_sim

    def forward(self, data, data_aug, memory_bank):
        graph_online_proj_one = self.online_encoder(data)
        graph_online_proj_two = self.online_encoder(data_aug)

        graph_online_pred_one = self.online_predictor(graph_online_proj_one)
        graph_online_pred_two = self.online_predictor(graph_online_proj_two)

        with torch.no_grad():
            target_encoder = self._get_target_encoder()
            graph_target_proj_one = target_encoder(data)
            graph_target_proj_two = target_encoder(data_aug)

        if self.model == 'graphacl':
            l1_graph = self.infoNCE_loss(graph_online_pred_one, graph_target_proj_two.detach(),
                                         memory_bank, self.temperature)
            l2_graph = self.infoNCE_loss(graph_online_pred_two, graph_target_proj_one.detach(),
                                         memory_bank, self.temperature)
        elif self.model == 'graphacl_star':
            l1_graph = self.BYOL_like_loss(graph_online_pred_one, graph_target_proj_two.detach(), memory_bank,
                                           self.alpha)
            l2_graph = self.BYOL_like_loss(graph_online_pred_two, graph_target_proj_one.detach(), memory_bank,
                                           self.alpha)
        else:
            raise ValueError('Invalid model!')

        loss_sim = 0.5 * (l1_graph + l2_graph).mean()

        # Set Orthogonality Loss
        loss_so = contrastive_loss_wo_aug(memory_bank.W.t(), T=0.2)

        # Set Divergence Loss
        loss_sd = contrastive_loss_wo_aug(memory_bank.W, T=0.2)

        loss_list = np.array([loss_sim.item(), loss_so.item(), loss_sd.item()])

        return loss_sim, loss_so, loss_sd, loss_list

    def embed(self, batch_data):
        graph_online_proj_one = self.online_encoder(batch_data)
        return graph_online_proj_one.detach()


