import os
import pickle

import torch
import torch.nn.functional as F
from torch_geometric.utils import get_laplacian, to_scipy_sparse_matrix, to_dense_adj
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import expm


class PositionEncoding(object):
    def __init__(self, savepath=None, zero_diag=False):
        self.savepath = savepath
        self.zero_diag = zero_diag

    def apply_to(self, dataset, split='train'):
        saved_pos_enc = self.load(split)
        all_pe = []
        dataset.pe_list = []
        for i, g in enumerate(dataset):
            if saved_pos_enc is None:
                pe = self.compute_pe(g)
                all_pe.append(pe)
            else:
                pe = saved_pos_enc[i]
            if self.zero_diag:
                pe = pe.clone()
                if hasattr(self, "use_edge_attr") and self.use_edge_attr:
                    pe.diagonal(dim1=1, dim2=2)[:] = 0
                else:
                    pe.diagonal()[:] = 0
            dataset.pe_list.append(pe)

        self.save(all_pe, split)

        return dataset

    def save(self, pos_enc, split):
        if self.savepath is None:
            return
        if not os.path.isfile(self.savepath + "." + split):
            with open(self.savepath + "." + split, 'wb') as handle:
                pickle.dump(pos_enc, handle)

    def load(self, split):
        if self.savepath is None:
            return None
        if not os.path.isfile(self.savepath + "." + split):
            return None
        with open(self.savepath + "." + split, 'rb') as handle:
            pos_enc = pickle.load(handle)
        return pos_enc

    def compute_pe(self, graph):
        pass


class DiffusionEncoding(PositionEncoding):
    def __init__(self, savepath, beta=1., use_edge_attr=False, normalization=None, zero_diag=False, num_edge_features=4):
        """
        normalization: for Laplacian None. sym or rw
        """
        super().__init__(savepath, zero_diag)
        self.beta = beta
        self.normalization = normalization
        self.use_edge_attr = use_edge_attr
        self.num_edge_features = num_edge_features
        if isinstance(num_edge_features, int):
            self.num_edge_features_true = num_edge_features
        else:
            self.num_edge_features_true = sum(num_edge_features)

    def compute_pe(self, graph):
        if self.use_edge_attr:
            # edge_attr_list = F.one_hot(graph.edge_attr - 1, self.num_edge_features).float()
            edge_attr_list = edge_attr_one_hot(graph.edge_attr, self.num_edge_features)
            pe_tensors = torch.zeros((self.num_edge_features_true, graph.num_nodes, graph.num_nodes))
            for i in range(self.num_edge_features_true):
                pe_i = self.compute_pe_from_edge_weight(
                    graph.edge_index, edge_attr_list[:, i], graph.num_nodes)
                pe_tensors[i] = pe_i
            return pe_tensors
        return self.compute_pe_from_edge_weight(graph.edge_index, None, graph.num_nodes)

    def compute_pe_from_edge_weight(self, edge_index, edge_weight, num_nodes):
        edge_index, edge_weight = get_laplacian(
                edge_index, edge_weight, normalization=self.normalization,
                num_nodes=num_nodes)
        L = to_scipy_sparse_matrix(edge_index, edge_weight).tocsc()
        L = expm(-self.beta * L)
        return torch.from_numpy(L.toarray())


class PStepRWEncoding(PositionEncoding):
    def __init__(self, savepath, p=1, beta=0.5, use_edge_attr=False, normalization=None, zero_diag=False, num_edge_features=4):
        super().__init__(savepath, zero_diag)
        self.p = p
        self.beta = beta
        self.normalization = normalization
        self.use_edge_attr = use_edge_attr
        self.num_edge_features = num_edge_features
        if isinstance(num_edge_features, int):
            self.num_edge_features_true = num_edge_features
        else:
            self.num_edge_features_true = sum(num_edge_features)

    def compute_pe(self, graph):
        if self.use_edge_attr:
            # edge_attr_list = F.one_hot(graph.edge_attr - 1, self.num_edge_features).float()
            edge_attr_list = edge_attr_one_hot(graph.edge_attr, self.num_edge_features)
            pe_tensors = torch.zeros((self.num_edge_features_true, graph.num_nodes, graph.num_nodes))
            for i in range(self.num_edge_features_true):
                pe_i = self.compute_pe_from_edge_weight(
                    graph.edge_index, edge_attr_list[:, i], graph.num_nodes)
                pe_tensors[i] = pe_i
            return pe_tensors
        return self.compute_pe_from_edge_weight(graph.edge_index, None, graph.num_nodes)

    def compute_pe_from_edge_weight(self, edge_index, edge_weight, num_nodes):
        edge_index, edge_weight = get_laplacian(
            edge_index, edge_weight, normalization=self.normalization,
            num_nodes=num_nodes)
        L = to_scipy_sparse_matrix(edge_index, edge_weight).tocsc()
        L = sp.identity(L.shape[0], dtype=L.dtype) - self.beta * L
        tmp = L
        for _ in range(self.p - 1):
            tmp = tmp.dot(L)
        return torch.from_numpy(tmp.toarray())


class AdjEncoding(PositionEncoding):
    def __init__(self, savepath, normalization=None, zero_diag=False):
        """
        normalization: for Laplacian None. sym or rw
        """
        super().__init__(savepath, zero_diag)
        self.normalization = normalization

    def compute_pe(self, graph):
        return to_dense_adj(graph.edge_index)

class FullEncoding(PositionEncoding):
    def __init__(self, savepath, zero_diag=False):
        """
        normalization: for Laplacian None. sym or rw
        """
        super().__init__(savepath, zero_diag)

    def compute_pe(self, graph):
        return torch.ones((graph.num_nodes, graph.num_nodes))

## Absolute position encoding
class LapEncoding(PositionEncoding):
    def __init__(self, dim, use_edge_attr=False, normalization=None):
        """
        normalization: for Laplacian None. sym or rw
        """
        self.pos_enc_dim = dim
        self.normalization = normalization
        self.use_edge_attr = use_edge_attr

    def compute_pe(self, graph):
        edge_attr = graph.edge_attr if self.use_edge_attr else None
        edge_index, edge_attr = get_laplacian(
            graph.edge_index, edge_attr, normalization=self.normalization,
            num_nodes=graph.num_nodes)
        L = to_scipy_sparse_matrix(edge_index, edge_attr).tocsc()
        EigVal, EigVec = np.linalg.eig(L.toarray())
        idx = EigVal.argsort() # increasing order
        EigVal, EigVec = EigVal[idx], np.real(EigVec[:,idx])
        return torch.from_numpy(EigVec[:, 1:self.pos_enc_dim+1]).float()

    def apply_to(self, dataset):
        dataset.lap_pe_list = []
        for i, g in enumerate(dataset):
            pe = self.compute_pe(g)
            dataset.lap_pe_list.append(pe)

        return dataset

def edge_attr_one_hot(edge_attr, num_edge_features):
    """one hot encoding for edge attributes
    edge_attr: num_edges x edge_types
    num_edge_features: int or list
    """
    if isinstance(num_edge_features, int):
        return F.one_hot(edge_attr - 1, num_edge_features).float()
    all_one_hot_feat = []
    for col in range(len(num_edge_features)):
        one_hot_feat = F.one_hot(edge_attr[:, col], num_edge_features[col])
        all_one_hot_feat.append(one_hot_feat)
    all_one_hot_feat = torch.cat(all_one_hot_feat, dim=1)
    return all_one_hot_feat.float()

POSENCODINGS = {
    "diffusion": DiffusionEncoding,
    "pstep": PStepRWEncoding,
    "adj": AdjEncoding,
}
