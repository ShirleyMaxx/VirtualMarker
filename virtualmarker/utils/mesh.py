"""
Code source: 
    https://github.com/nkolot/GraphCMR/blob/master/models/graph_layers.py
    https://github.com/nkolot/GraphCMR/blob/master/utils/mesh.py
Definition of SparseMM, Mesh used for downsample/upsample mesh
"""
from __future__ import division
import numpy as np
import scipy.sparse

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class SparseMM(torch.autograd.Function):
    """Redefine sparse @ dense matrix multiplication to enable backpropagation.
    The builtin matrix multiplication operation does not support backpropagation in some cases.
    """
    @staticmethod
    def forward(ctx, sparse, dense):
        ctx.req_grad = dense.requires_grad
        ctx.save_for_backward(sparse)
        return torch.matmul(sparse, dense)

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = None
        sparse, = ctx.saved_tensors
        if ctx.req_grad:
            grad_input = torch.matmul(sparse.t(), grad_output)
        return None, grad_input

def spmm(sparse, dense):
    return SparseMM.apply(sparse.to(dense.device), dense)

def scipy_to_pytorch(A, U, D):
    """Convert scipy sparse matrices to pytorch sparse matrix."""
    ptU = []
    ptD = []
    
    for i in range(len(U)):
        u = scipy.sparse.coo_matrix(U[i])
        i = torch.LongTensor(np.array([u.row, u.col]))
        v = torch.FloatTensor(u.data)
        ptU.append(torch.sparse.FloatTensor(i, v, u.shape))
    
    for i in range(len(D)):
        d = scipy.sparse.coo_matrix(D[i])
        i = torch.LongTensor(np.array([d.row, d.col]))
        v = torch.FloatTensor(d.data)
        ptD.append(torch.sparse.FloatTensor(i, v, d.shape)) 

    return ptU, ptD


def adjmat_sparse(adjmat, nsize=1):
    """Create row-normalized sparse graph adjacency matrix."""
    adjmat = scipy.sparse.csr_matrix(adjmat)
    if nsize > 1:
        orig_adjmat = adjmat.copy()
        for _ in range(1, nsize):
            adjmat = adjmat * orig_adjmat
    adjmat.data = np.ones_like(adjmat.data)
    for i in range(adjmat.shape[0]):
        adjmat[i,i] = 1
    num_neighbors = np.array(1 / adjmat.sum(axis=-1))
    adjmat = adjmat.multiply(num_neighbors)
    adjmat = scipy.sparse.coo_matrix(adjmat)
    row = adjmat.row
    col = adjmat.col
    data = adjmat.data
    i = torch.LongTensor(np.array([row, col]))
    v = torch.from_numpy(data).float()
    adjmat = torch.sparse.FloatTensor(i, v, adjmat.shape)
    return adjmat

def get_graph_params(filename, nsize=1):
    """Load and process graph adjacency matrix and upsampling/downsampling matrices."""
    data = np.load(filename, encoding='latin1', allow_pickle=True)
    A = data['A']
    U = data['U']
    D = data['D']
    U, D = scipy_to_pytorch(A, U, D)
    A = [adjmat_sparse(a, nsize=nsize) for a in A]
    return A, U, D


class Mesh(object):
    """Mesh object that is used for handling certain graph operations."""
    def __init__(self, mesh_template, filename='data/smpl/mesh_downsampling.npz',
                 num_downsampling=1, nsize=1):
        self._A, self._U, self._D = get_graph_params(filename=filename, nsize=nsize)
        self._A = [a.cuda() for a in self._A]
        self._U = [u.cuda() for u in self._U]
        self._D = [d.cuda() for d in self._D]
        self.num_downsampling = num_downsampling

        # normalize template vertices
        ref_vertices = mesh_template[0].clone()
        center = 0.5*(ref_vertices.max(dim=0)[0] + ref_vertices.min(dim=0)[0])[None]
        ref_vertices -= center
        ref_vertices /= ref_vertices.abs().max().item()

        self._ref_vertices = ref_vertices.cuda()

    @property
    def adjmat(self):
        """Return the graph adjacency matrix at the specified subsampling level."""
        return self._A[self.num_downsampling].float()

    @property
    def ref_vertices(self):
        """Return the template vertices at the specified subsampling level."""
        ref_vertices = self._ref_vertices
        for i in range(self.num_downsampling):
            ref_vertices = torch.spmm(self._D[i], ref_vertices)
        return ref_vertices

    def downsample(self, x, n1=0, n2=None):
        """Downsample mesh."""
        if n2 is None:
            n2 = self.num_downsampling
        if x.ndimension() < 3:
            for i in range(n1, n2):
                x = spmm(self._D[i], x)
        elif x.ndimension() == 3:
            out = []
            for i in range(x.shape[0]):
                y = x[i]
                for j in range(n1, n2):
                    y = spmm(self._D[j], y)
                out.append(y)
            x = torch.stack(out, dim=0)
        return x

    def upsample(self, x, n1=1, n2=0):
        """Upsample mesh."""
        if x.ndimension() < 3:
            for i in reversed(range(n2, n1)):
                x = spmm(self._U[i], x)
        elif x.ndimension() == 3:
            out = []
            for i in range(x.shape[0]):
                y = x[i]
                for j in reversed(range(n2, n1)):
                    y = spmm(self._U[j], y)
                out.append(y)
            x = torch.stack(out, dim=0)
        return x