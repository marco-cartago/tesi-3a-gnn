"""This module implements most of the layer used in this work"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from dataclasses import dataclass


@dataclass
class EnrichedGraph:

    """Implements a single object that keeps track of the embeddings and the
    graph structure. Provides support for doing training with batches."""

    adj_mat: torch.Tensor  # The adjacency matrix of the graph
    emb_mat: torch.Tensor  # Each row should be an embedding
    corrisp: torch.Tensor  # The membership of each embedding to its respective graph

    def to(self, device):
        return EnrichedGraph(
            adj_mat=self.adj_mat.to(device),
            emb_mat=self.emb_mat.to(device),
            corrisp=self.corrisp.to(device),
        )

    @property
    def device(self):
        return self.adj_mat.device

    def __add__(self, other):

        if self.adj_mat.shape != other.adj_mat.shape:
            raise ValueError(
                f"Adjacency matrix dimensions must match: {self.adj_mat.shape} vs {other.adj_mat.shape}"
            )
        if self.emb_mat.shape != other.emb_mat.shape:
            raise ValueError(
                f"Embedding matrix dimensions must match: {self.emb_mat.shape} vs {other.emb_mat.shape}"
            )

        return EnrichedGraph(
            adj_mat=self.adj_mat,
            emb_mat=self.emb_mat + other.emb_mat,
            corrisp=self.corrisp,
        )


class GraphConv(nn.Module):

    """Apply a graph convolution, returns an updated graph. 
    The node aggregation is done via average"""

    def __init__(self, in_emb_dim: int, out_emb_dim: int,
                 act=torch.relu
                 ) -> None:

        super().__init__()
        # Setting layer costants
        self.in_emb_dim = in_emb_dim
        self.out_emb_dim = out_emb_dim
        self.act = act

        # Parameter iniziatization
        self.linear = nn.Linear(in_emb_dim, out_emb_dim, bias=False)
        self.bias = nn.Parameter(torch.empty(1, out_emb_dim))
        nn.init.xavier_uniform_(self.bias)

    def forward(self, graph: EnrichedGraph) -> EnrichedGraph:
        emb = graph.emb_mat

        # Message passing: sum aggregation
        aggregated = torch.matmul(graph.adj_mat, emb)

        # Apply average
        neighbours = torch.sum(graph.adj_mat, dim=1).view(-1, 1)
        averaged_aggregation = aggregated * neighbours

        # Transform and apply activation
        transformed = self.linear(averaged_aggregation)
        output = self.act(transformed + self.bias)

        # Create (and return) new graph with updated embeddings
        res = EnrichedGraph(
            adj_mat=graph.adj_mat,
            emb_mat=output,
            corrisp=graph.corrisp,
        )
        return res


class GraphAttention(nn.Module):

    """Given an input `EnrichedGraph` returns an updated copy of the embeddings 
    via message passing. Returns the new embeddings"""

    def __init__(self, in_emb_dim, out_emb_dim,
                 act=torch.relu, lr_slope: float = 0.01
                 ) -> None:

        super().__init__()
        # Layer constants
        self.in_emb_dim = in_emb_dim
        self.out_emb_dim = out_emb_dim
        self.lr_slope = lr_slope
        self.act = act

        # Parameters
        self.W = nn.Parameter(torch.empty(size=(out_emb_dim, in_emb_dim)))
        self.a = nn.Parameter(torch.empty(size=(2*out_emb_dim, 1)))

        # Layer initialization
        nn.init.xavier_normal_(self.W)
        nn.init.xavier_normal_(self.a)

    def forward(self, g: EnrichedGraph) -> torch.Tensor:

        # Transform embedding
        e_new = torch.matmul(g.emb_mat, self.W.T)

        # Split attention vector
        a1 = self.a[:self.out_emb_dim, :]
        a2 = self.a[self.out_emb_dim:, :]
        source_scores = torch.matmul(e_new, a1)
        target_scores = torch.matmul(e_new, a2)
        e = source_scores + target_scores.T

        # Apply activation and connectivity mask
        e = F.leaky_relu(e, negative_slope=self.lr_slope)
        connectivity_mask = -9e16 * torch.ones_like(e)
        e = torch.where(g.adj_mat > 0, e, connectivity_mask)

        # Normalize attention weights
        attention = F.softmax(e, dim=-1)

        # Apply activation and update
        e_transf = torch.matmul(attention, e_new)
        emb_mat = self.act(e_transf)

        return emb_mat


class GraphMultiHeadAttention(nn.Module):

    """Concatenates the output of single `GraphAttention` modules in order
    to get multi-head graph attention."""

    def __init__(self, in_emb_dim: int, out_head_size: int,
                 n_heads: int = 4, act=torch.relu, lr_slope: float = 0.01
                 ) -> None:

        super().__init__()
        self.in_emb_dim = in_emb_dim
        self.out_head_size = out_head_size
        self.lr_slope = lr_slope
        self.act = act
        self.n_heads = n_heads

        # Initalizing and storing attention heads
        self.attention_heads = nn.ModuleList([
            GraphAttention(self.in_emb_dim, self.out_head_size)
            for _ in range(n_heads)
        ]
        )

    def forward(self, g: EnrichedGraph) -> EnrichedGraph:
        head_outputs = [head(g) for head in self.attention_heads]
        g.emb_mat = torch.cat(head_outputs, dim=-1)
        return g


class EmbddingTransform(nn.Module):

    """Applys to all embeddings the same set of operations"""

    def __init__(self, module: nn.Module) -> None:
        super().__init__()
        self.module = module

    def forward(self, x: EnrichedGraph) -> EnrichedGraph:
        # n_obs, e_dim = x.emb_mat.shape
        emb_new = self.module(x.emb_mat)
        new_graph = EnrichedGraph(
            adj_mat=x.adj_mat, emb_mat=emb_new, corrisp=x.corrisp)

        return new_graph


class GlobalAggregator(nn.Module):

    """Collapses via a an average all the node embeddings.
    This produces an embedding of the graph."""

    def __init__(self):
        super().__init__()

    def forward(self, g: EnrichedGraph) -> torch.Tensor:

        device = g.device
        emb_mat = g.emb_mat
        corr = g.corrisp.to(device)

        # :)
        corr_idx = corr.long()

        num_groups = int(corr_idx.max().item()) + 1
        emb_dim = emb_mat.size(1)

        # Create the result tensor
        sum_per_group = torch.zeros(num_groups, emb_dim, device=device)

        # expand corr_idx to match the embedding dimension for scattering
        idx_expanded = corr_idx.unsqueeze(1).expand(-1, emb_dim)
        sum_per_group = sum_per_group.scatter_add_(0, idx_expanded, emb_mat)

        # Count nodes per group and normalize
        counts = torch.bincount(
            corr_idx, minlength=num_groups).unsqueeze(1).clamp(min=1)

        result = sum_per_group / counts

        return result
