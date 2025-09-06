from layers import *


class SingleGAT(nn.Module):

    def __init__(self, emb_dim: int, out_dim: int, upscale_dim: int = 16, act=F.relu):
        super().__init__()

        # Graph layers
        self.graph_layers = nn.ModuleList(
            [
                GraphAttention(emb_dim, upscale_dim, act=act),
                GraphAttention(upscale_dim, upscale_dim, act=act),
                GraphAttention(upscale_dim, upscale_dim, act=act),
                # GraphAttention(2 * upscale_dim, 4 * upscale_dim, act=act),
                GlobalAggregator(),
            ]
        )

        # MLP layers
        self.mlp_layers = nn.ModuleList(
            [
                nn.Linear(upscale_dim, 16),
                nn.Linear(16, 16),
                nn.Linear(16, out_dim),
            ]
        )

        self.act = act

    def graph_convolve(self, x: EnrichedGraph, collapse=True):
        ll = self.graph_layers if collapse else self.graph_layers[:-1]
        for layer in ll:
            x = layer(x)
        return x

    def nn_layers(self, x):
        for layer in self.mlp_layers[:-1]:
            x = self.act(layer(x))

        x = self.mlp_layers[-1](x)
        return x

    def forward(self, x):
        y = self.graph_convolve(x)
        y = self.nn_layers(y)
        return y


class GAT(nn.Module):

    def __init__(self, emb_dim: int, out_dim: int, embedding_size: int = 128, n_conv: int = 3, lr_slope=0.01, act=F.gelu):
        super().__init__()

        self.conv0 = GraphAttention(emb_dim, embedding_size, act=act)
        self.convos = nn.ModuleList(
            [GraphAttention(embedding_size, embedding_size, lr_slope=lr_slope, act=act)
             for _ in range(n_conv - 1)])

        self.collapse = GlobalAggregator()

        self.nn1 = nn.Linear(embedding_size, 16)
        self.nn2 = nn.Linear(16, 8)
        self.out = nn.Linear(8, out_dim)

    def graph_convolve(self, x: EnrichedGraph):
        x = self.conv0(x)
        for layer in self.convos:
            x = layer(x)
        return x

    def graph_collapse(self, x: EnrichedGraph):
        return self.collapse(x)

    def collapse_convolve(self, x: EnrichedGraph):
        x = self.conv0(x)
        for layer in self.convos:
            x = layer(x)
        x = self.collapse(x)
        return x

    def nn_layers(self, x):
        x = F.relu(self.nn1(x))
        x = F.relu(self.nn2(x))
        x = self.out(x)

        return x

    def forward(self, x):
        y = self.collapse_convolve(x)
        y = self.nn_layers(y)
        return y

###############################################################################


class JustAggrConvGNN(nn.Module):

    def __init__(self, in_dim, out_dim, upscale_dim=128, act=F.gelu):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.upscale_dim = upscale_dim
        self.act = act

        self.graph_layers = nn.ModuleList([GlobalAggregator()])

        self.mlp_layers = nn.ModuleList([
            nn.Linear(in_dim, 64),
            nn.GELU(),
            nn.Linear(64, out_dim),
        ]
        )

    def graph_conv(self, x: EnrichedGraph, stop_at=None):
        if stop_at is None:
            layer_list = self.graph_layers
        else:
            if abs(stop_at) > len(self.graph_layers):
                raise ValueError("Graph layers are out of range.")
            layer_list = self.graph_layers[:stop_at]

        for layer in layer_list:
            x = layer(x)
        return x

    def nn_layers(self, x):
        for layer in self.mlp_layers:
            x = layer(x)
        return x

    def forward(self, x):
        y = self.graph_conv(x)
        y = self.nn_layers(y)
        return y


# One convolution
class OneConvGCNN(nn.Module):

    def __init__(self, in_dim, out_dim, upscale_dim=128, act=F.gelu):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.upscale_dim = upscale_dim
        self.act = act

        self.graph_layers = nn.ModuleList(
            [
                GraphConv(in_dim, upscale_dim),
                GlobalAggregator()
            ]
        )

        self.mlp_layers = nn.ModuleList(
            [
                nn.Linear(upscale_dim, 64),
                nn.GELU(),
                nn.Linear(64, out_dim),
            ]
        )

    def graph_conv(self, x: EnrichedGraph, stop_at=None):
        if stop_at is None:
            layer_list = self.graph_layers
        else:
            if abs(stop_at) > len(self.graph_layers):
                raise ValueError("Graph layers are out of range.")
            layer_list = self.graph_layers[:stop_at]

        for layer in layer_list:
            x = layer(x)
        return x

    def nn_layers(self, x):
        for layer in self.mlp_layers:
            x = layer(x)
        return x

    def forward(self, x):
        y = self.graph_conv(x)
        y = self.nn_layers(y)
        return y


class TwoConvGCNN(nn.Module):

    def __init__(self, in_dim, out_dim, upscale_dim=128, act=F.gelu):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.upscale_dim = upscale_dim
        self.act = act

        self.graph_layers = nn.ModuleList(
            [
                GraphConv(in_dim, upscale_dim),
                GraphConv(upscale_dim, upscale_dim),
                GlobalAggregator()
            ]
        )

        self.mlp_layers = nn.ModuleList(
            [
                nn.Linear(upscale_dim, 64),
                nn.GELU(),
                nn.Linear(64, out_dim),
            ]
        )

    def graph_conv(self, x: EnrichedGraph, stop_at=None):
        if stop_at is None:
            layer_list = self.graph_layers
        else:
            if abs(stop_at) > len(self.graph_layers):
                raise ValueError("Graph layers are out of range.")
            layer_list = self.graph_layers[:stop_at]

        for layer in layer_list:
            x = layer(x)
        return x

    def nn_layers(self, x):
        for layer in self.mlp_layers:
            x = layer(x)
        return x

    def forward(self, x):
        y = self.graph_conv(x)
        y = self.nn_layers(y)
        return y


class OneConvGAT(nn.Module):

    def __init__(self, in_dim: int, out_dim: int,
                 upscale_dim: int = 128, n_head: int = 3, act=F.gelu):

        super().__init__()

        self.graph_layers = nn.ModuleList(
            [
                GraphMultiHeadAttention(
                    in_dim, upscale_dim, n_heads=n_head,
                    act=act),
                GraphMultiHeadAttention(
                    upscale_dim * n_head, upscale_dim, n_heads=n_head,
                    act=act,),
                GraphMultiHeadAttention(
                    upscale_dim * n_head, upscale_dim, n_heads=n_head,
                    act=act,),
                GlobalAggregator(),
            ]
        )

        self.mlp_layers = nn.ModuleList(
            [
                nn.Linear(upscale_dim * n_head, upscale_dim // 2),
                nn.GELU(),
                nn.Linear(upscale_dim//2, upscale_dim//2),
                nn.GELU(),
                nn.Linear(upscale_dim//2, out_dim),
            ]
        )

        self.act = act

    def graph_conv(self, x: EnrichedGraph, stop_at=None):

        if stop_at is None:
            layer_list = self.graph_layers
        else:
            if abs(stop_at) > len(self.graph_layers):
                raise ValueError("Graph layers are out of range.")
            layer_list = self.graph_layers[:stop_at]

        for layer in layer_list:
            x = layer(x)
        return x

    def nn_layers(self, x):
        for layer in self.mlp_layers:
            x = layer(x)
        return x

    def forward(self, x):
        y = self.graph_conv(x)
        y = self.nn_layers(y)
        return y


class MultipleGAT(nn.Module):

    def __init__(self, in_dim: int, out_dim: int,
                 upscale_dim: int = 21, n_head: int = 3, act=F.gelu):

        super().__init__()

        self.graph_layers = nn.ModuleList(
            [
                GraphMultiHeadAttention(
                    in_dim, upscale_dim, n_heads=n_head,
                    act=act),
                GraphMultiHeadAttention(
                    upscale_dim * n_head, upscale_dim, n_heads=n_head,
                    act=act,),
                GraphMultiHeadAttention(
                    upscale_dim * n_head, upscale_dim, n_heads=n_head,
                    act=act,),
                GlobalAggregator(),
            ]
        )

        self.mlp_layers = nn.ModuleList(
            [
                nn.Linear(upscale_dim * n_head, upscale_dim // 2),
                nn.GELU(),
                nn.Linear(upscale_dim//2, upscale_dim//2),
                nn.GELU(),
                nn.Linear(upscale_dim//2, out_dim),
            ]
        )

        self.act = act

    def graph_conv(self, x: EnrichedGraph, stop_at=None):

        if stop_at is None:
            layer_list = self.graph_layers
        else:
            if abs(stop_at) > len(self.graph_layers):
                raise ValueError("Graph layers are out of range.")
            layer_list = self.graph_layers[:stop_at]

        for layer in layer_list:
            x = layer(x)
        return x

    def nn_layers(self, x):
        for layer in self.mlp_layers:
            x = layer(x)
        return x

    def forward(self, x):
        y = self.graph_conv(x)
        y = self.nn_layers(y)
        return y
