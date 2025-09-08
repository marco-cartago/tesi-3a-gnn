from layers import *


class JustAggrConvGCNN(nn.Module):

    """Agg()"""

    def __init__(self, in_dim, out_dim, upscale_dim=128, act=F.gelu):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.upscale_dim = upscale_dim
        self.act = act

        self.graph_layers = nn.ModuleList([GlobalAggregator()])

        self.mlp_layers = nn.ModuleList(
            [
                nn.Linear(in_dim, 64),
                nn.GELU(),
                nn.Linear(64, 32),
                nn.GELU(),
                nn.Linear(32, out_dim),
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

    """Conv()+Agg()"""

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
                nn.Linear(64, 32),
                nn.GELU(),
                nn.Linear(32, out_dim),
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

    """Conv()+Conv()+Agg()"""

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
                nn.Linear(64, 32),
                nn.GELU(),
                nn.Linear(32, out_dim),
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


class OneLayerOneHeadGAT(nn.Module):

    """1xAttConv()+Agg()"""

    def __init__(self, in_dim: int, out_dim: int,
                 upscale_dim: int = 128, act=F.gelu):

        super().__init__()

        self.graph_layers = nn.ModuleList(
            [
                GraphMultiHeadAttention(
                    in_dim, upscale_dim, n_heads=1,
                    act=act),
                GlobalAggregator(),
            ]
        )

        self.mlp_layers = nn.ModuleList(
            [
                nn.Linear(upscale_dim, 64),
                nn.GELU(),
                nn.Linear(64, 32),
                nn.GELU(),
                nn.Linear(32, out_dim),
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


class MultiHeadGAT(nn.Module):

    """2 x AttConv() + Agg"""

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
                GlobalAggregator(),
            ]
        )

        self.mlp_layers = nn.ModuleList(
            [
                nn.Linear(upscale_dim * n_head, 64),
                nn.GELU(),
                nn.Linear(64, 32),
                nn.GELU(),
                nn.Linear(32, out_dim),
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
