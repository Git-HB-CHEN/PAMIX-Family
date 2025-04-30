import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Any
from einops import repeat, rearrange
from functools import partial
from collections import OrderedDict
from timm.models.layers import DropPath, trunc_normal_

DropPath.__repr__ = lambda self: f"timm.DropPath({self.drop_prob})"

try:
    from .csm_triton import CrossScanTriton, CrossMergeTriton
    from .csms6s import CrossScan, CrossMerge, SelectiveScanOflex
    from .router_network import AggregationRouter
    from .PMEvolution import PrototypeMemoryEvolution
except:
    from csm_triton import CrossScanTriton, CrossMergeTriton
    from csms6s import CrossScan, CrossMerge, SelectiveScanOflex
    from router_network import AggregationRouter
    from PMEvolution import PrototypeMemoryEvolution

try:
    from mamba_ssm.ops.selective_scan_interface import selective_scan_fn, selective_scan_ref
except:
    pass


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.,channels_first=False):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        Linear = Linear2d if channels_first else nn.Linear
        self.fc1 = Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class gMlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.,channels_first=False):
        super().__init__()
        self.channel_first = channels_first
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        Linear = Linear2d if channels_first else nn.Linear
        self.fc1 = Linear(in_features, 2 * hidden_features)
        self.act = act_layer()
        self.fc2 = Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x: torch.Tensor):
        x = self.fc1(x)
        x, z = x.chunk(2, dim=(1 if self.channel_first else -1))
        x = self.fc2(x * self.act(z))
        x = self.drop(x)
        return x


class Linear2d(nn.Linear):
    def forward(self, x: torch.Tensor):
        # B, C, H, W = x.shape
        return F.conv2d(x, self.weight[:, :, None, None], self.bias)


class LayerNorm2d(nn.LayerNorm):
    def forward(self, x: torch.Tensor):
        x = x.permute(0, 2, 3, 1)
        x = nn.functional.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        x = x.permute(0, 3, 1, 2)
        return x


class Permute(nn.Module):
    def __init__(self, *args):
        super().__init__()
        self.args = args

    def forward(self, x: torch.Tensor):
        return x.permute(*self.args)


class SoftmaxSpatial(nn.Softmax):
    def forward(self, x: torch.Tensor):
        if self.dim == -1:
            B, C, H, W = x.shape
            return super().forward(x.view(B, C, -1)).view(B, C, H, W)
        elif self.dim == 1:
            B, H, W, C = x.shape
            return super().forward(x.view(B, -1, C)).view(B, H, W, C)
        else:
            raise NotImplementedError


# =====================================================
class mamba_init:
    @staticmethod
    def dt_init(dt_rank, d_inner, dt_scale=1.0, dt_init="random", dt_min=0.001, dt_max=0.1, dt_init_floor=1e-4):
        dt_proj = nn.Linear(dt_rank, d_inner, bias=True)

        dt_init_std = dt_rank ** -0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        dt = torch.exp(
            torch.rand(d_inner) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)

        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            dt_proj.bias.copy_(inv_dt)

        return dt_proj

    @staticmethod
    def A_log_init(d_state, d_inner, copies=-1, device=None, merge=True):

        A = repeat(
            torch.arange(1, d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=d_inner,
        ).contiguous()
        A_log = torch.log(A)  # Keep A_log in fp32
        if copies > 0:
            A_log = repeat(A_log, "d n -> r d n", r=copies)
            if merge:
                A_log = A_log.flatten(0, 1)
        A_log = nn.Parameter(A_log)
        A_log._no_weight_decay = True
        return A_log

    @staticmethod
    def D_init(d_inner, copies=-1, device=None, merge=True):
        # D "skip" parameter
        D = torch.ones(d_inner, device=device)
        if copies > 0:
            D = repeat(D, "n1 -> r n1", r=copies)
            if merge:
                D = D.flatten(0, 1)
        D = nn.Parameter(D)  # Keep in fp32
        D._no_weight_decay = True
        return D

class SS2Dv2:
    def __initv2__(
            self,
            d_model=96,
            d_state=16,
            ssm_ratio=2.0,
            dt_rank="auto",
            act_layer=nn.SiLU,
            d_conv=3,  # < 2 means no conv
            conv_bias=True,
            dropout=0.0,
            bias=False,
            dt_min=0.001,
            dt_max=0.1,
            dt_init="random",
            dt_scale=1.0,
            dt_init_floor=1e-4,
            initialize="v0",
            forward_type="v2",
            channel_first=False,
            **kwargs,
    ):
        factory_kwargs = {"device": None, "dtype": None}
        super().__init__()
        d_inner = int(ssm_ratio * d_model)
        dt_rank = math.ceil(d_model / 16) if dt_rank == "auto" else dt_rank
        self.channel_first = channel_first
        self.with_dconv = d_conv > 1
        Linear = Linear2d if channel_first else nn.Linear
        LayerNorm = LayerNorm2d if channel_first else nn.LayerNorm
        self.forward = self.forwardv2

        def checkpostfix(tag, value):
            ret = value[-len(tag):] == tag
            if ret:
                value = value[:-len(tag)]
            return ret, value

        self.disable_z, forward_type = checkpostfix("_noz", forward_type)
        self.out_norm = LayerNorm(d_inner)
        self.forward_core = partial(self.forward_corev2, force_fp32=False, SelectiveScan=SelectiveScanOflex,
                                    no_einsum=True, CrossScan=CrossScanTriton, CrossMerge=CrossMergeTriton)
        k_group = 4

        # in proj =======================================
        d_proj = d_inner if self.disable_z else (d_inner * 2)
        self.in_proj = Linear(d_model, d_proj, bias=bias)
        self.act: nn.Module = act_layer()

        # conv =======================================
        if self.with_dconv:
            self.conv2d = nn.Conv2d(
                in_channels=d_inner,
                out_channels=d_inner,
                groups=d_inner,
                bias=conv_bias,
                kernel_size=d_conv,
                padding=(d_conv - 1) // 2,
                **factory_kwargs,
            )

        # x proj ============================
        self.x_proj = [
            nn.Linear(d_inner, (dt_rank + d_state * 2), bias=False)
            for _ in range(k_group)
        ]
        self.x_proj_weight = nn.Parameter(torch.stack([t.weight for t in self.x_proj], dim=0))  # (K, N, inner)
        del self.x_proj

        # out proj =======================================
        self.out_act = nn.Identity()
        self.out_proj = Linear(d_inner, d_model, bias=bias)
        self.dropout = nn.Dropout(dropout) if dropout > 0. else nn.Identity()

        # dt proj ============================
        self.dt_projs = [
            self.dt_init(dt_rank, d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor)
            for _ in range(k_group)
        ]
        self.dt_projs_weight = nn.Parameter(torch.stack([t.weight for t in self.dt_projs], dim=0))  # (K, inner, rank)
        self.dt_projs_bias = nn.Parameter(torch.stack([t.bias for t in self.dt_projs], dim=0))  # (K, inner)
        del self.dt_projs

        # A, D =======================================
        self.A_logs = self.A_log_init(d_state, d_inner, copies=k_group, merge=True)  # (K * D, N)
        self.Ds = self.D_init(d_inner, copies=k_group, merge=True)  # (K * D)

    def forward_corev2(
            self,
            x: torch.Tensor = None,
            # ==============================
            to_dtype=True,  # True: final out to dtype
            force_fp32=False,  # True: input fp32
            # ==============================
            CrossScan=CrossScan,
            CrossMerge=CrossMerge,
            **kwargs,
    ):
        x_proj_weight = self.x_proj_weight
        x_proj_bias = getattr(self, "x_proj_bias", None)
        dt_projs_weight = self.dt_projs_weight
        dt_projs_bias = self.dt_projs_bias
        A_logs = self.A_logs
        Ds = self.Ds
        delta_softplus = True
        out_norm = getattr(self, "out_norm", None)
        channel_first = self.channel_first
        to_fp32 = lambda *args: (_a.to(torch.float32) for _a in args)

        B, D, H, W = x.shape
        D, N = A_logs.shape
        K, D, R = dt_projs_weight.shape
        L = H * W

        self.selective_scan = selective_scan_fn

        xs = CrossScan.apply(x)

        x_dbl = F.conv1d(xs.view(B, -1, L), x_proj_weight.view(-1, D, 1),
                         bias=(x_proj_bias.view(-1) if x_proj_bias is not None else None), groups=K)
        dts, Bs, Cs = torch.split(x_dbl.view(B, K, -1, L), [R, N, N], dim=2)
        dts = F.conv1d(dts.contiguous().view(B, -1, L), dt_projs_weight.view(K * D, -1, 1), groups=K)

        xs = xs.view(B, -1, L)
        dts = dts.contiguous().view(B, -1, L)
        As = -torch.exp(A_logs.to(torch.float))  # (k * c, d_state)
        Bs = Bs.contiguous().view(B, K, N, L)
        Cs = Cs.contiguous().view(B, K, N, L)
        Ds = Ds.to(torch.float)  # (K * c)
        delta_bias = dt_projs_bias.view(-1).to(torch.float)

        if force_fp32:
            xs, dts, Bs, Cs = to_fp32(xs, dts, Bs, Cs)

        ys: torch.Tensor = self.selective_scan(
            xs, dts, As, Bs, Cs, Ds, None, delta_bias, delta_softplus
        ).view(B, K, -1, H, W)

        y: torch.Tensor = CrossMerge.apply(ys)

        if getattr(self, "__DEBUG__", False):
            setattr(self, "__data__", dict(
                A_logs=A_logs, Bs=Bs, Cs=Cs, Ds=Ds,
                us=xs, dts=dts, delta_bias=delta_bias,
                ys=ys, y=y, H=H, W=W,
            ))

        y = y.view(B, -1, H, W)
        if not channel_first:
            y = y.view(B, -1, H * W).transpose(dim0=1, dim1=2).contiguous().view(B, H, W, -1)  # (B, L, C)
        y = out_norm(y)

        return (y.to(x.dtype) if to_dtype else y)

    def forwardv2(self, x: torch.Tensor, **kwargs):
        x = self.in_proj(x)
        if not self.disable_z:
            x, z = x.chunk(2, dim=(1 if self.channel_first else -1))  # (b, h, w, d)
            if not self.disable_z_act:
                z = self.act(z)
        if not self.channel_first:
            x = x.permute(0, 3, 1, 2).contiguous()
        if self.with_dconv:
            x = self.conv2d(x)  # (b, d, h, w)
        x = self.act(x)
        y = self.forward_core(x)
        y = self.out_act(y)
        if not self.disable_z:
            y = y * z
        out = self.dropout(self.out_proj(y))
        return out


class SS2D(nn.Module, mamba_init, SS2Dv2):
    def __init__(
            self,
            # basic dims ===========
            d_model=96,
            d_state=16,
            ssm_ratio=2.0,
            dt_rank="auto",
            act_layer=nn.SiLU,
            # dwconv ===============
            d_conv=3,  # < 2 means no conv
            conv_bias=True,
            # ======================
            dropout=0.0,
            bias=False,
            # dt init ==============
            dt_min=0.001,
            dt_max=0.1,
            dt_init="random",
            dt_scale=1.0,
            dt_init_floor=1e-4,
            initialize="v0",
            # ======================
            forward_type="v2",
            channel_first=False,
            # ======================
            **kwargs,
    ):
        super().__init__()
        kwargs.update(
            d_model=d_model, d_state=d_state, ssm_ratio=ssm_ratio, dt_rank=dt_rank,
            act_layer=act_layer, d_conv=d_conv, conv_bias=conv_bias, dropout=dropout, bias=bias,
            dt_min=dt_min, dt_max=dt_max, dt_init=dt_init, dt_scale=dt_scale, dt_init_floor=dt_init_floor,
            initialize=initialize, forward_type=forward_type, channel_first=channel_first,
        )
        self.__initv2__(**kwargs)


# =====================================================
class VSSBlock(nn.Module):
    def __init__(
            self,
            hidden_dim: int = 0,
            drop_path: float = 0,
            norm_layer: nn.Module = nn.LayerNorm,
            channel_first=False,
            # =============================
            ssm_d_state: int = 16,
            ssm_ratio=2.0,
            ssm_dt_rank: Any = "auto",
            ssm_act_layer=nn.SiLU,
            ssm_conv: int = 3,
            ssm_conv_bias=True,
            ssm_drop_rate: float = 0,
            ssm_init="v0",
            forward_type="v2",
            # =============================
            mlp_ratio=4.0,
            mlp_act_layer=nn.GELU,
            mlp_drop_rate: float = 0.0,
            gmlp=False,
            **kwargs,
    ):
        super().__init__()

        self.norm = norm_layer(hidden_dim)
        self.op = SS2D(
            d_model=hidden_dim,
            d_state=ssm_d_state,
            ssm_ratio=ssm_ratio,
            dt_rank=ssm_dt_rank,
            act_layer=ssm_act_layer,
            d_conv=ssm_conv,
            conv_bias=ssm_conv_bias,
            dropout=ssm_drop_rate,
            initialize=ssm_init,
            forward_type=forward_type,
            channel_first=channel_first,
        )

        self.drop_path = DropPath(drop_path)

        _MLP = Mlp if not gmlp else gMlp
        self.norm2 = norm_layer(hidden_dim)
        mlp_hidden_dim = int(hidden_dim * mlp_ratio)
        self.mlp = _MLP(in_features=hidden_dim, hidden_features=mlp_hidden_dim, act_layer=mlp_act_layer,
                        drop=mlp_drop_rate, channels_first=channel_first)

    def _forward(self, input: torch.Tensor):
            x = input
            x = x + self.drop_path(self.op(self.norm(x)))
            x = x + self.drop_path(self.mlp(self.norm2(x)))  # FFN
            return x

    def forward(self, input: torch.Tensor):
        return self._forward(input)


class _shared_extractor(nn.Module):
    def __init__(
            self,
            in_chans = 3,
            patch_size = 4,
            depths=[2, 2, 8, 2],
            dims=[96, 192, 384, 768],
            # =========================
            ssm_d_state=1,
            ssm_ratio=1.0,
            ssm_dt_rank="auto",
            ssm_conv=3,
            ssm_conv_bias=False,
            ssm_drop_rate=0.0,
            ssm_init="v0",
            forward_type="v05_noz",
            # =========================
            mlp_ratio=4.0,
            mlp_drop_rate=0.0,
            gmlp=False,
            # =========================
            drop_path_rate=0.2,
            patch_norm=True,
            use_checkpoint=False,
            # =========================
            **kwargs,
    ):
        super().__init__()
        self.channel_first = True
        self.num_layers = len(depths) // 2
        self.num_features = dims[-1]
        self.dims = dims
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule

        norm_layer = LayerNorm2d
        ssm_act_layer = nn.SiLU
        mlp_act_layer = nn.GELU

        self.patch_embed = self._make_patch_embed_v2(in_chans, dims[0], patch_size, patch_norm, norm_layer,
                                                     channel_first=self.channel_first)

        _make_downsample = self._make_downsample_v3

        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            downsample = _make_downsample(
                self.dims[i_layer],
                self.dims[i_layer + 1],
                norm_layer=norm_layer,
                channel_first=self.channel_first,
            ) if (i_layer < self.num_layers - 1) else nn.Identity()

            self.layers.append(self._make_layer(
                dim=self.dims[i_layer],
                drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                use_checkpoint=use_checkpoint,
                norm_layer=norm_layer,
                downsample=downsample,
                channel_first=self.channel_first,
                # =================
                ssm_d_state=ssm_d_state,
                ssm_ratio=ssm_ratio,
                ssm_dt_rank=ssm_dt_rank,
                ssm_act_layer=ssm_act_layer,
                ssm_conv=ssm_conv,
                ssm_conv_bias=ssm_conv_bias,
                ssm_drop_rate=ssm_drop_rate,
                ssm_init=ssm_init,
                forward_type=forward_type,
                # =================
                mlp_ratio=mlp_ratio,
                mlp_act_layer=mlp_act_layer,
                mlp_drop_rate=mlp_drop_rate,
                gmlp=gmlp,
            ))

        self.apply(self._init_weights)

    def _init_weights(self, m: nn.Module):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    # used in building optimizer
    @torch.jit.ignore
    def no_weight_decay(self):
        return {"pos_embed"}

    # used in building optimizer
    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {}

    @staticmethod
    def _make_patch_embed_v2(in_chans=3, embed_dim=96, patch_size=4, patch_norm=True, norm_layer=nn.LayerNorm,
                             channel_first=False):
        # if channel first, then Norm and Output are both channel_first
        stride = patch_size // 2
        kernel_size = stride + 1
        padding = 1
        return nn.Sequential(
            nn.Conv2d(in_chans, embed_dim // 2, kernel_size=kernel_size, stride=stride, padding=padding),
            (nn.Identity() if (channel_first or (not patch_norm)) else Permute(0, 2, 3, 1)),
            (norm_layer(embed_dim // 2) if patch_norm else nn.Identity()),
            (nn.Identity() if (channel_first or (not patch_norm)) else Permute(0, 3, 1, 2)),
            nn.GELU(),
            nn.Conv2d(embed_dim // 2, embed_dim, kernel_size=kernel_size, stride=stride, padding=padding),
            (nn.Identity() if channel_first else Permute(0, 2, 3, 1)),
            (norm_layer(embed_dim) if patch_norm else nn.Identity()),
        )

    @staticmethod
    def _make_downsample_v3(dim=96, out_dim=192, norm_layer=nn.LayerNorm, channel_first=False):
        # if channel first, then Norm and Output are both channel_first
        return nn.Sequential(
            (nn.Identity() if channel_first else Permute(0, 3, 1, 2)),
            nn.Conv2d(dim, out_dim, kernel_size=3, stride=2, padding=1),
            (nn.Identity() if channel_first else Permute(0, 2, 3, 1)),
            norm_layer(out_dim),
        )

    @staticmethod
    def _make_layer(
            dim=96,
            drop_path=[0.1, 0.1],
            use_checkpoint=False,
            norm_layer=nn.LayerNorm,
            downsample=nn.Identity(),
            channel_first=False,
            # ===========================
            ssm_d_state=16,
            ssm_ratio=2.0,
            ssm_dt_rank="auto",
            ssm_act_layer=nn.SiLU,
            ssm_conv=3,
            ssm_conv_bias=True,
            ssm_drop_rate=0.0,
            ssm_init="v0",
            forward_type="v2",
            # ===========================
            mlp_ratio=4.0,
            mlp_act_layer=nn.GELU,
            mlp_drop_rate=0.0,
            gmlp=False,
            **kwargs,
    ):
        # if channel first, then Norm and Output are both channel_first
        depth = len(drop_path)
        blocks = []
        for d in range(depth):
            blocks.append(VSSBlock(
                hidden_dim=dim,
                drop_path=drop_path[d],
                norm_layer=norm_layer,
                channel_first=channel_first,
                ssm_d_state=ssm_d_state,
                ssm_ratio=ssm_ratio,
                ssm_dt_rank=ssm_dt_rank,
                ssm_act_layer=ssm_act_layer,
                ssm_conv=ssm_conv,
                ssm_conv_bias=ssm_conv_bias,
                ssm_drop_rate=ssm_drop_rate,
                ssm_init=ssm_init,
                forward_type=forward_type,
                mlp_ratio=mlp_ratio,
                mlp_act_layer=mlp_act_layer,
                mlp_drop_rate=mlp_drop_rate,
                gmlp=gmlp,
                use_checkpoint=use_checkpoint,
            ))

        return nn.Sequential(OrderedDict(
            blocks=nn.Sequential(*blocks, ),
            downsample=downsample,
        ))

    def forward(self, x: torch.Tensor):
        x = self.patch_embed(x)
        for layer in self.layers:
            x = layer(x)
        return x


class _expert_extractor(nn.Module):
    def __init__(
            self,
            depths=[2, 2, 8, 2],
            dims=[96, 192, 384, 768],
            # =========================
            ssm_d_state=1,
            ssm_ratio=1.0,
            ssm_dt_rank="auto",
            ssm_conv=3,
            ssm_conv_bias=False,
            ssm_drop_rate=0.0,
            ssm_init="v0",
            forward_type="v05_noz",
            # =========================_shared_extractor
            mlp_ratio=4.0,
            mlp_drop_rate=0.0,
            gmlp=False,
            # =========================
            drop_path_rate=0.2,
            patch_norm=True,
            use_checkpoint=False,
            # =========================
            **kwargs,
    ):
        super().__init__()
        self.channel_first = True
        self.num_layers = len(depths) // 2
        self.num_features = dims[-1]
        self.dims = dims
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule

        norm_layer = LayerNorm2d
        ssm_act_layer = nn.SiLU
        mlp_act_layer = nn.GELU

        _make_downsample = self._make_downsample_v3

        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers, len(depths)):
            downsample = _make_downsample(
                self.dims[i_layer - 1],
                self.dims[i_layer],
                norm_layer=norm_layer,
                channel_first=self.channel_first,
            ) if (i_layer < len(depths)) else nn.Identity()

            self.layers.append(self._make_layer(
                dim=self.dims[i_layer],
                drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                use_checkpoint=use_checkpoint,
                norm_layer=norm_layer,
                downsample=downsample,
                channel_first=self.channel_first,
                # =================
                ssm_d_state=ssm_d_state,
                ssm_ratio=ssm_ratio,
                ssm_dt_rank=ssm_dt_rank,
                ssm_act_layer=ssm_act_layer,
                ssm_conv=ssm_conv,
                ssm_conv_bias=ssm_conv_bias,
                ssm_drop_rate=ssm_drop_rate,
                ssm_init=ssm_init,
                forward_type=forward_type,
                # =================
                mlp_ratio=mlp_ratio,
                mlp_act_layer=mlp_act_layer,
                mlp_drop_rate=mlp_drop_rate,
                gmlp=gmlp,
            ))

        self.apply(self._init_weights)

    def _init_weights(self, m: nn.Module):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    # used in building optimizer
    @torch.jit.ignore
    def no_weight_decay(self):
        return {"pos_embed"}

    # used in building optimizer
    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {}

    @staticmethod
    def _make_downsample_v3(dim=96, out_dim=192, norm_layer=nn.LayerNorm, channel_first=False):
        # if channel first, then Norm and Output are both channel_first
        return nn.Sequential(
            (nn.Identity() if channel_first else Permute(0, 3, 1, 2)),
            nn.Conv2d(dim, out_dim, kernel_size=3, stride=2, padding=1),
            (nn.Identity() if channel_first else Permute(0, 2, 3, 1)),
            norm_layer(out_dim),
        )

    @staticmethod
    def _make_layer(
            dim=96,
            drop_path=[0.1, 0.1],
            use_checkpoint=False,
            norm_layer=nn.LayerNorm,
            downsample=nn.Identity(),
            channel_first=False,
            # ===========================
            ssm_d_state=16,
            ssm_ratio=2.0,
            ssm_dt_rank="auto",
            ssm_act_layer=nn.SiLU,
            ssm_conv=3,
            ssm_conv_bias=True,
            ssm_drop_rate=0.0,
            ssm_init="v0",
            forward_type="v2",
            # ===========================
            mlp_ratio=4.0,
            mlp_act_layer=nn.GELU,
            mlp_drop_rate=0.0,
            gmlp=False,
            **kwargs,
    ):
        # if channel first, then Norm and Output are both channel_first
        depth = len(drop_path)
        blocks = []
        for d in range(depth):
            blocks.append(VSSBlock(
                hidden_dim=dim,
                drop_path=drop_path[d],
                norm_layer=norm_layer,
                channel_first=channel_first,
                ssm_d_state=ssm_d_state,
                ssm_ratio=ssm_ratio,
                ssm_dt_rank=ssm_dt_rank,
                ssm_act_layer=ssm_act_layer,
                ssm_conv=ssm_conv,
                ssm_conv_bias=ssm_conv_bias,
                ssm_drop_rate=ssm_drop_rate,
                ssm_init=ssm_init,
                forward_type=forward_type,
                mlp_ratio=mlp_ratio,
                mlp_act_layer=mlp_act_layer,
                mlp_drop_rate=mlp_drop_rate,
                gmlp=gmlp,
                use_checkpoint=use_checkpoint,
            ))

        return nn.Sequential(OrderedDict(
            downsample=downsample,
            blocks=nn.Sequential(*blocks, ),
        ))

    def forward(self, x: torch.Tensor):
        for layer in self.layers:
            x = layer(x)
        return x



class PAMIX(nn.Module):
    def __init__(self, num4class=2, num4expert=4, dim=768):
        super().__init__()
        self.num4expert = num4expert

        self._shared_extractor = _shared_extractor()
        self._router_selection = AggregationRouter(num4expert=num4expert, channel=dim//4, nums=14**2)
        self._expert_extractor_list = nn.ModuleList(_expert_extractor() for _ in range(num4expert))
        self._projector_list = nn.ModuleList(nn.Sequential(OrderedDict(
            norm=LayerNorm2d(dim), # B,H,W,C
            avgpool=nn.AdaptiveAvgPool2d(1),
            flatten=nn.Flatten(1)
        ))  for _ in range(num4expert))

        self._classifier = nn.Sequential(
            nn.Linear(dim, 128),
            nn.ReLU(True),
            nn.Linear(128, 20),
            nn.ReLU(True),
            nn.Linear(20, num4class)
        )
        self.lrpe = PrototypeMemoryEvolution(feat_dim=dim, num_classes=num4class, num4expert=num4expert)

    def forward(self, x, label=None):
        b,s,c,h,w = x.shape
        x = rearrange(x, 'b s c h w -> (b s) c h w', b=b, s=s, c=c, h=h, w=w)
        x_shared = self._shared_extractor(x)
        x_router = self._router_selection(x_shared)
        x_divide = x_shared.view(b, s, x_shared.shape[1], x_shared.shape[2], x_shared.shape[3])

        x_mixture = []
        x_level = []
        for ii in range(self.num4expert):
            x_expert_feature = self._projector_list[ii](self._expert_extractor_list[ii](x_divide[:,ii,:,:]))
            x_level.append(x_expert_feature)
            x_project_feature = x_expert_feature * x_router[:,ii]
            x_mixture.append(x_project_feature)
        x_mixture_feature = torch.stack(x_mixture, 2).sum(2)
        x_level_feature = torch.stack(x_level, 1)

        return self._classifier(x_mixture_feature), self.lrpe(x_mixture_feature, x_level_feature, label), x_mixture_feature


class UniPAMIX(nn.Module):
    def __init__(self, num4class=2, dim=768):
        super().__init__()

        self._shared_extractor = _shared_extractor()
        self._expert_extractor = _expert_extractor()
        self._projector = nn.Sequential(OrderedDict(
            norm=LayerNorm2d(dim), # B,H,W,C
            avgpool=nn.AdaptiveAvgPool2d(1),
            flatten=nn.Flatten(1)
        ))
        self._classifier = nn.Sequential(
            nn.Linear(dim, 128),
            nn.ReLU(True),
            nn.Linear(128, 20),
            nn.ReLU(True),
            nn.Linear(20, num4class)
        )

    def forward(self, in_image):
        pri_feat = self._shared_extractor(in_image)
        adv_feat = self._expert_extractor(pri_feat)
        dia_feat = self._projector(adv_feat)
        out_prob = self._classifier(dia_feat)

        return out_prob, dia_feat


def _migrate_pretrained_weights(model, pretrianed_path):
    pretrianed_dict = torch.load(pretrianed_path)['model']
    pretrianed_dict_sel = {}
    for k, v in model.state_dict().items():
        if k.split('.')[0] == '_shared_extractor':
            if not 'msff' in k:
                pretrianed_dict_sel[k] = pretrianed_dict[k.replace('_shared_extractor.','')]
            else:
                if 'bias' in k:
                    pretrianed_dict_sel[k] = torch.nn.init.zeros_(v)
                else:
                    pretrianed_dict_sel[k] = torch.nn.init.xavier_uniform_(torch.empty_like(v))

        elif k.split('.')[0] in ['_expert_extractor_list']:
            k_n = k
            for iii in range(11):
                k_n = k_n.replace('_expert_extractor_list.{}.'.format(iii),'')

            if 'layers.0.downsample' in k_n:
                k_n = k_n.replace('layers.0.downsample', 'layers.1.downsample')
            elif 'layers.0.blocks' in k_n:
                k_n = k_n.replace('layers.0.blocks', 'layers.2.blocks')
            elif 'layers.1.downsample' in k_n:
                k_n = k_n.replace('layers.1.downsample', 'layers.2.downsample')
            elif 'layers.1.blocks' in k_n:
                k_n = k_n.replace('layers.1.blocks', 'layers.3.blocks')

            if not 'msff' in k:
                pretrianed_dict_sel[k] = pretrianed_dict[k_n]
            else:
                if 'bias' in k:
                    pretrianed_dict_sel[k] = torch.nn.init.zeros_(v)
                else:
                    pretrianed_dict_sel[k] = torch.nn.init.xavier_uniform_(torch.empty_like(v))
        else:
            if 'bias' in k:
                pretrianed_dict_sel[k] = torch.nn.init.zeros_(v)
            elif 'norm.weight' in k:
                pretrianed_dict_sel[k] = torch.nn.init.ones_(v)
            else:
                pretrianed_dict_sel[k] = torch.nn.init.xavier_uniform_(torch.empty_like(v))

    return pretrianed_dict_sel


def _migrate_pamix_t_weights(model, pamix_t_path, num4expert=4):

    pretrianed_dict = torch.load(pamix_t_path)
    pretrianed_dict_sel = {}
    for k, v in model.state_dict().items():
        if k.split('.')[0] == '_shared_extractor' or k.split('.')[0] == '_classifier':
            pretrianed_dict_sel[k] = pretrianed_dict[k]
        elif k.split('.')[0] == '_expert_extractor':
            pretrianed_dict_sel[k] = pretrianed_dict[
                k.replace('_expert_extractor', '_expert_extractor_list.{}'.format(num4expert-1))
            ]
        elif k.split('.')[0] == '_projector':
            pretrianed_dict_sel[k] = pretrianed_dict[
                k.replace('_projector', '_projector_list.{}'.format(num4expert-1))
            ]
    return pretrianed_dict_sel


