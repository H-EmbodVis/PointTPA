import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
import math

def scatter_points_to_batch(points, batch_idx):
    splits_points = [points[batch_idx == i] for i in range(batch_idx.max() + 1)]
    padded_points = pad_sequence(splits_points, batch_first=True)
    return padded_points.permute(0, 2, 1).contiguous()

def scatter_points_to_batch_f(points, batch_idx):
    splits_points = [points[batch_idx == i] for i in range(batch_idx.max() + 1)]
    padded_points = pad_sequence(splits_points, batch_first=True)
    return padded_points

def restore_points_from_batch(batch_points: torch.Tensor, counts: torch.Tensor):
    device = batch_points.device
    batch_points = batch_points.permute(0, 2, 1).contiguous()
    _, _, c = batch_points.shape
    indices = torch.arange(batch_points.size(1), device=device)  # [N]
    mask = indices.unsqueeze(0) < counts.unsqueeze(1)  # [b, N]
    points = batch_points[mask.unsqueeze(-1).expand_as(batch_points)].view(-1, c)
    return points

def restore_points_from_batch_f(batch_points: torch.Tensor, counts: torch.Tensor):
    device = batch_points.device
    _, _, c = batch_points.shape
    indices = torch.arange(batch_points.size(1), device=device)  # [N]
    mask = indices.unsqueeze(0) < counts.unsqueeze(1)  # [b, N]
    points = batch_points[mask.unsqueeze(-1).expand_as(batch_points)].view(-1, c)
    return points

class Attention(nn.Module):
    def __init__(self, in_planes, ratios, K, init_weight=True):
        super(Attention, self).__init__()
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        if in_planes!=3:
            hidden_planes = max(int(in_planes*ratios)+1, 16)
        else:
            hidden_planes = K
        self.fc1 = nn.Conv1d(in_planes, hidden_planes, 1, bias=False)
        #self.bn = nn.BatchNorm2d(hidden_planes)
        self.fc2 = nn.Conv1d(hidden_planes, K, 1, bias=True)
        #self.topk_k = None
        if init_weight:
            self._initialize_weights()


    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            if isinstance(m ,nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.avgpool(x)  # [B, C, 1]
        x = self.fc1(x)  # [B, hidden, 1]
        x = F.relu(x)
        x = self.fc2(x).squeeze(-1)  # [B, K]
        return F.softmax(x / 4., 1)
    
class DynamicParameterProjector(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, 
                 padding=0, dilation=1, groups=1, 
                 bias=True, K=4, init_weight=True):
        super(DynamicParameterProjector, self).__init__()
        assert in_planes%groups==0
        self.in_planes = in_planes
        self.out_planes = out_planes
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.bias = bias
        self.K = K
        self.weight = nn.Parameter(torch.randn(K, out_planes, in_planes//groups, kernel_size), requires_grad=True)
        if bias:
            self.bias = nn.Parameter(torch.zeros(K, out_planes))
        else:
            self.bias = None
        if init_weight:
            self._initialize_weights()

    def _initialize_weights(self):
        for i in range(self.K):
            nn.init.kaiming_uniform_(self.weight[i])

    def forward(self, x, softmax_attention):
        B, C, L = x.size()
        #print(f"before{x.shape}")
        if self.kernel_size == 1 and self.groups == 1:
            aggregate_weight = torch.einsum('bk,koc->boc',
                                            softmax_attention, self.weight.squeeze(-1))
            output = torch.einsum('boc,bcl->bol', aggregate_weight, x)

            if self.bias is not None:
                aggregate_bias = torch.mm(softmax_attention, self.bias)
                output += aggregate_bias.unsqueeze(-1).expand_as(output)
            return output

        aggregate_weight = torch.mm(softmax_attention, self.weight.view(self.K, -1))
        aggregate_weight = aggregate_weight.reshape(B, self.out_planes, C // self.groups, self.kernel_size)
        aggregate_weight = aggregate_weight.reshape(B * self.out_planes, C // self.groups, self.kernel_size)
        x_reshaped = x.reshape(1, B * C, L)

        if self.bias is not None:
            aggregate_bias = torch.mm(softmax_attention, self.bias).view(B * self.out_planes)
        else:
            aggregate_bias = None

        output = F.conv1d(x_reshaped,
                          weight=aggregate_weight,
                          bias=aggregate_bias,
                          stride=self.stride,
                          padding=self.padding,
                          dilation=self.dilation,
                          groups=B * self.groups)  # B * groups

        # restore to original shape [B, out_planes, L_out]
        output = output.reshape(B, self.out_planes, -1)
        return output

class SerialNeighborhoodGrouping:
    def __init__(self, param: int, mode: str = "num"):
        self.param = int(param)
        self.valid_mode = {"num", "length"}
        self.set_mode(mode)

    def set_mode(self, mode: str):
        if mode not in self.valid_mode:
            raise ValueError(f"mode must be one of {self.valid_mode}, but got {mode}")
        self.mode = mode

    def _prepare(self, points: torch.Tensor, batch_idx: torch.Tensor):
        if self.param <= 0:
            raise ValueError(f"param must be > 0, but got {self.param}")
        if points.ndim != 2:
            raise ValueError(f"points must be [N, C], but got shape={tuple(points.shape)}")
        if batch_idx.ndim != 1 or batch_idx.shape[0] != points.shape[0]:
            raise ValueError("batch_idx must be [N] and aligned with points")

        points_batch = scatter_points_to_batch(points, batch_idx)   # [B, C, N]
        B, C, N = points_batch.shape

        # ceil division
        count = (N + self.param - 1) // self.param
        padded_N = self.param * count

        padded = points_batch.new_zeros(B, C, padded_N)
        padded[:, :, :N] = points_batch

        meta = {
            "B": B,
            "C": C,
            "N": N,
            "count": count,
            "param": self.param,
            "mode": self.mode,
        }
        return padded, meta

    def divide(self, points: torch.Tensor, batch_idx: torch.Tensor, mode: str = None):
        use_mode = self.mode if mode is None else mode
        if use_mode not in self.valid_mode:
            raise ValueError(f"mode must be one of {self.valid_mode}, but got {use_mode}")

        padded, meta = self._prepare(points, batch_idx)
        B, C, count, param = meta["B"], meta["C"], meta["count"], meta["param"]

        if use_mode == "num":
            # [B, C, Np] -> [B, C, num=param, slice_len=count]
            grouped = padded.view(B, C, param, count)
            slice_len = count
            num_groups = param
        else:
            # [B, C, Np] -> [B, C, num=count, slice_len=param]
            grouped = padded.view(B, C, count, param)
            slice_len = param
            num_groups = count

        # [B, C, num, slice_len] -> [B*num, C, slice_len]
        neighbors = grouped.permute(0, 2, 1, 3).reshape(B * num_groups, C, slice_len)

        meta.update({
            "mode": use_mode,
            "slice_len": slice_len,
            "num_groups": num_groups,
        })
        return neighbors, meta

    def merge(self, neighbors: torch.Tensor, meta: dict):
        B = meta["B"]
        N = meta["N"]
        mode = meta["mode"]
        num_groups = meta["num_groups"]
        slice_len = meta["slice_len"]

        if neighbors.ndim != 3:
            raise ValueError(f"neighbors must be [B*num, C_out, slice_len], got {neighbors.shape}")

        _, C_out, got_slice_len = neighbors.shape
        if got_slice_len != slice_len:
            raise ValueError(f"slice_len mismatch: expect {slice_len}, got {got_slice_len}")

        # [B*num, C_out, slice_len] -> [B, num, C_out, slice_len]
        x = neighbors.view(B, num_groups, C_out, slice_len)
        # -> [B, C_out, num, slice_len] -> [B, C_out, padded_N]
        x = x.permute(0, 2, 1, 3).reshape(B, C_out, -1)
        # crop back to original N
        x = x[:, :, :N]
        return x

        

class DynamicParameterLayer(nn.Module):
    def __init__(self, in_channel, out_channel, 
                 stage, layer, param=100, mode="num",
                 kernel_size=1, ratios=0.25, 
                 K=4, bias=True,
                 stride=1, padding=0):
        super().__init__()

        self.attn = Attention(in_planes=in_channel, ratios=ratios, K=K)
        self.dpp = DynamicParameterProjector(in_planes=in_channel, out_planes=out_channel, 
                                             kernel_size=kernel_size, K=K, bias=bias,
                                      stride=stride, padding=padding)
        self.sng = SerialNeighborhoodGrouping(param=param, mode=mode)
        self.stage = stage
        self.layer = layer
        self.mode = mode

    def forward(self, x, batch_idx, counts):
        #segment serialized point
        neighbors, meta = self.sng.divide(x, batch_idx, mode=self.mode)
        #get batch attention weight [B*num, K]
        attn_weights = self.attn(neighbors)
        #conduct dynamic projection based on weight [B*S, out_C, len]
        proj_out = self.dpp(neighbors, attn_weights)
        #merge neighbors to original shape
        output_batch = self.sng.merge(proj_out, meta)
        #merge batch
        return restore_points_from_batch(output_batch, counts)    

class DynamicParameterModule(nn.Module):
    def __init__(self, in_channel, mid, stage, layer, bias=True,
                 scale=1., dy_down=True, dy_up=False, param=100, 
                 kernel_size=1, ratios=0.25, 
                 K=4, order_idx=0, mode="num",
                 stride=1, padding=0):
        super().__init__()
        self.norm = nn.LayerNorm(in_channel)
        self.order_idx = order_idx

        if dy_down:
            self.down = DynamicParameterLayer(
                in_channel=in_channel, out_channel=mid, stage=stage, bias=bias,
                layer=layer, param=param, kernel_size=kernel_size, ratios=ratios,
                K=K, mode=mode, stride=stride, padding=padding)
        else: self.down = nn.Linear(in_channel, mid)

        self.act = nn.ReLU()

        if dy_up:
            self.up = DynamicParameterLayer(
                in_channel=mid, out_channel=in_channel, stage=stage, bias=bias,
                layer=layer, param=param, kernel_size=kernel_size, ratios=ratios,
                K=K, mode=mode, stride=stride, padding=padding)
            with torch.no_grad():
                nn.init.zeros_(self.up.dpp.weight)
                if bias: nn.init.zeros_(self.up.dpp.bias)
        else: 
            self.up = nn.Linear(mid, in_channel)
            with torch.no_grad():
                nn.init.zeros_(self.up.weight)
                nn.init.zeros_(self.up.bias)
        

        self.scale = scale
        self.dy_down = dy_down
        self.dy_up = dy_up

    def forward(self, point):
        #get basic information
        batch_idx = point.batch.to(torch.long)    
        feat = point.feat[point.serialized_order[self.order_idx]]
        counts = torch.bincount(batch_idx)
        #norm
        feat = self.norm(feat)
        #down projection
        if self.dy_down:
            feat = self.down(feat, batch_idx, counts)
        else: feat = self.down(feat)
        #activation
        feat = self.act(feat)
        #up projection
        if self.dy_up:
            feat = self.up(feat, batch_idx, counts)
        else: feat = self.up(feat)
        #scale
        feat = feat * self.scale
        # if feat.dtype != x.feat.dtype:
        #     feat = feat.to(x.feat.dtype)
        return feat[point.serialized_inverse[self.order_idx]]

