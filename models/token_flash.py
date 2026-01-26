from functools import lru_cache
import math
import torch
import torch.nn as nn
from timm.models.layers import DropPath
from flash_attn import flash_attn_func

# ---------- MLP ----------
class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None,
                 act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

# ---------- 修正后的TokenFlash ----------
class TokenFlash(nn.Module):
    """
    FlashAttention2-based drop-in replacement for Token_performer.
    Fixed dimension alignment issues.
    """

    def __init__(self, dim, in_dim, head_cnt=1, kernel_ratio=0.5,
                 dp1=0., dp2=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm,
                 causal=False, **kwargs):
        super().__init__()
        
        # 保存原始维度信息
        self.dim = dim
        self.orig_in_dim = in_dim
        self.head_cnt = head_cnt
        
        # 更合理的维度对齐策略
        self.in_dim, self.num_heads = self._smart_align_dim_and_heads(dim, in_dim, head_cnt)
        self.head_dim = self.in_dim // self.num_heads
        
        # 验证维度设置
        # self._validate_dimensions()
        
        self.scale = self.head_dim ** -0.5
        self.causal = causal

        # QKV投影 - 从输入dim到调整后的in_dim
        self.qkv = nn.Linear(dim, self.in_dim * 3, bias=False)
        
        # 输出投影 - 从调整后的in_dim回到原始in_dim
        self.proj = nn.Linear(self.in_dim, self.orig_in_dim)
        self.proj_drop = nn.Dropout(dp1)
        
        # 如果需要维度转换，添加残差连接投影
        if dim != self.orig_in_dim:
            self.residual_proj = nn.Linear(dim, self.orig_in_dim)
        else:
            self.residual_proj = nn.Identity()

        # 归一化层
        self.norm1 = norm_layer(dim)
        self.norm2 = norm_layer(self.orig_in_dim)
        self.drop_path = DropPath(dp2) if dp2 > 0. else nn.Identity()

        # MLP - 使用原始输出维度
        mlp_hidden = max(int(self.orig_in_dim * kernel_ratio), 32)
        self.mlp = Mlp(self.orig_in_dim, hidden_features=mlp_hidden,
                       out_features=self.orig_in_dim, act_layer=act_layer, drop=dp1)

    
    def _smart_align_dim_and_heads(self, dim, in_dim, head_cnt):
        """
        更智能的维度对齐策略，确保head_dim在8-128范围内
        """
        # 基本要求：in_dim必须是head_cnt的倍数
        min_head_dim = 8
        max_head_dim = 128
        
        # 尝试找到合适的头数和维度
        for num_heads in range(head_cnt, 0, -1):
            # 计算所需的in_dim
            required_in_dim = max(in_dim, num_heads * min_head_dim)
            if required_in_dim % num_heads != 0:
                required_in_dim = ((required_in_dim + num_heads - 1) // num_heads) * num_heads
            
            head_dim = required_in_dim // num_heads
            
            # 检查head_dim是否在有效范围内
            if min_head_dim <= head_dim <= max_head_dim:
                print(f"Dimension alignment: input_dim={dim}, original_in_dim={in_dim}, "
                      f"aligned_in_dim={required_in_dim}, num_heads={num_heads}, head_dim={head_dim}")
                return required_in_dim, num_heads
        
        # 如果找不到合适的配置，使用默认值
        default_num_heads = max(1, in_dim // 64)
        default_in_dim = default_num_heads * 64
        print(f"Warning: Using default dimension alignment: input_dim={dim}, original_in_dim={in_dim}, "
              f"aligned_in_dim={default_in_dim}, num_heads={default_num_heads}, head_dim={64}")
        return default_in_dim, default_num_heads

    def _validate_dimensions(self):
        """验证所有维度设置是否合理"""
        assert self.in_dim % self.num_heads == 0, \
            f"in_dim({self.in_dim}) must be divisible by num_heads({self.num_heads})"
        
        self.head_dim = self.in_dim // self.num_heads
        assert 8 <= self.head_dim <= 128, \
            f"head_dim({self.head_dim}) should be between 8 and 128 for FlashAttention"
        
        print(f"Validation passed: in_dim={self.in_dim}, num_heads={self.num_heads}, "
              f"head_dim={self.head_dim}")


    def forward(self, x):
        """
        x: (B, N, C) where C = dim
        return: (B, N, orig_in_dim)
        """
        B, N, C = x.shape
        assert C == self.dim, f"Input dimension {C} doesn't match expected {self.dim}"
        
        # pre-norm
        x_norm = self.norm1(x)

        # QKV投影: (B, N, dim) -> (B, N, 3 * in_dim)
        qkv = self.qkv(x_norm)
        qkv = qkv.reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, H, N, D)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # FlashAttention2要求(B, N, H, D)格式
        q = q.transpose(1, 2)  # (B, N, H, D)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # FlashAttention前向传播

        attn_out = flash_attn_func(
            q, k, v,
            dropout_p=self.proj_drop.p if self.training else 0.0,
            softmax_scale=self.scale,
            causal=self.causal
        )  # (B, N, H, D)
     

        # 重塑: (B, N, H, D) -> (B, N, in_dim)
        attn_out = attn_out.reshape(B, N, self.in_dim)
        
        # 投影回原始输出维度
        attn_out = self.proj(attn_out)
        attn_out = self.proj_drop(attn_out)
      
        # print(f"Residual connection: x_original shape {x.shape}, "
        #         f"attn_out shape {attn_out.shape}")

        # 正确的残差连接：投影输入以匹配输出维度
        x_residual = self.residual_proj(x)
        x = x_residual + self.drop_path(attn_out)

        # MLP部分
        mlp_out = self.mlp(self.norm2(x))
        x = x + self.drop_path(mlp_out)

        return x

