from torch import nn
import torch.nn.functional as F
import torch


#activation is used for defining different activation functions
class activation():

    #negative_slope=0.2_Leaky ReLU
    def __init__(self, act_type, negative_slope=0.2, inplace=True):
        super().__init__()
        self._act_type = act_type
        self.negative_slope = negative_slope
        self.inplace = inplace

    def __call__(self, input):
        if self._act_type == 'leaky':
            return F.leaky_relu(input, negative_slope=self.negative_slope, inplace=self.inplace)
        elif self._act_type == 'relu':
            return F.relu(input, inplace=self.inplace)
        elif self._act_type == 'sigmoid':
            return torch.sigmoid(input)
        else:
            raise NotImplementedError

class ED(nn.Module):
    def __init__(self, encoder, decoder, hidden_ch, H, W, dropout: float = 0.0, use_gap: bool = False):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.use_gap = use_gap
        self.dropout = nn.Dropout(dropout) if dropout and dropout > 0 else nn.Identity()
        self.hidden_ch, self.H, self.W = hidden_ch, H, W
        in_features = hidden_ch if use_gap else hidden_ch * H * W
        self.head = nn.Linear(in_features, 1, bias=True)

    @staticmethod
    def _prep_mask(mask, device, dtype, B, T, H, W):
        if mask.dim() == 2:           # (H,W)
            mask_hw = mask.unsqueeze(0).unsqueeze(0)
        elif mask.dim() == 3:         # (B/H,1,H,W) ->  (B,1,H,W)
            mask_hw = mask.unsqueeze(1) if mask.shape[1] != 1 else mask
        else:                          # (1,1,H,W) or (B,1,H,W)
            mask_hw = mask
        mask_hw = mask_hw.to(device=device, dtype=dtype)       # (B?/1,1,H,W)
        if mask_hw.shape[0] == 1 and B > 1:
            mask_hw = mask_hw.expand(B, -1, -1, -1)            # (B,1,H,W)
        mask_t = mask_hw.unsqueeze(1).expand(B, T, 1, H, W)    # (B,T,1,H,W)
        return mask_hw, mask_t

    def forward(self, x_img: torch.Tensor, y_hist: torch.Tensor = None, mask: torch.Tensor = None) -> torch.Tensor:
        B, T, C, H, W = x_img.shape
        # —— mask——
        mask_hw = None
        mask_t  = None
        if mask is None and hasattr(self, "mask_hw_buf"):
            # buffer: (1,1,H,W) 或 (B,1,H,W)
            mask_hw = self.mask_hw_buf.to(device=x_img.device, dtype=x_img.dtype)
            if mask_hw.dim() == 4 and mask_hw.shape[0] == 1 and B > 1:
                mask_hw = mask_hw.expand(B, -1, -1, -1)                      # (B,1,H,W)
            mask_t = mask_hw.unsqueeze(1).expand(B, T, 1, self.H, self.W)     # (B,T,1,H,W)
        elif mask is not None:
            mask_hw, mask_t = self._prep_mask(mask, x_img.device, x_img.dtype, B, T, self.H, self.W)

        # —— 输入门控：盆外像元直接置零 ——
        if mask_t is not None:
            x_img = x_img * mask_t

        # Encoder：提时空特征
        enc_out = self.encoder(x_img)                                # Consistent with encoder.py :contentReference[oaicite:2]{index=2}
        # Decoder：Self+h0,c0
        dec_feat = self.decoder(enc_out, y_hist)                     # (B, hidden_ch, H, W) :contentReference[oaicite:3]{index=3}

        # 头部映射：默认用 GAP；若有 mask 则用 masked GAP
        if self.use_gap:
            if mask_hw is not None:
                m = mask_hw.to(dec_feat.dtype)                       # (B,1,H,W)
                num = (dec_feat * m).sum(dim=(2, 3))                 # (B, hidden_ch)
                den = m.sum(dim=(2, 3)).clamp_min(1.0)               # (B,1) 
                vec = num / den
            else:
                vec = dec_feat.mean(dim=(2, 3))                      # (B, hidden_ch)
            out = self.head(self.dropout(vec))                       # (B,1)
        else:
            if mask_hw is not None:
                dec_feat = dec_feat * mask_hw                      
            flat = torch.flatten(dec_feat, start_dim=1)              # (B, hidden_ch*H*W)
            out = self.head(self.dropout(flat))                      # (B,1)

        return out.squeeze(-1)

