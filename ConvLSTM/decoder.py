# decoder.py
from torch import nn
from utils import make_layers
import torch


def _broadcast_runoff(y_hist, H, W):
    if y_hist.dim() == 2:
        y_hist = y_hist.unsqueeze(-1)                 
    B, Ty, _ = y_hist.shape
    y = y_hist.permute(1, 0, 2).reshape(Ty, B, 1, 1, 1)  
    y = y.expand(Ty, B, 1, H, W)
    return y


class Decoder(nn.Module):
    def __init__(self, subnets, rnns):
        super().__init__()
        assert len(subnets) == len(rnns)
        self.blocks = len(subnets)
        for index, (params, rnn) in enumerate(zip(subnets, rnns)):
            setattr(self, f'rnn{self.blocks - index}', rnn)
            setattr(self, f'stage{self.blocks - index}', make_layers(params))

    def forward_by_stage(self, y_seq_map, state, subnet, rnn):
        Ty, B, C, H, W = y_seq_map.size()
        out_seq, _ = rnn(y_seq_map, state, seq_len=Ty)  # (Ty, B, Ch, H, W)

        last = out_seq[-1]                               # (B, Ch, H, W)
        last = subnet(last)                              # (B, Cout, H, W)
        last_rep = last.unsqueeze(0).expand(Ty, B, last.size(1), last.size(2), last.size(3)).contiguous()
        return last_rep, last

    def forward(self, hidden_states, y_hist):

        h_last = hidden_states[-1][0] if isinstance(hidden_states[-1], (tuple, list)) else hidden_states[-1]
        B, C, H, W = h_last.shape
        y_seq_map = _broadcast_runoff(y_hist, H, W)      # (Ty,B,1,H,W)

        feat_last = None
        for i in reversed(range(1, self.blocks + 1)):
            enc_state = hidden_states[-1] if self.blocks == 1 else hidden_states[i - 1]
            y_seq_map, feat_last = self.forward_by_stage(
                y_seq_map,
                enc_state,
                getattr(self, f'stage{i}'),
                getattr(self, f'rnn{i}')
            )
        return feat_last  # (B, C, H, W)
