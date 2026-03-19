import torch


def make_causal_mask(seq_len: int, device: torch.device) -> torch.Tensor:
    mask = torch.tril(torch.ones(seq_len, seq_len, device=device))
    return mask.unsqueeze(0).unsqueeze(0)  


def make_padding_mask(seq: torch.Tensor, pad_idx: int = 0) -> torch.Tensor:
    return (seq != pad_idx).unsqueeze(1).unsqueeze(2)  
