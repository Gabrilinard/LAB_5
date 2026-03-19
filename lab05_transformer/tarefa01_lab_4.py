import math
import torch
import torch.nn as nn
import torch.nn.functional as F

def atencao_produto_escalar(q, k, v, mascara=None):
    d_k = q.size(-1)
    
    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)

    if mascara is not None:
        scores = scores.masked_fill(mascara == 0, float("-inf"))

    pesos = F.softmax(scores, dim=-1)
    pesos = pesos.nan_to_num(nan=0.0)
    
    saida = torch.matmul(pesos, v)
    return saida, pesos

class AtencaoMultihead(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        self.n_heads = n_heads
        self.d_k = d_model // n_heads

        self.proj_q = nn.Linear(d_model, d_model)
        self.proj_k = nn.Linear(d_model, d_model)
        self.proj_v = nn.Linear(d_model, d_model)
        self.proj_out = nn.Linear(d_model, d_model)
        
        self.camada_dropout = nn.Dropout(dropout)

    def separar_cabecas(self, x):
        batch, seq, _ = x.shape
        return x.view(batch, seq, self.n_heads, self.d_k).transpose(1, 2)

    def reunir_cabecas(self, x):
        batch, _, seq, _ = x.shape
        return x.transpose(1, 2).contiguous().view(batch, seq, -1)

    def forward(self, q, k, v, mascara=None):
        q = self.separar_cabecas(self.proj_q(q))
        k = self.separar_cabecas(self.proj_k(k))
        v = self.separar_cabecas(self.proj_v(v))

        resultado_attn, _ = atencao_produto_escalar(q, k, v, mascara)
        
        unido = self.reunir_cabecas(resultado_attn)
        return self.proj_out(self.camada_dropout(unido))


class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.camada1 = nn.Linear(d_model, d_ff)
        self.camada2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = F.relu(self.camada1(x))
        x = self.dropout(x)
        return self.camada2(x)


class AddNorm(nn.Module):
    def __init__(self, d_model, dropout=0.1):
        super().__init__()
        self.norma = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, saida_subcamada):
        return self.norma(x + self.dropout(saida_subcamada))


def rodar_testes():
    d_model = 512
    d_ff = 2048
    n_heads = 8
    batch = 2
    seq = 10
    
    entrada = torch.randn(batch, seq, d_model)
    print(f"Shape inicial: {entrada.shape}")

    mha = AtencaoMultihead(d_model, n_heads)
    out_mha = mha(entrada, entrada, entrada)
    print(f"MHA ok: {out_mha.shape}")
    
    ffn = FeedForward(d_model, d_ff)
    out_ffn = ffn(out_mha)
    print(f"FFN ok: {out_ffn.shape}")
    
    residuo = AddNorm(d_model)
    final = residuo(entrada, out_ffn)
    print(f"Final ok: {final.shape}")

if __name__ == "__main__":
    rodar_testes()