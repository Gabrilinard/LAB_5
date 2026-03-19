import torch
import torch.nn as nn
from tarefa01_lab_4 import AtencaoMultihead, FeedForward, AddNorm

class EncoderBlock(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attention = AtencaoMultihead(d_model, n_heads, dropout)
        self.add_norm_1 = AddNorm(d_model, dropout)
        
        self.ffn = FeedForward(d_model, d_ff, dropout)
        self.add_norm_2 = AddNorm(d_model, dropout)

    def forward(self, x, mascara=None):
        atencao = self.self_attention(x, x, x, mascara)
        
        x = self.add_norm_1(x, atencao)
        
        ffn_out = self.ffn(x)
        
        x = self.add_norm_2(x, ffn_out)
        
        return x

class Encoder(nn.Module):
    def __init__(self, num_layers, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList([
            EncoderBlock(d_model, n_heads, d_ff, dropout) 
            for _ in range(num_layers)
        ])

    def forward(self, x, mascara=None):
        for layer in self.layers:
            x = layer(x, mascara)
        return x 
    

def rodar_teste_encoder():
    d_model = 512
    n_heads = 8
    d_ff = 2048
    qntd_camadas = 6
    
    encoder_completo = Encoder(qntd_camadas, d_model, n_heads, d_ff)
    
    x = torch.randn(2, 10, d_model)
    
    z = encoder_completo(x)
    
    print(f"Entrada (X): {x.shape}")
    print(f"Memória Rica (Z): {z.shape}")
    print("Sucesso: O Encoder processou todas as camadas e manteve as dimensões!")

if __name__ == "__main__":
    rodar_teste_encoder()