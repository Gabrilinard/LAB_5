import torch
import torch.nn as nn
import torch.nn.functional as F
from tarefa01_lab_4 import AtencaoMultihead, FeedForward, AddNorm

class DecoderBlock(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attn = AtencaoMultihead(d_model, n_heads, dropout)
        self.add_norm_1 = AddNorm(d_model, dropout)
        
        self.cross_attn = AtencaoMultihead(d_model, n_heads, dropout)
        self.add_norm_2 = AddNorm(d_model, dropout)
        
        self.ffn = FeedForward(d_model, d_ff, dropout)
        self.add_norm_3 = AddNorm(d_model, dropout)

    def forward(self, y, z, mask_causal=None, mask_encoder=None):
        auto_atencao = self.self_attn(y, y, y, mask_causal)
        y = self.add_norm_1(y, auto_atencao)

        ponte = self.cross_attn(y, z, z, mask_encoder)
        y = self.add_norm_2(y, ponte)

        saida_ffn = self.ffn(y)
        y = self.add_norm_3(y, saida_ffn)

        return y

class Decoder(nn.Module):
    def __init__(self, num_layers, d_model, n_heads, d_ff, vocab_size, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList([
            DecoderBlock(d_model, n_heads, d_ff, dropout) 
            for _ in range(num_layers)
        ])
        
        self.linear_saida = nn.Linear(d_model, vocab_size)

    def forward(self, y, z, mask_causal=None, mask_encoder=None):
        for layer in self.layers:
            y = layer(y, z, mask_causal, mask_encoder)
        
        return self.linear_saida(y)



def rodar_teste_decoder():
    d_model, n_heads, d_ff = 512, 8, 2048
    num_camadas = 6
    vocab_size = 5000 #
    
    decoder = Decoder(num_camadas, d_model, n_heads, d_ff, vocab_size)
    
    y = torch.randn(2, 5, d_model)  
    z = torch.randn(2, 10, d_model) 
    
    probabilidades = decoder(y, z)
    
    print(f"Entrada Alvo (Y): {y.shape}")
    print(f"Memória Encoder (Z): {z.shape}")
    print(f"Saída Probabilidades: {probabilidades.shape}") 
    print("Sucesso: O Decoder gerou as probabilidades para o vocabulário!")

if __name__ == "__main__":
    rodar_teste_decoder()
