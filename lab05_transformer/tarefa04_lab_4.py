import torch
import torch.nn as nn
from tarefa02_lab_4 import Encoder
from tarefa03_lab_4 import Decoder
from mask import make_causal_mask
from embedding import TransformerEmbedding 

class TransformerCompleto(nn.Module):
    def __init__(self, n_layers, d_model, n_heads, d_ff, vocab_size):
        super().__init__()
        self.embedding = TransformerEmbedding(vocab_size, d_model)
        self.encoder = Encoder(n_layers, d_model, n_heads, d_ff)
        self.decoder = Decoder(n_layers, d_model, n_heads, d_ff, vocab_size)

    def encode(self, src_ids):
        x = self.embedding(src_ids)
        return self.encoder(x)

    def decode(self, tgt_ids, z, tgt_mask):
        y = self.embedding(tgt_ids)
        return self.decoder(y, z, mask_causal=tgt_mask)

def executar_prova_final():
    d_model, n_heads, d_ff, n_layers = 512, 8, 2048, 6
    vocab_size = 5000
    max_len = 10
    
    vocab = {"Thinking": 10, "Machines": 11, "<START>": 1, "<EOS>": 2}
    
    model = TransformerCompleto(n_layers, d_model, n_heads, d_ff, vocab_size)
    model.eval()

    src_indices = torch.tensor([[vocab["Thinking"], vocab["Machines"]]]) 
    decoder_input_indices = torch.tensor([[vocab["<START>"]]]) 
    lista_indices_gerados = [vocab["<START>"]]

    print(f"Entrada (IDs): {src_indices.tolist()} -> 'Thinking Machines'")
    print("Iniciando loop auto-regressivo...\n")

    with torch.no_grad():
        z = model.encode(src_indices)

        while len(lista_indices_gerados) < max_len:
            tgt_len = decoder_input_indices.size(1)
            tgt_mask = make_causal_mask(tgt_len, decoder_input_indices.device)

            logits = model.decode(decoder_input_indices, z, tgt_mask)
            
            next_idx = torch.argmax(logits[0, -1, :]).item()
            lista_indices_gerados.append(next_idx)

            print(f"Passo {len(lista_indices_gerados)-1}: Previu ID {next_idx}")

            if next_idx == vocab["<EOS>"]:
                print("Token <EOS> detectado.")
                break

            next_tensor = torch.tensor([[next_idx]])
            decoder_input_indices = torch.cat([decoder_input_indices, next_tensor], dim=1)

    print(f"Resultado Final: {lista_indices_gerados}")
    print("Nota: Pesos aleatórios detectados. Os IDs previstos não formarão frases reais até o treino.")

if __name__ == "__main__":
    executar_prova_final()
