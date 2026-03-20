import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from tarefa04_lab_4 import TransformerCompleto
from mask import make_causal_mask
from tarefa01_dataset import buscar_dados_traducao
from tarefa02_tokenizacao import preparar_tokenizador, transformar_texto_em_numeros

DIMENSAO_MODELO    = 128   
CABECAS_ATENCAO    = 4     
CAMADA_ESCONDIDA   = 256   
NUMERO_CAMADAS     = 2    
TAMANHO_MAXIMO     = 64    
TAMANHO_LOTE       = 64    
EPOCAS             = 10   
TAXA_APRENDIZADO   = 1e-3  
ID_PREENCHIMENTO   = 0     
TOTAL_DE_AMOSTRAS  = 200
VOCAB_SIZE_LIMITE  = 10000


def organizar_dados_para_treino():
    tokenizador = preparar_tokenizador()
    lista_de_pares = buscar_dados_traducao(limite=TOTAL_DE_AMOSTRAS)
    entrada_en, entrada_de, alvos = transformar_texto_em_numeros(
        lista_de_pares, tokenizador, limite_tamanho=TAMANHO_MAXIMO
    )
    return entrada_en, entrada_de, alvos, tokenizador


def executar_treinamento(entrada_en, entrada_de, alvos):
    dispositivo = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n Treinando no dispositivo: {dispositivo}")

    modelo = TransformerCompleto(
        n_layers   = NUMERO_CAMADAS,
        d_model    = DIMENSAO_MODELO,
        n_heads    = CABECAS_ATENCAO,
        d_ff       = CAMADA_ESCONDIDA,
        vocab_size = VOCAB_SIZE_LIMITE,
    ).to(dispositivo)

    total_parametros = sum(p.numel() for p in modelo.parameters() if p.requires_grad)
    print(f"O modelo possui {total_parametros:,} conexões treináveis.")

    funcao_erro = nn.CrossEntropyLoss(ignore_index=ID_PREENCHIMENTO)
    otimizador  = torch.optim.Adam(modelo.parameters(), lr=TAXA_APRENDIZADO)

    dataset    = TensorDataset(entrada_en, entrada_de, alvos)
    carregador = DataLoader(dataset, batch_size=TAMANHO_LOTE, shuffle=True)

    print(f"Iniciando: {EPOCAS} épocas | {len(carregador)} lotes por época\n")
    print(f"{'Época':>6} | {'Erro Médio (Loss)':>18}")

    historico_de_erros = []

    for epoca in range(1, EPOCAS + 1):
        modelo.train()
        soma_erro_epoca = 0.0

        for lote_en, lote_de, lote_alvo in carregador:
            lote_en   = lote_en.clamp(max=VOCAB_SIZE_LIMITE - 1)
            lote_de   = lote_de.clamp(max=VOCAB_SIZE_LIMITE - 1)
            lote_alvo = lote_alvo.clamp(max=VOCAB_SIZE_LIMITE - 1)

            comprimento_alvo = lote_de.size(1)
            mascara_causal   = make_causal_mask(comprimento_alvo, dispositivo)

            memoria_z = modelo.encode(lote_en)
            previsoes = modelo.decode(lote_de, memoria_z, mascara_causal)

            logits = previsoes.transpose(1, 2)
            erro   = funcao_erro(logits, lote_alvo)

            otimizador.zero_grad()
            erro.backward()
            torch.nn.utils.clip_grad_norm_(modelo.parameters(), max_norm=1.0)
            otimizador.step()

            soma_erro_epoca += erro.item()

        erro_final_epoca = soma_erro_epoca / len(carregador)
        historico_de_erros.append(erro_final_epoca)
        print(f"{epoca:>6} | {erro_final_epoca:>18.4f}")

    print("\n Treinamento concluído com sucesso!")
    melhoria = (historico_de_erros[0] - historico_de_erros[-1]) / historico_de_erros[0] * 100
    print(f"O erro caiu de {historico_de_erros[0]:.4f} para {historico_de_erros[-1]:.4f} ({melhoria:.1f}% de aprendizado)")

    return modelo, historico_de_erros


if __name__ == "__main__":
    en, de, alvos, _ = organizar_dados_para_treino()
    modelo_final, historico = executar_treinamento(en, de, alvos)
    torch.save(modelo_final.state_dict(), "modelo_treinado.pt")
    print("\n O conhecimento do modelo foi salvo em 'modelo_treinado.pt'!")