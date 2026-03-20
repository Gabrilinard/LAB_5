import torch
import torch.nn as nn

from tarefa04_lab_4 import TransformerCompleto
from mask import make_causal_mask
from tarefa01_dataset import buscar_dados_traducao
from tarefa02_tokenizacao import preparar_tokenizador, transformar_texto_em_numeros

DIMENSAO_MODELO    = 32
CABECAS_ATENCAO    = 4
CAMADA_ESCONDIDA   = 64
NUMERO_CAMADAS     = 2
TAMANHO_MAXIMO     = 32
ID_PREENCHIMENTO   = 0
EPOCAS_TESTE       = 1000
TAXA_APRENDIZADO   = 1e-3
TAMANHO_MINI_DATASET = 8

def testar_aprendizado_e_gerar():
    dispositivo = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizador = preparar_tokenizador()
    TAMANHO_VOCABULARIO = tokenizador.vocab_size

    pares_exemplo = buscar_dados_traducao(limite=TAMANHO_MINI_DATASET)
    entrada_ingles, entrada_alemao, alvos_esperados = transformar_texto_em_numeros(
        pares_exemplo, tokenizador, limite_tamanho=TAMANHO_MAXIMO
    )

    entrada_ingles = entrada_ingles.to(dispositivo)
    entrada_alemao = entrada_alemao.to(dispositivo)
    alvos_esperados = alvos_esperados.to(dispositivo)

    modelo = TransformerCompleto(
        n_layers   = NUMERO_CAMADAS,
        d_model    = DIMENSAO_MODELO,
        n_heads    = CABECAS_ATENCAO,
        d_ff       = CAMADA_ESCONDIDA,
        vocab_size = TAMANHO_VOCABULARIO,
    ).to(dispositivo)

    parametros_treinaveis = sum(p.numel() for p in modelo.parameters() if p.requires_grad)
    print(f"Modelo criado com {parametros_treinaveis:,} parametros treinaveis.")

    criterio_erro = nn.CrossEntropyLoss(ignore_index=ID_PREENCHIMENTO)
    otimizador    = torch.optim.Adam(modelo.parameters(), lr=TAXA_APRENDIZADO)

    print(f"\nIniciando Overfitting em {TAMANHO_MINI_DATASET} frases por {EPOCAS_TESTE} epocas...")
    print(f"{'Epoca':>6} | {'Erro (Loss)':>10}")
    print("-" * 25)

    for epoca in range(1, EPOCAS_TESTE + 1):
        modelo.train()

        comprimento_sequencia = entrada_alemao.size(1)
        mascara_causal        = make_causal_mask(comprimento_sequencia, dispositivo)

        memoria_do_encoder = modelo.encode(entrada_ingles)
        predicoes          = modelo.decode(entrada_alemao, memoria_do_encoder, mascara_causal)

        logits = predicoes.transpose(1, 2)
        erro   = criterio_erro(logits, alvos_esperados)

        otimizador.zero_grad()
        erro.backward()
        torch.nn.utils.clip_grad_norm_(modelo.parameters(), max_norm=1.0)
        otimizador.step()

        if epoca % 100 == 0 or epoca == 1:
            print(f"{epoca:>6} | {erro.item():>10.4f}")

    print(f"\nErro final: {erro.item():.4f}")

    indice_frase    = 0
    ingles_original = pares_exemplo[indice_frase]["ingles"]
    alemao_esperado = pares_exemplo[indice_frase]["alemao"]

    print(f"\nProva de Fogo --")
    print(f"  Frase EN (entrada) : '{ingles_original}'")
    print(f"  Traducao Real (DE) : '{alemao_esperado}'")

    modelo.eval()
    with torch.no_grad():
        token_ingles = entrada_ingles[indice_frase].unsqueeze(0)
        memoria_z    = modelo.encode(token_ingles)

        id_inicio = tokenizador.cls_token_id
        id_fim    = tokenizador.sep_token_id

        contexto_decoder = torch.tensor([[id_inicio]], device=dispositivo)
        tokens_gerados   = [id_inicio]

        for _ in range(TAMANHO_MAXIMO):
            tamanho_atual = contexto_decoder.size(1)
            mascara       = make_causal_mask(tamanho_atual, dispositivo)

            distribuicao_probs = modelo.decode(contexto_decoder, memoria_z, mascara)
            proximo_token_id   = torch.argmax(distribuicao_probs[0, -1, :]).item()

            tokens_gerados.append(proximo_token_id)

            if proximo_token_id == id_fim:
                break

            proximo_token_tensor = torch.tensor([[proximo_token_id]], device=dispositivo)
            contexto_decoder     = torch.cat([contexto_decoder, proximo_token_tensor], dim=1)

    ids_limpos = [t for t in tokens_gerados if t not in (id_inicio, id_fim, ID_PREENCHIMENTO)]
    traducao_final = tokenizador.decode(ids_limpos, skip_special_tokens=True)

    print(f"  Traducao do Modelo : '{traducao_final}'")

if __name__ == "__main__":
    testar_aprendizado_e_gerar()