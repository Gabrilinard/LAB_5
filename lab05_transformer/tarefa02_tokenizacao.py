import torch
from transformers import AutoTokenizer
from tarefa01_dataset import buscar_dados_traducao

NOME_DO_MODELO = "bert-base-multilingual-cased"
ID_DO_PREENCHIMENTO = 0  
TAMANHO_MAXIMO = 64      

def preparar_tokenizador():
    print(f"Carregando o dicionário do modelo: {NOME_DO_MODELO}...")
    ferramenta_tokenizar = AutoTokenizer.from_pretrained(NOME_DO_MODELO)
    
    tamanho_vocabulario = ferramenta_tokenizar.vocab_size
    print(f"Dicionário carregado com {tamanho_vocabulario} palavras.")
    
    return ferramenta_tokenizar


def transformar_texto_em_numeros(lista_de_pares, tokenizador, limite_tamanho = TAMANHO_MAXIMO):
    print(f"Processando {len(lista_de_pares)} frases...")

    lista_entrada_ingles = []    
    lista_entrada_alemao = []   
    lista_saida_esperada = []    

    id_inicio = tokenizador.cls_token_id 
    id_fim    = tokenizador.sep_token_id  

    for par in lista_de_pares:
        ingles_processado = tokenizador(
            par["ingles"],
            max_length=limite_tamanho,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        lista_entrada_ingles.append(ingles_processado["input_ids"].squeeze(0))

        numeros_alemao = tokenizador(
            par["alemao"],
            add_special_tokens=False,
            truncation=True,
            max_length=limite_tamanho - 2, 
        )["input_ids"]

        entrada_alemao_com_start = [id_inicio] + numeros_alemao
        saida_alemao_com_end     = numeros_alemao + [id_fim]

        def ajustar_tamanho(sequencia):
            sequencia = sequencia[:limite_tamanho]
            faltando = limite_tamanho - len(sequencia)
            return sequencia + [ID_DO_PREENCHIMENTO] * faltando

        lista_entrada_alemao.append(torch.tensor(ajustar_tamanho(entrada_alemao_com_start), dtype=torch.long))
        lista_saida_esperada.append(torch.tensor(ajustar_tamanho(saida_alemao_com_end),     dtype=torch.long))

    tensores_ingles = torch.stack(lista_entrada_ingles)
    tensores_alemao = torch.stack(lista_entrada_alemao)
    tensores_alvo   = torch.stack(lista_saida_esperada)

    print(f"Formato final dos dados:")
    print(f"   Entrada (EN): {tensores_ingles.shape}")
    print(f"   Entrada (DE): {tensores_alemao.shape}")
    print(f"   Alvo (Labels): {tensores_alvo.shape}")
    
    return tensores_ingles, tensores_alemao, tensores_alvo


if __name__ == "__main__":
    meu_tokenizador = preparar_tokenizador()
    meus_pares = buscar_dados_traducao(limite=10)
    
    en_ids, de_ids, labels = transformar_texto_em_numeros(meus_pares, meu_tokenizador)

    print("\n Verificando o primeiro par processado:")
    print("   IDs de Entrada (EN):", en_ids[0][:10].tolist(), "...")
    print("   IDs do Decoder (DE):", de_ids[0][:10].tolist(), "...")
    print("   IDs de Resposta (DE):", labels[0][:10].tolist(), "...")