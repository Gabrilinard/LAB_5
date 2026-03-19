from datasets import load_dataset
from typing import List, Dict

TAMANHO_AMOSTRA_PADRAO = 200
NOME_DATASET = "bentrevett/multi30k"

def buscar_dados_traducao(limite: int = TAMANHO_AMOSTRA_PADRAO) -> List[Dict[str, str]]:
    print(f"Iniciando download do dataset '{NOME_DATASET}'...")
    
    try:
        dados_brutos = load_dataset(NOME_DATASET, split="train")
        
        subconjunto_amostra = dados_brutos.select(range(limite))
        
        pares_traducao = [
            {"ingles": registro["en"], "alemao": registro["de"]} 
            for registro in subconjunto_amostra
        ]

        print(f"Sucesso! {len(pares_traducao)} pares selecionados.")
        return pares_traducao

    except Exception as erro:
        print(f"Erro ao carregar os dados: {erro}")
        return []

def exibir_previa(lista_dados: List[Dict[str, str]], qtd_previa: int = 3):
    print(f"\n--- Prévia dos Primeiros {qtd_previa} Pares ---")
    for indice, par in enumerate(lista_dados[:qtd_previa], 1):
        print(f"Par #{indice}:")
        print(f"  🇬🇧 EN: {par['ingles']}")
        print(f"  🇩🇪 DE: {par['alemao']}\n")

if __name__ == "__main__":
    previa_dataset = buscar_dados_traducao()
    
    if previa_dataset:
        exibir_previa(previa_dataset)