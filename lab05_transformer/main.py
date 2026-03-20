import argparse
import torch


def executar_tarefa1():
    from tarefa01_dataset import buscar_dados_traducao
    print("=" * 60)
    print("TAREFA 1 - Dataset")
    print("=" * 60)
    pares = buscar_dados_traducao()
    print(f"\n{len(pares)} pares EN-DE carregados com sucesso.\n")
    return pares


def executar_tarefa2(pares):
    from tarefa02_tokenizacao import preparar_tokenizador, transformar_texto_em_numeros
    print("=" * 60)
    print("TAREFA 2 - Tokenizacao")
    print("=" * 60)
    tokenizador = preparar_tokenizador()
    src, tgt, lbl = transformar_texto_em_numeros(pares, tokenizador)
    print("\nTokenizacao concluida.\n")
    return src, tgt, lbl


def executar_tarefa3(src, tgt, lbl):
    from tarefa03_training_loop import executar_treinamento
    print("=" * 60)
    print("TAREFA 3 - Training Loop")
    print("=" * 60)
    modelo, historico = executar_treinamento(src, tgt, lbl)
    torch.save(modelo.state_dict(), "modelo_treinado.pt")
    print("\nPesos salvos em modelo_treinado.pt\n")
    return modelo


def executar_tarefa4():
    from tarefa04_overfitting import testar_aprendizado_e_gerar
    print("=" * 60)
    print("TAREFA 4 - Prova de Fogo (Overfitting Test)")
    print("=" * 60)
    testar_aprendizado_e_gerar()
    print("\nOverfitting test concluido.\n")


def main():
    parser = argparse.ArgumentParser(description="Lab 05 - Transformer End-to-End")
    parser.add_argument(
        "--tarefa", type=int, choices=[1, 2, 3, 4],
        help="Executa apenas a tarefa especificada (1-4). Padrao: todas."
    )
    args = parser.parse_args()

    if args.tarefa is None or args.tarefa == 1:
        pares = executar_tarefa1()

    if args.tarefa is None or args.tarefa == 2:
        if args.tarefa == 2:
            from tarefa01_dataset import buscar_dados_traducao
            pares = buscar_dados_traducao()
        src, tgt, lbl = executar_tarefa2(pares)

    if args.tarefa is None or args.tarefa == 3:
        if args.tarefa == 3:
            from tarefa01_dataset import buscar_dados_traducao
            from tarefa02_tokenizacao import preparar_tokenizador, transformar_texto_em_numeros
            pares = buscar_dados_traducao()
            tok = preparar_tokenizador()
            src, tgt, lbl = transformar_texto_em_numeros(pares, tok)
        executar_tarefa3(src, tgt, lbl)

    if args.tarefa is None or args.tarefa == 4:
        executar_tarefa4()

    print("=" * 60)
    print("Laboratorio 5 concluido!")
    print("=" * 60)


if __name__ == "__main__":
    main()