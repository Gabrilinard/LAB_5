# Laboratório 5 — Treinamento Fim-a-Fim do Transformer

Projeto final da **Unidade I** — conecta a arquitetura Transformer construída nos Labs anteriores a um dataset real do Hugging Face e implementa o loop completo de Forward → Loss → Backward → Step.

---

## Índice

- [Pré-requisitos](#pré-requisitos)
- [Instalação](#instalação)
- [Estrutura de Arquivos](#estrutura-de-arquivos)
- [Arquitetura do Modelo](#arquitetura-do-modelo)
- [Como Executar](#como-executar)
- [Tarefa 1 — Dataset](#tarefa-1--dataset-hugging-face)
- [Tarefa 2 — Tokenização](#tarefa-2--tokenização-básica)
- [Tarefa 3 — Training Loop](#tarefa-3--training-loop)
- [Tarefa 4 — Prova de Fogo](#tarefa-4--prova-de-fogo-overfitting-test)
- [Ferramentas de IA Utilizadas](#ferramentas-de-ia-utilizadas)
- [Observação sobre Poder Computacional](#observação-sobre-poder-computacional)

---

## Pré-requisitos

- Python 3.9 ou superior
- pip atualizado
- Acesso à internet (para baixar o dataset e o tokenizador na primeira execução)

---

## Instalação

### 1. Clone o repositório

```bash
git clone <url-do-repositorio>
cd lab05_transformer
```

### 2. (Opcional) Crie um ambiente virtual

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Linux / macOS
source venv/bin/activate
```

### 3. Instale as dependências

```bash
pip install -r requirements.txt
```

O arquivo `requirements.txt` contém:

```
torch>=2.0.0
transformers>=4.40.0
datasets>=2.19.0
```

- **torch** — framework de deep learning, responsável por todos os tensores, gradientes e operações matriciais
- **transformers** — biblioteca da Hugging Face usada para carregar o tokenizador pré-treinado `bert-base-multilingual-cased`
- **datasets** — biblioteca da Hugging Face usada para baixar e manipular o dataset `bentrevett/multi30k`

---

## Estrutura de Arquivos

```
lab05_transformer/
│
├── __init__.py                   # Exporta todas as classes do projeto
├── embedding.py                  # PositionalEncoding + TransformerEmbedding
├── mask.py                       # make_causal_mask, make_padding_mask
│
├── tarefa01_lab_4.py             # [Lab4] AtencaoMultihead, FeedForward, AddNorm
├── tarefa02_lab_4.py             # [Lab4] EncoderBlock, Encoder
├── tarefa03_lab_4.py             # [Lab4] DecoderBlock, Decoder
├── tarefa04_lab_4.py             # [Lab4] TransformerCompleto + loop auto-regressivo
│
├── tarefa01_dataset.py           # [Lab5-T1] Carrega dataset multi30k (Hugging Face)
├── tarefa02_tokenizacao.py       # [Lab5-T2] Tokenização + padding + tokens especiais
├── tarefa03_training_loop.py     # [Lab5-T3] Training loop (Adam + CrossEntropyLoss)
├── tarefa04_overfitting.py       # [Lab5-T4] Prova de Fogo (overfitting test)
│
├── main.py                       # Entry point — executa tudo ou por tarefa
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Arquitetura do Modelo

O modelo segue a arquitetura original do paper *"Attention Is All You Need"* (Vaswani et al., 2017), construída progressivamente nos Labs anteriores:

```
Entrada (EN ids)
      │
      ▼
TransformerEmbedding       ← token embedding escalado + positional encoding
      │
      ▼
Encoder (N camadas)
  └── EncoderBlock
        ├── Multi-Head Self-Attention
        └── FeedForward + AddNorm
      │
      ▼ memória Z
      
Entrada (DE ids deslocada)
      │
      ▼
TransformerEmbedding
      │
      ▼
Decoder (N camadas)
  └── DecoderBlock
        ├── Masked Multi-Head Self-Attention  ← máscara causal
        ├── Cross-Attention (Q=decoder, K/V=encoder)
        └── FeedForward + AddNorm
      │
      ▼
Linear (d_model → vocab_size)
      │
      ▼
Logits  →  CrossEntropyLoss durante treino
        →  argmax durante inferência
```

---

## Como Executar

### Executar todas as tarefas em sequência

```bash
python main.py
```

### Executar uma tarefa específica

```bash
python main.py --tarefa 1
python main.py --tarefa 2
python main.py --tarefa 3
python main.py --tarefa 4
```

### Executar arquivos individualmente

```bash
python tarefa01_dataset.py
python tarefa02_tokenizacao.py
python tarefa03_training_loop.py
python tarefa04_overfitting.py
```

---

## Tarefa 1 — Dataset (Hugging Face)

**Arquivo:** `tarefa01_dataset.py`

Carrega o dataset de tradução EN → DE `bentrevett/multi30k` diretamente do Hugging Face Hub. É selecionado um subconjunto de **200 pares** para garantir execução rápida em CPU.

A função principal `buscar_dados_traducao(limite)` retorna uma lista de dicionários com as chaves `ingles` e `alemao`.

### Saída obtida

```
Iniciando download do dataset 'bentrevett/multi30k'...
Sucesso! 200 pares selecionados.

--- Prévia dos Primeiros 3 Pares ---
Par #1:
  EN: Two young, White males are outside near many bushes.
  DE: Zwei junge weiße Männer sind im Freien in der Nähe vieler Büsche.

Par #2:
  EN: Several men in hard hats are operating a giant pulley system.
  DE: Mehrere Männer mit Schutzhelmen bedienen ein Antriebsradsystem.

Par #3:
  EN: A little girl climbing into a wooden playhouse.
  DE: Ein kleines Mädchen klettert in ein Spielhaus aus Holz.
```

---

## Tarefa 2 — Tokenização Básica

**Arquivo:** `tarefa02_tokenizacao.py`

Converte os pares de frases em tensores de inteiros usando o tokenizador `bert-base-multilingual-cased` da Hugging Face. O vocabulário deste tokenizador possui 119.547 palavras/subpalavras, o que o torna adequado para lidar com inglês e alemão simultaneamente.

### Decisões de implementação

- O **encoder** recebe a frase em inglês tokenizada normalmente com `[CLS]` e `[SEP]` automáticos
- O **decoder** recebe a frase em alemão com `[CLS]` (usado como `<START>`) no início, sem `[SEP]` no final — esta é a entrada deslocada
- O **label** (resposta esperada) recebe a frase em alemão sem `[CLS]` no início, com `[SEP]` (usado como `<EOS>`) no final — este é o alvo do modelo
- Todas as sequências são preenchidas com zeros (padding) até `MAX_LEN = 64`

### Hiperparâmetros

| Parâmetro | Valor |
|---|---|
| Tokenizador | `bert-base-multilingual-cased` |
| Tamanho máximo | 64 tokens |
| Token `<START>` | ID do `[CLS]` = 101 |
| Token `<EOS>` | ID do `[SEP]` = 102 |
| Padding | 0 |

### Saída obtida

```
Carregando o dicionário do modelo: bert-base-multilingual-cased...
Dicionário carregado com 119547 palavras.
Processando 10 frases...
Formato final dos dados:
   Entrada (EN): torch.Size([10, 64])
   Entrada (DE): torch.Size([10, 64])
   Alvo (Labels): torch.Size([10, 64])

Verificando o primeiro par processado:
   IDs de Entrada (EN): [101, 13214, 14739, 117, 12136, 24617, 10301, 17555, 12883, 11299] ...
   IDs do Decoder (DE): [101, 29556, 45106, 85916, 30894, 10762, 10211, 76700, 10106, 10118] ...
   IDs de Resposta (DE): [29556, 45106, 85916, 30894, 10762, 10211, 76700, 10106, 10118, 25672] ...
```

Os IDs do Decoder começam com `101` (`<START>`) e os Labels começam sem ele, deslocados em uma posição — o modelo aprende a prever o próximo token a cada passo.

---

## Tarefa 3 — Training Loop

**Arquivo:** `tarefa03_training_loop.py`

Implementa o loop completo de treinamento supervisionado: Forward → Loss → Backward → Step. O modelo é instanciado com dimensões reduzidas para viabilizar execução em CPU, e treinado por 10 épocas sobre 200 pares de frases.

### Hiperparâmetros

| Parâmetro | Valor |
|---|---|
| `d_model` | 128 |
| `n_heads` | 4 |
| `d_ff` | 256 |
| `n_layers` | 2 |
| `batch_size` | 64 |
| `epochs` | 10 |
| `lr` | 1e-3 |
| Otimizador | Adam |
| Perda | CrossEntropyLoss (`ignore_index=0`) |
| Amostras | 200 pares |
| `vocab_size` | 10.000 |

### Detalhes do fluxo

1. A frase em inglês passa pelo `Encoder`, gerando a memória `Z`
2. A frase em alemão deslocada passa pelo `Decoder` junto com `Z`, gerando os logits
3. Os logits são comparados com o label (resposta esperada) via `CrossEntropyLoss`
4. O `ignore_index=0` garante que tokens de padding não penalizam o modelo
5. `loss.backward()` calcula os gradientes e `optimizer.step()` atualiza os pesos
6. `clip_grad_norm_` com `max_norm=1.0` evita explosão de gradientes

### Saída obtida

```
Treinando no dispositivo: cpu
O modelo possui 3.232.528 conexões treináveis.
Iniciando: 10 épocas | 4 lotes por época

 Época |  Erro Médio (Loss)
     1 |             6.0691
     2 |             3.3492
     3 |             2.5768
     4 |             1.8472
     5 |             1.3223
     6 |             0.9169
     7 |             0.6888
     8 |             0.5915
     9 |             0.5460
    10 |             0.5136

Treinamento concluído com sucesso!
O erro caiu de 6.0691 para 0.5136 (91.5% de aprendizado)
```

O Loss caiu **91.5%** entre a primeira e a última época, demonstrando que os gradientes estão fluindo corretamente por toda a arquitetura.

---

## Tarefa 4 — Prova de Fogo (Overfitting Test)

**Arquivo:** `tarefa04_overfitting.py`

Técnica clássica de debugging de redes neurais: forçar o modelo a memorizar um conjunto mínimo de dados para provar que a arquitetura e o fluxo de gradientes estão corretos. Um modelo pequeno é treinado por **1000 épocas** sobre apenas **8 frases**, e ao final o loop auto-regressivo é usado para gerar a tradução de uma das frases vistas no treino.

### Hiperparâmetros

| Parâmetro | Valor |
|---|---|
| `d_model` | 32 |
| `n_heads` | 4 |
| `d_ff` | 64 |
| `n_layers` | 2 |
| `epochs` | 500 |
| `lr` | 1e-3 |
| Frases no mini-dataset | 8 |
| `MAX_LEN` | 32 tokens |

### Loop Auto-regressivo

A geração da tradução funciona token a token:

1. A frase em inglês é codificada pelo Encoder, gerando a memória `Z`
2. O Decoder recebe o token `<START>` e, a cada passo, prevê o próximo token via `argmax`
3. O token previsto é concatenado ao contexto e o passo se repete
4. O loop termina quando o modelo gera `<EOS>` ou atinge o comprimento máximo

### Saída obtida

```
Modelo criado com 7.813.307 parametros treinaveis.

Iniciando Overfitting em 8 frases por 1000 epocas...
 Epoca | Erro (Loss)
-------------------------
    1  |    11.8492
   100 |     4.6949
   200 |     1.3908
   300 |     0.3302
   400 |     0.1256
   500 |     0.0677

Erro final: 0.0677

Prova de Fogo --
  Frase EN (entrada) : 'Two young, White males are outside near many bushes.'
  Traducao Real (DE) : 'Zwei junge weiße Männer sind im Freien in der Nähe vieler Büsche.'
  Traducao do Modelo : 'Zwei junge weiße Männer sind im Freien in der Nähe vieler Büsche.'
```

O modelo reproduziu a tradução **exata** da frase vista no treino, confirmando que a arquitetura assimilou o padrão matricial com sucesso e que os gradientes fluem corretamente por todos os componentes: Embedding → Encoder → Decoder → Linear.

---

## Ferramentas de IA Utilizadas

- **Claude (Anthropic)** — auxiliou na estruturação dos arquivos das Tarefas 1 e 2, na correção de bugs e na escrita do README, conforme permitido pelo enunciado.
- O fluxo de Forward/Backward da Tarefa 3 interage estritamente com as classes construídas nos laboratórios anteriores.

---

## Observação sobre Poder Computacional

Não é objetivo obter um tradutor fluente — o modelo do Google de 2017 treinou por 3,5 dias em 8 GPUs dedicadas. O critério de avaliação é a **queda da curva de Loss** e a **integridade do fluxo de tensores**, ambos demonstrados com sucesso nas saídas acima.