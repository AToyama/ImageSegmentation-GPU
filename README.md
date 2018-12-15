# Segmentação de Imagem na GPU
Segmentador de Imagem em c++ com Nvgraph
Projeto da matéria de supercomputação
INSPER

## Projeto

O projeto trata-se de um segmentador de imagem na GPU utilzando a ferramenta da nvidia Nvgraph para o [algoritmo de caminho mínimo](https://pt.wikipedia.org/wiki/Algoritmo_de_Dijkstra) utilizado no processo de segmentação e a API da Nvidia de computação paralela para realizar o filtro de bordas na imagem.

## Dependências

* [nvgraph](https://docs.nvidia.com/cuda/nvgraph/index.html) - Fazer o download das páginas HTML
* [CUDA](https://devblogs.nvidia.com/even-easier-introduction-cuda/) - Realizar o parsing dos HTML

## Rodando o Programa

Após ter clonado o repositório e instalado as dependências, você terá dentro da pasta "./segmentacao" os arquivos "runSeq.py" e "runNvgraph.py", destinados respectivamente a execução sequencial do programa e a execuço com nvgraph. Para compilar e rodar os programas, basta seguir o seguinte formato no seu terminal com um dos dois arquivos:

```
$ python [ nome do arquivo ]>
```

## Realizaço dos testes:

No repositório temos as seguintes imagens de teste

- teste1.pgm (10x10)
- teste.pgm (500x500)
- lemon.pgm (5000X3333)

Quando o programa pedir uma imagem de entrada basta insirar uma delas, já para basta qualquer nome com a extensão .pgm que o programa ira gerar a imagem de saida.

Após executar seu programa com suas coordenadas de sementes desejadas para eralizar a segmentação, o programa retornara as seguintes medidas de tempo:

- Tempo para construção do Grafo (apenas para a versão nvgraph)
- Tempo para executar a função sssp
- Tempo para geração da imagem de sada
- Tempo de execução total do programa
