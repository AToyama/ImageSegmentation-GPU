import os
os.system("nvcc --std=c++11 main_cuda.cu imagem.cpp -o segmentacao_cuda -lnvgraph")
entrada = input("\nImagem de entrada:\n")
saida = input("\nImagem de saida:\n")
os.system("./segmentacao_cuda "+entrada+" "+saida)