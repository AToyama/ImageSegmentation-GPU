import os
os.system("nvcc --std=c++11 main_seq.cu imagem.cpp -o segmentacao_seq")
entrada = input("\nImagem de entrada:\n")
saida = input("\nImagem de saida:\n")
os.system("./segmentacao_seq "+entrada+" "+saida)