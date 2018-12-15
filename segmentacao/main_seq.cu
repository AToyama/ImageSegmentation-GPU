#include <iostream>
#include <queue>
#include <vector>
#include <assert.h>
#include <fstream>
#include <cuda_runtime.h>

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include "imagem.h"

#define MAX(y,x) (y>x?y:x)    // Calcula valor maximo
#define MIN(y,x) (y<x?y:x)    // Calcula valor minimo

typedef std::pair<double, int> custo_caminho;

typedef std::pair<double *, int *> result_sssp;


struct compare_custo_caminho {
    bool operator()(custo_caminho &c1, custo_caminho &c2) {
        return c2.first < c1.first;
    }
};


// FILTRO DE BORDAS
__global__ void edgeFilter(unsigned char *in, unsigned char *out, int rowEnd, int colEnd) {
    
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    int j=blockIdx.y*blockDim.y+threadIdx.y;

    int rowStart = 0, colStart = 0;

    int di,dj;
    
    if(i < rowEnd && j < colEnd){

        int min = 256;
        int max = 0;
        for(di = MAX(rowStart, i - 1); di <= MIN(i + 1, rowEnd - 1); di++) {
            for(dj = MAX(colStart, j - 1); dj <= MIN(j + 1, colEnd - 1); dj++) {
                if(min>in[di*(colEnd-colStart)+dj]) min = in[di*(colEnd-colStart)+dj];
                if(max<in[di*(colEnd-colStart)+dj]) max = in[di*(colEnd-colStart)+dj]; 
            }
        }
        out[i*(colEnd-colStart)+j] = max-min;
    }
}

//entrada img e semente
result_sssp SSSP(imagem *img, std::vector<int> source) {

    std::priority_queue<custo_caminho, std::vector<custo_caminho>, compare_custo_caminho > Q;
    
    double *custos = new double[img->total_size];
    int *predecessor = new int[img->total_size];
    bool *analisado = new bool[img->total_size];

    // criar obj de saida 
    result_sssp res(custos, predecessor);
    
 
    for (int i = 0; i < img->total_size; i++) {

        predecessor[i] =-1;
        custos[i] = __DBL_MAX__;
        analisado[i] = false;
    }

    
    for(int i = 0; i < source.size(); i++)
    {
        Q.push(custo_caminho(0.0, source[i]));
        predecessor[source[i]] = source[i];
        custos[source[i]] = 0.0;   
    }

    while (!Q.empty()) {

        //pair<double, int> custo caminho
        custo_caminho cm = Q.top();
        Q.pop();

        // seed
        int vertex = cm.second;

        if (analisado[vertex]) continue; // já tem custo mínimo calculado
        analisado[vertex] = true;

        //         
        double custo_atual = cm.first;
        assert(custo_atual == custos[vertex]);

        // pixel de cima
        int acima = vertex - img->cols;
        if (acima >= 0) {   // se existir pixel acima na img
            // custo do pixel = custo atual da seed + | seed - acima |
            double custo_acima = custo_atual + get_edge(img, vertex, acima);
            if (custo_acima < custos[acima]) { // se for  caminho de menor valor
                custos[acima] = custo_acima;
                // coloca o  pixel acima na fila como prox seed
                Q.push(custo_caminho(custo_acima, acima));
                // link do pixel acima com a seed atual
                predecessor[acima] = vertex;
            }
        }
       // pixel de baixo
        int abaixo = vertex + img->cols;
        if (abaixo < img->total_size) { // se existir pixel abaixo na img
            double custo_abaixo = custo_atual + get_edge(img, vertex, abaixo);
            if (custo_abaixo < custos[abaixo]) {
                custos[abaixo] = custo_abaixo;
                Q.push(custo_caminho(custo_abaixo, abaixo));
                predecessor[abaixo] = vertex;
            }
        }

        // pixel da direita
        int direita = vertex + 1;
        if (direita < img->total_size) { // se existir pixel direito na img
            double custo_direita = custo_atual + get_edge(img, vertex, direita);
            if (custo_direita < custos[direita]) {
                custos[direita] = custo_direita;
                Q.push(custo_caminho(custo_direita, direita));
                predecessor[direita] = vertex;
            }
        }
        // pixel da esquerda
        int esquerda = vertex - 1;
        if (esquerda >= 0) { // se existir pixel esuqerda na img
            double custo_esquerda = custo_atual + get_edge(img, vertex, esquerda);
            if (custo_esquerda < custos[esquerda]) {
                custos[esquerda] = custo_esquerda;
                Q.push(custo_caminho(custo_esquerda, esquerda));
                predecessor[esquerda] = vertex;
            }
        }
    }
    
    delete[] analisado;
    
    return res;
}


int main(int argc, char **argv) {

    if (argc < 3) {
        std::cout << "Uso:  segmentacao_sequencial entrada.pgm saida.pgm\n";
        return -1;
    }

    std::string path(argv[1]);
    std::string path_output(argv[2]);
    imagem *img = read_pgm(path);
    
    int nrows = img->rows;
    int ncols = img->cols;  

    int n_fg, n_bg;
    int x, y;
    
    float total_time, graph_time, sssp_time, output_time;

    // variaveis de contagem de tempo
    cudaEvent_t total_begin, total_end, begin, end;
    cudaEventCreate(&total_begin);
    cudaEventCreate(&total_end);
    cudaEventCreate(&begin);
    cudaEventCreate(&end);

    // inicio da contagem de tempo total do programa
    cudaEventRecord(total_begin);

    // numero de sementes de frente e de fundo
    std::cout << "numero de sementes de frente e de fundo:\n";
    std::cin >> n_fg >> n_bg;

    std::cout << "posições das sementes de frente:\n";

    //posição das sementes de frentes
    std::vector<int> seed_fg(n_fg);

    for(int i = 0; i < n_fg; i++)
    {
        std::cin >> x >> y;
        seed_fg[i] = y * img->cols + x;
    }

    std::cout << "posições das sementes de fundo:\n";

    //posição das sementes de fundo
    std::vector<int> seed_bg(n_bg);

    for(int i = 0; i < n_bg; i++)
    {
        std::cin >> x >> y;
        seed_bg[i] = y * img->cols + x;    
    }
    

    //FILTRO DE BORDAS
    imagem *edge = new_image(nrows, ncols);

    thrust::device_vector<unsigned char> input(img->pixels, img->pixels + img->total_size );
    thrust::device_vector<unsigned char> output(edge->pixels, edge->pixels + edge->total_size );

    dim3 dimGrid(ceil(nrows/16.0), ceil(ncols/16.0), 1);
    dim3 dimBlock(16, 16, 1);

    edgeFilter<<<dimGrid,dimBlock>>>(thrust::raw_pointer_cast(input.data()), thrust::raw_pointer_cast(output.data()), nrows, ncols);

    thrust::host_vector<unsigned char> output_data(output);
    for(int i = 0; i != output_data.size(); i++) {
        edge->pixels[i] = output_data[i];
    }

    write_pgm(edge, "edge.pgm");

    //inicio da contagem de tempo do sssp
    cudaEventRecord(begin);

    // retorna pair<double *, int *> result_sssp; -> <custos, predecessores>
    result_sssp fg = SSSP(img, seed_fg);
    result_sssp bg = SSSP(img, seed_bg);

    //fim da contagem
    cudaEventRecord(end);
    cudaEventSynchronize(end);
    cudaEventElapsedTime(&sssp_time, begin, end);

    //inicio da contagem de tempo da construção da imagem de saida
    cudaEventRecord(begin);

    // começo da geração da saida
    imagem *saida = new_image(img->rows, img->cols);
    
    // p/ cada pixel
    for (int k = 0; k < saida->total_size; k++) {
        
        //se o custo do pixel até a semente de fundo for maior que a da frente, deixa o pixel preto
        if (fg.first[k] > bg.first[k]) {
            saida->pixels[k] = 0;
        // senão ele pertence a semente da frente, deixe branco
        } else {
            saida->pixels[k] = 255;
        }
    }
    
    //escrita da imagem de saida
    write_pgm(saida, path_output); 
       
    //fim da contagem
    cudaEventRecord(end);
    cudaEventSynchronize(end);
    cudaEventElapsedTime(&output_time, begin, end);    

    //fim da contagem de tempo total
    cudaEventRecord(total_end);
    cudaEventSynchronize(total_end);
    cudaEventElapsedTime(&total_time, total_begin, total_end);

    cudaEventDestroy(begin);
    cudaEventDestroy(end);
    cudaEventDestroy(total_begin);
    cudaEventDestroy(total_end);

    std::cout << "graph_time: " << graph_time <<  "\n";
    std::cout << "sssp_time: " << sssp_time << "\n";
    std::cout << "output_time:  " << output_time <<  "\n";
    std::cout << "total_time: " << total_time <<  "\n";

    return 0;
}
