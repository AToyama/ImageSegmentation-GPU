#include <stdlib.h>
#include <stdio.h>
#include <ctype.h>
#include <string.h>
#include <iostream>
#include <queue>
#include <vector>
#include <assert.h>
#include <fstream>
#include <cuda_runtime.h>
#include <algorithm>

#include <nvgraph.h>
#include "imagem.h"
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#define MAX(y,x) (y>x?y:x)    // Calcula valor maximo
#define MIN(y,x) (y<x?y:x)    // Calcula valor minimo

//scp -i supercomp-final.pem -r ./segmentacao ubuntu@ec2-54-144-197-153.compute-1.amazonaws.com:~/toy
//scp -i supercomp-final.pem ./segmentacao/main_cuda.cu ubuntu@ec2-54-144-197-153.compute-1.amazonaws.com:~/toy/segmentacao
//scp -i supercomp-final.pem  ubuntu@ec2-54-144-197-153.compute-1.amazonaws.com:~/toy/segmentacao/saida.pgm ./segmentacao
//ssh -i supercomp-final.pem ubuntu@ec2-54-144-197-153.compute-1.amazonaws.com
//nvcc -std=c++11 imagem.cpp main_cuda.cu -o segmentacao_cuda -lnvgraph

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

// checagem de erros do nvgraph
void check_status(nvgraphStatus_t status)
{
    if ((int)status != 0)
    {
        printf("ERROR : %d\n",status);
        exit(0);
    }
}

// funcão SSSP do nvgraph
int SSSP(int size, int edges, std::vector<float> weights, std::vector<int> source, std::vector<int> dest_offset, int source_vert, std::vector<float> &out) {

    const size_t n = size;
    const size_t nnz = edges;
    float * sssp_1_h;

    int * source_indices_h = (int*) malloc(source.size()*sizeof(int));
    int * destination_offsets_h = (int*) malloc(dest_offset.size()*sizeof(int));
    float * weights_h = (float*)malloc(edges*sizeof(float));

    // conversão dos vetor do graph 
    for (int i = 0; i < source.size(); i++){
        source_indices_h[i] = source[i];
    }
    for (int i = 0; i < weights.size(); i++){
        weights_h[i] = weights[i];
    }
    for (int i = 0; i < dest_offset.size(); i++){
        destination_offsets_h[i] = dest_offset[i];
    }
    
    const size_t vertex_numsets = 1, edge_numsets = 1;
    void** vertex_dim;

    // variaveis nvgraph
    nvgraphStatus_t status;
    nvgraphHandle_t handle;
    nvgraphGraphDescr_t graph;
    nvgraphCSCTopology32I_t CSC_input;
    cudaDataType_t edge_dimT = CUDA_R_32F;
    cudaDataType_t* vertex_dimT;

    //dados de saida
    sssp_1_h = (float*)malloc(size*sizeof(float));

    vertex_dim  = (void**)malloc(vertex_numsets*sizeof(void*));
    vertex_dimT = (cudaDataType_t*)malloc(vertex_numsets*sizeof(cudaDataType_t));
    CSC_input = (nvgraphCSCTopology32I_t) malloc(sizeof(struct nvgraphCSCTopology32I_st));

    vertex_dim[0]= (void*)sssp_1_h;
    //vertex_dim[1];
    vertex_dimT[0] = CUDA_R_32F;
    //vertex_dimT[1]= CUDA_R_32F;  
    
    check_status(nvgraphCreate(&handle));
    check_status(nvgraphCreateGraphDescr (handle, &graph));

    //parametros da montagem do grafo
    CSC_input->nvertices = n;
    CSC_input->nedges = nnz;
    CSC_input->destination_offsets = destination_offsets_h;
    CSC_input->source_indices = source_indices_h;

    // montagem do grafo
    check_status(nvgraphSetGraphStructure(handle, graph, (void*)CSC_input, NVGRAPH_CSC_32));
    check_status(nvgraphAllocateVertexData(handle, graph, vertex_numsets, vertex_dimT));
    check_status(nvgraphAllocateEdgeData  (handle, graph, edge_numsets, &edge_dimT));
    check_status(nvgraphSetEdgeData(handle, graph, (void*)weights_h, 0));
    
    //sssp
    check_status(nvgraphSssp(handle, graph, 0,  &source_vert, 0));

    //pegar dados de saida
    check_status(nvgraphGetVertexData(handle, graph, (void*)sssp_1_h, 0));
    
    for(int i = 0; i < CSC_input->nvertices;i++){
        out[i] = sssp_1_h[i];
    }

    //desalocando dados auxiliares
    free(destination_offsets_h);
    free(source_indices_h);
    free(weights_h);
    free(sssp_1_h);
    free(vertex_dim);
    free(vertex_dimT);
    free(CSC_input);
    
    //destroi o grafo
    check_status(nvgraphDestroyGraphDescr (handle, graph));
    check_status(nvgraphDestroy (handle));
    
    return 0;
}

// GERAÇÃO DOS VETORES
void vectorsGen(imagem *img, std::vector<int> &seeds, std::vector<int> &source, std::vector<int> &dest_offset, std::vector<float> &weights,int ghost){

    // inicia com um zero
    dest_offset.push_back(0);

    for(int pixel = 0; pixel < img->total_size ; pixel++){
    
        int offset = dest_offset[pixel];
        int pixel_row = pixel/img->rows;
        int pixel_col = pixel - pixel_row*img->rows;

        // tratamento da ghost seed
        if (find(begin(seeds), end(seeds), pixel) != end(seeds)) {
            source.push_back(ghost);
            weights.push_back(0.0);
            offset++;
        }

        // pixel de cima
        int acima = pixel - img->cols;
        if (pixel_row > 0) {
            offset++;
            source.push_back(acima);
            double custo = get_edge( img, pixel , acima);
            weights.push_back(custo);
        }

        // pixel de baixo
        int abaixo = pixel + img->cols;
        if (pixel_row < img->rows - 1) { 
            offset++;
            source.push_back(abaixo);
            double custo = get_edge( img, pixel ,abaixo);
            weights.push_back(custo);       
        }

        // pixel da direita
        int direita = pixel + 1;
        if (pixel_col < img->cols - 1) {
            offset++;
            source.push_back(direita);
            double custo = get_edge( img, pixel , direita);
            weights.push_back(custo);
        }

        // pixel da esquerda
        int esquerda = pixel - 1;
        if (pixel_col > 0) {
            offset++;
            source.push_back(esquerda);
            double custo = get_edge( img, pixel , esquerda);
            weights.push_back(custo);
        }

        dest_offset.push_back(offset);
    }
}


int main(int argc, char **argv) {

    if (argc < 3) {
        std::cout << "Uso:  segmentacao_cuda entrada.pgm saida.pgm\n";
        return -1;
    }

    //caminho do input e do output
    std::string path(argv[1]);
    std::string out_path(argv[2]);

    std::vector<int> source_fg,source_bg, dest_offset_fg, dest_offset_bg;
    std::vector<float> weights_fg, weights_bg;

    int n_fg, n_bg, x, y;
    float total_time, graph_time, sssp_time, output_time;

    // variaveis de contagem de tempo
    cudaEvent_t total_begin, total_end, begin, end;
    cudaEventCreate(&total_begin);
    cudaEventCreate(&total_end);
    cudaEventCreate(&begin);
    cudaEventCreate(&end);

    // inicio da contagem de tempo total do programa
    cudaEventRecord(total_begin);
    
    imagem *img = read_pgm(path);

    int nrows = img->rows;
    int ncols = img->cols;
    
    // numero de sementes de frente e de fundo
    std::cout << "\nnumero de sementes de frente e de fundo:\n";
    std::cin >> n_fg >> n_bg;
    
    std::vector<int> seeds_fg(n_fg), seeds_bg(n_bg);

    if(n_fg <= 0 || n_bg <= 0){
        std::cout << "numero de sementes não pode ser menor que zero";
        return -1;
    }

    std::cout << "posições das sementes de frente:\n";

    for(int i = 0; i < n_fg; i++)
    {
        std::cin >> x >> y;
        seeds_fg[i] = y * img->cols + x;
    }

    std::cout << "posições das sementes de fundo:\n";

    for(int i = 0; i < n_bg; i++)
    {
        std::cin >> x >> y;
        seeds_bg[i] = y * img->cols + x;    
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

    int fg_ghost = img->total_size+1;
    int bg_ghost = img->total_size;

    // começo da contagem de tempo da geração dos vetores
    cudaEventRecord(begin);

    //geração dos vetores
    vectorsGen(edge, seeds_bg, source_bg, dest_offset_bg, weights_bg, bg_ghost);
    vectorsGen(edge, seeds_fg, source_fg, dest_offset_fg, weights_fg, fg_ghost);

    //fim da contagem
    cudaEventRecord(end);
    cudaEventSynchronize(end);
    cudaEventElapsedTime(&graph_time, begin, end);

    //SSSP 
    imagem *saida = new_image(nrows, ncols);
    
    //numero de arestas
    int edges_fg = 2*((ncols-1)*nrows+(nrows-1)*ncols) + n_fg;
    int edges_bg = 2*((ncols-1)*nrows+(nrows-1)*ncols) + n_bg;
   
    std::vector<float> out_fg (img->total_size);
    std::vector<float> out_bg (img->total_size);

    //inicio da contagem de tempo do sssp
    cudaEventRecord(begin);

    //funções sssp para semente de frente e semente de fundo
    SSSP(img->total_size,edges_bg,weights_bg,source_bg,dest_offset_bg, bg_ghost, out_bg);
    SSSP(img->total_size,edges_fg,weights_fg,source_fg,dest_offset_fg, fg_ghost, out_fg);

    //fim da contagem
    cudaEventRecord(end);
    cudaEventSynchronize(end);
    cudaEventElapsedTime(&sssp_time, begin, end);

    //inicio da contagem de tempo da construção da imagem de saida
    cudaEventRecord(begin);
    //img de saida
    
    for (int i = 0;i < saida->total_size; i++) {
        if (out_fg[i] > out_bg[i]) {
            saida->pixels[i] = 0;
        } else {
            saida->pixels[i] = 255;
        }
    }

    write_pgm(saida, out_path);

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

    std::cout << "\n--------------------------------------------\n";
    std::cout << "--------------------TIME--------------------\n";
    std::cout << "--------------------------------------------\n\n";
    std::cout << "graph_time: " << graph_time <<  "\n";
    std::cout << "sssp_time: " << sssp_time << "\n";
    std::cout << "output_time:  " << output_time <<  "\n";
    std::cout << "total_time: " << total_time <<  "\n";

    return 0;
}
