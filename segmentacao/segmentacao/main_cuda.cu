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

//scp -i supercomp-certo.pem -r ./segmentacao_sequencial ubuntu@ec2-54-160-10-252.compute-1.amazonaws.com:~/toy
//scp -i supercomp-certo.pem ./segmentacao_sequencial/main_cuda.cu ubuntu@ec2-54-160-10-252.compute-1.amazonaws.com:~/toy/segmentacao_sequencial
//scp -i supercomp-certo.pem  ubuntu@ec2-54-160-10-252.compute-1.amazonaws.com:~/toy/segmentacao/saida.pgm ./segmentacao
//ssh -i supercomp-certo.pem ubuntu@ec2-54-160-10-252.compute-1.amazonaws.com
//nvcc -std=c++11 imagem.cpp main_cuda.cu -o segmentacao_cuda -lnvgraph


// FILTRO DE BORDAS
__global__ void edgeFilter(unsigned char *in, unsigned char *out, int rowStart, int rowEnd, int colStart, int colEnd) {
    
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    int j=blockIdx.y*blockDim.y+threadIdx.y;

    int di,dj;
    
    if(i < rowEnd && j < colEnd){

        for(i = rowStart; i < rowEnd; ++i) {
            for(j = colStart; j < colEnd; ++j) {
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
    }
}

void check(nvgraphStatus_t status)
{
    if ((int)status != 0)
    {
        printf("ERROR : %d\n",status);
        exit(0);
    }
}

void SSSP(const size_t n, const size_t nnz, float* weights_h, int* source_indices_h, int* destination_offsets_h, int source_vert, float *sssp_1_h) {
    
    const size_t vertex_numsets = 2, edge_numsets = 1;
    void** vertex_dim;
    // nvgraph variables
    nvgraphStatus_t status; nvgraphHandle_t handle;
    nvgraphGraphDescr_t graph;
    nvgraphCSCTopology32I_t CSC_input;
    cudaDataType_t edge_dimT = CUDA_R_32F;
    cudaDataType_t* vertex_dimT;
    // Init host data
    
    vertex_dim  = (void**)malloc(vertex_numsets*sizeof(void*));
    vertex_dimT = (cudaDataType_t*)malloc(vertex_numsets*sizeof(cudaDataType_t));
    CSC_input = (nvgraphCSCTopology32I_t) malloc(sizeof(struct nvgraphCSCTopology32I_st));
    vertex_dim[0]= (void*)sssp_1_h; vertex_dimT[0] = CUDA_R_32F;

    check(nvgraphCreate(&handle));
    check(nvgraphCreateGraphDescr (handle, &graph));
    CSC_input->nvertices = n; CSC_input->nedges = nnz;
    CSC_input->destination_offsets = destination_offsets_h;
    CSC_input->source_indices = source_indices_h;
    // Set graph connectivity and properties (tranfers)
    check(nvgraphSetGraphStructure(handle, graph, (void*)CSC_input, NVGRAPH_CSC_32));
    check(nvgraphAllocateVertexData(handle, graph, vertex_numsets, vertex_dimT));
    check(nvgraphAllocateEdgeData  (handle, graph, edge_numsets, &edge_dimT));
    check(nvgraphSetEdgeData(handle, graph, (void*)weights_h, 0));
    
    check(nvgraphSssp(handle, graph, 0,  &source_vert, 0));
    // Get and print result
    check(nvgraphGetVertexData(handle, graph, (void*)sssp_1_h, 0));
    //Clean 
    free(sssp_1_h); free(vertex_dim);
    free(vertex_dimT); free(CSC_input);
    check(nvgraphDestroyGraphDescr(handle, graph));
    check(nvgraphDestroy(handle));

}

//( img in, seeds fg, seeds bg, source, dest off, weights, fg ghost, bg ghost )
void vectorsGen(imagem *img, std::vector<int> &seeds_fg, std::vector<int> &seeds_bg, std::vector<int> &source, std::vector<int> &dest_offset, std::vector<float> &weights, int fg_ghost, int bg_ghost){

    // inicia com um zero
    dest_offset.push_back(0);


    for(int pixel = 0; pixel < img->total_size ; pixel++){

        int offset = dest_offset[pixel];
        int pixel_row = pixel/img->rows;
        int pixel_col = pixel - pixel_row*img->rows;

        //if pixel in front seed -> add to vectors ghost seed 
        //elif pixel back seed -> add to vectors ghost seed

        if (find(begin(seeds_fg), end(seeds_fg), pixel) != end(seeds_fg)) {
            source.push_back(fg_ghost);
            weights.push_back(0.0);
            offset++;
        }

        if (find(begin(seeds_bg), end(seeds_bg), pixel) != end(seeds_bg)) {
            source.push_back(bg_ghost);
            weights.push_back(0.0);
            offset++;
        }


        // pixel de cima
        int acima = pixel - img->cols;
        if (pixel_row > 0) {
            offset++;
            source.push_back(acima);
            weights.push_back(get_edge( img, pixel , acima));
        }

        // pixel de baixo
        int abaixo = pixel + img->cols;
        if (pixel_row < img->rows - 1) { 
            offset++;
            source.push_back(abaixo);
            weights.push_back(get_edge( img, pixel , abaixo));
        }

        // pixel da esquerda
        int esquerda = pixel - 1;
        if (pixel_col > 0) {
            offset++;
            source.push_back(esquerda);
            weights.push_back(get_edge( img, pixel , esquerda));
        }

        // pixel da direita
        int direita = pixel + 1;
        if (pixel_col < img->cols - 1) {
            offset++;
            source.push_back(direita);
            weights.push_back(get_edge( img, pixel, direita));
        }

        dest_offset.push_back(offset);
    }
}



int main(int argc, char **argv) {

    if (argc < 3) {
        std::cout << "Uso:  segmentacao_cuda entrada.pgm saida.pgm\n";
        return -1;
    }

    std::string path(argv[1]);
    std::string out_path(argv[2]);
    std::vector<int> source, dest_offset;
    std::vector<float> weights;

    int n_fg, n_bg, x, y;

    imagem *img = read_pgm(path);

    int nrows = img->rows;
    int ncols = img->cols;
    
    // numero de sementes de frente e de fundo
    std::cout << "numero de sementes de frente e de fundo:\n";
    std::cin >> n_fg >> n_bg;
    
    std::cout << "posições das sementes de frente:\n";
    std::vector<int> seeds_fg(n_fg);

    for(int i = 0; i < n_fg; i++)
    {
        std::cin >> x >> y;
        seeds_fg[i] = y * img->cols + x;
    }

    std::cout << "posições das sementes de fundo:\n";
    std::vector<int> seeds_bg(n_bg);

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

    edgeFilter<<<dimGrid,dimBlock>>>(thrust::raw_pointer_cast(input.data()), thrust::raw_pointer_cast(output.data()), 0, nrows, 0, ncols);

    thrust::host_vector<unsigned char> output_data(output);
    for(int i = 0; i != output_data.size(); i++) {
        edge->pixels[i] = output_data[i];
    }

    int fg_ghost = img->total_size+1;
    int bg_ghost = img->total_size;

    //( img in, seeds fg, seeds bg, source, dest off, weights, fg ghost, bg ghost )
    vectorsGen(edge, seeds_fg, seeds_bg, source, dest_offset, weights, fg_ghost, bg_ghost);

    //SSSP 
    imagem *saida = new_image(nrows, ncols);
    //std::vector<int> out_fg, out_bg;

    int * source_ = (int*) malloc(source.size()*sizeof(int));
    for (int i = 0; i < source.size(); i++){
        source_[i] = source[i];
    }

    float * weights_ = (float*)malloc(weights.size()*sizeof(float));
    for (int i = 0; i < weights.size(); i++){
        weights_[i] = weights[i];
    }

    int * dest_offset_ = (int*) malloc(dest_offset.size()*sizeof(int));
    for (int i = 0; i < dest_offset.size(); i++){
        dest_offset_[i] = dest_offset[i];
        
    }
    printf("sizes: %d %d %d",source.size(),weights.size(),dest_offset.size());
    size_t nnz = (2*((img->cols-1)*img->rows+(img->rows-1)*img->cols))+1;
    //( num_pixels, source_vert, destination_offsets_h, source_indices_h, weights_h, sssp_output )
  
    std::cout <<"img size" << img->total_size << "\n";
    std::cout <<"edge size" << edge->total_size << "\n";
    std::cout <<"nnz" << nnz << "\n";
    std::cout <<"bg ghost"  << bg_ghost << "\n";
    std::cout <<"fg ghost" << fg_ghost << "\n";
    

    std::cout << "\n";
    for(int i = 0; i< weights.size();i++){
        if(weights[i] != 0){
        std::cout << weights[i] << "\n";
}    }
    

    //SSSP(img->total_size, fg_ghost, dest_offset, source, weights, out_fg);
    float * out_fg = (float*)malloc((edge->total_size+2)*sizeof(float));
    //(size, edges, float* weights_h, int* source_indic  es, int* destination_offsets_h, int source_vert, float *sssp_1_h
    SSSP(edge->total_size+2,nnz,weights_,dest_offset_,source_, bg_ghost, out_fg);

    float * out_bg = (float*)malloc((edge->total_size+2)*sizeof(float));
    SSSP(edge->total_size+2,nnz,weights_,dest_offset_,source_,bg_ghost, out_bg);
 
    //SSSP(img->total_size, bg_ghost, dest_offset, source, weights, out_bg);

    for (int i = 0;i < saida->total_size; i++) {
        if (out_fg[i] > out_bg[i]) {
            saida->pixels[i] = 100;
        } else {
            saida->pixels[i] = 255;
        }
    }

    write_pgm(saida, out_path);

    return 0;
}
