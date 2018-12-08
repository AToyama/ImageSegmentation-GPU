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

#include "nvgraph.h"
#include "imagem.h"
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
//#include <nvgraph.h>

#define MAX(y,x) (y>x?y:x)    // Calcula valor maximo
#define MIN(y,x) (y<x?y:x)    // Calcula valor minimo

//scp -i supercomp-certo.pem -r ./segmentacao_sequencial ubuntu@ec2-54-160-10-252.compute-1.amazonaws.com:~/toy
//scp -i supercomp-certo.pem ./segmentacao_sequencial/main_cuda.cu ubuntu@ec2-54-160-10-252.compute-1.amazonaws.com:~/toy/segmentacao_sequencial
//ssh -i supercomp-certo.pem ubuntu@ec2-54-160-10-252.compute-1.amazonaws.com


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

int SSSP(int &num_pixels, int &source_vert,std::vector<int> &destination_offsets_h, std::vector<int> &source_indices_h,std::vector<float> &weights_h,std::vector<int> &sssp_output){
    
    int* destination_offsets  = new int[destination_offsets_h.size()];
    for(unsigned i=0; i<destination_offsets_h.size();i++)
        destination_offsets[i] = destination_offsets_h[i];

    int* source_indices  = new int[source_indices_h.size()];
    for(unsigned s=0; s<source_indices_h.size();s++)
        source_indices[s] = source_indices_h[s];

    float* weights  = new float[weights_h.size()];
    for(unsigned w=0; w<weights_h.size();w++)
        weights[w] = weights_h[w];

    const size_t  n = num_pixels, nnz = weights_h.size(), vertex_numsets = 1, edge_numsets = 1;
    int i;
    float *sssp_1_h;
    void** vertex_dim;

    // nvgraph variables
    nvgraphStatus_t status;
    nvgraphHandle_t handle;
    nvgraphGraphDescr_t graph;
    nvgraphCSCTopology32I_t CSC_input;
    cudaDataType_t edge_dimT = CUDA_R_32F;
    cudaDataType_t* vertex_dimT;

    sssp_1_h = (float*)malloc(n*sizeof(float));
    vertex_dim  = (void**)malloc(vertex_numsets*sizeof(void*));
    vertex_dimT = (cudaDataType_t*)malloc(vertex_numsets*sizeof(cudaDataType_t));
    CSC_input = (nvgraphCSCTopology32I_t) malloc(sizeof(struct nvgraphCSCTopology32I_st));

    vertex_dim[0]= (void*)sssp_1_h; vertex_dim[1];
    vertex_dimT[0] = CUDA_R_32F; vertex_dimT[1]= CUDA_R_32F;

    check_status(nvgraphCreate(&handle));
    check_status(nvgraphCreateGraphDescr (handle, &graph));

    CSC_input->nvertices = n;
    CSC_input->nedges = nnz;
    CSC_input->destination_offsets = destination_offsets;
    CSC_input->source_indices = source_indices;

    check_status(nvgraphSetGraphStructure(handle, graph, (void*)CSC_input, NVGRAPH_CSC_32));
    check_status(nvgraphAllocateVertexData(handle, graph, vertex_numsets, vertex_dimT));
    check_status(nvgraphAllocateEdgeData  (handle, graph, edge_numsets, &edge_dimT));
    check_status(nvgraphSetEdgeData(handle, graph, (void*)weights, 0));

    check_status(nvgraphSssp(handle, graph, 0,  &source_vert, 0));

    check_status(nvgraphGetVertexData(handle, graph, (void*)sssp_1_h, 0));

    printf("sssp_1_h\n");
    for (i = 0; i<n; i++)  sssp_output.push_back(sssp_1_h[i]);
    printf("\nDone!\n");

    free(sssp_1_h);

    free(vertex_dim);
    free(vertex_dimT);
    free(CSC_input);

    //Clean 
    check_status(nvgraphDestroyGraphDescr (handle, graph));
    check_status(nvgraphDestroy (handle));

    return EXIT_SUCCESS;
}


//( img in, seeds fg, seeds bg, source, dest off, weights, fg ghost, bg ghost )
void vectorsGen(imagem *img, std::vector<int> &seeds_fg, std::vector<int> &seeds_bg, std::vector<int> &source, std::vector<int> &dest_offset, std::vector<float> weights, int fg_ghost, int bg_ghost){

    // inicia com um zero
    dest_offset.push_back(0);


    for(int pixel = 0; pixel < img->total_size ; pixel++){

        int offset = dest_offset[pixel];

        //if pixel in front seed -> add to vectors ghost seed 
        //elif pixel back seed -> add to vectors ghost seed

        for(int i = 0; i < seeds_fg.size(); i++){
            if(seeds_fg[i] = pixel) {

                offset++;
                source.push_back(fg_ghost);
                weights.push_back(0.0);
                break;
            }
            else if(seeds_bg[i] == pixel) {            

                offset++;
                source.push_back(bg_ghost);
                weights.push_back(0.0);
    
            }
        }
  
        



        // pixel de cima
        int acima = pixel - img->cols;
        if (acima >= 0) {
            offset++;
            source.push_back(acima);
            weights.push_back(get_edge( img, pixel , acima));
        }

        // pixel de baixo
        int abaixo = pixel + img->cols;
        if (abaixo < img->total_size) {
            offset++;
            source.push_back(abaixo);
            weights.push_back(get_edge( img, pixel , abaixo));
        }

        // pixel da esquerda
        int esquerda = pixel - 1;
        if (esquerda >= 0) {
            offset++;
            source.push_back(esquerda);
            weights.push_back(get_edge( img, pixel , esquerda));
        }

        // pixel da direita
        int direita = pixel + 1;
        if (direita < img->total_size) {
            offset++;
            source.push_back(direita);
            weights.push_back(get_edge( img, pixel, direita));
        }

        dest_offset.push_back(offset);
    }
    dest_offset.push_back(source.size());
}



int main(int argc, char **argv) {

    /*
    //variables for nvGraph
	nvgraphHandle_t handle;
	nvgraphGraphDescr_t graph;
	nvgraphCSCTopology32I_t col_major_topology;
	cudaDataType_t edge_dimT = CUDA_R_32F;
*/
    //initialization of nvGraph
	//nvgraphCreate(&handle);
	//nvgraphCreateGraphDescr(handle, &graph);

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
    imagem *saida = new_image(nrows, ncols);

    thrust::device_vector<unsigned char> input(img->pixels, img->pixels + img->total_size );
    thrust::device_vector<unsigned char> output(saida->pixels, saida->pixels + saida->total_size );

    dim3 dimGrid(ceil(nrows/16.0), ceil(ncols/16.0), 1);
    dim3 dimBlock(16, 16, 1);

    edgeFilter<<<dimGrid,dimBlock>>>(thrust::raw_pointer_cast(input.data()), thrust::raw_pointer_cast(output.data()), 0, nrows, 0, ncols);

    thrust::host_vector<unsigned char> output_data(output);
    for(int i = 0; i != output_data.size(); i++) {
        saida->pixels[i] = output_data[i];
    }
    write_pgm(saida, out_path);
            int fg_ghost = img->total_size+1;
    int bg_ghost = img->total_size+2;

    

    //( img in, seeds fg, seeds bg, source, dest off, weights, fg ghost, bg ghost )
    vectorsGen(img, seeds_fg, seeds_bg, source, dest_offset, weights, fg_ghost, bg_ghost);

    //SSSP

    return 0;
}
