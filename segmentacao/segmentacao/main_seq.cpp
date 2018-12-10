#include <iostream>
#include <queue>
#include <vector>
#include <assert.h>
#include <fstream>

#include "imagem.h"

typedef std::pair<double, int> custo_caminho;

typedef std::pair<double *, int *> result_sssp;


struct compare_custo_caminho {
    bool operator()(custo_caminho &c1, custo_caminho &c2) {
        return c2.first < c1.first;
    }
};

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
    
    //vector<int> n_fg, n_bg;
    int n_fg, n_bg;
    int x, y;
    
    // numero de sementes de frente e de fundo
    std::cin >> n_fg >> n_bg;


    //só aceita 1 semente de cada tipo
    //assert(n_fg == 1);
    //assert(n_bg == 1);
    
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
    
    // retorna pair<double *, int *> result_sssp; -> <custos, predecessores>
    result_sssp fg = SSSP(img, seed_fg);
    result_sssp bg = SSSP(img, seed_bg);

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
       
    return 0;
}
