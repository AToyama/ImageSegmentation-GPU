#include <iostream>
#include <queue>
#include <vector>
#include <fstream>

#include "imagem.h"


//( img in, seeds fg, seeds bg, source, dest off, weights, fg ghost, bg ghost )
void vectorsGen(imagem *img, std::vector<int> &seeds_fg, std::vector<int> &seeds_bg, std::vector<int> &source, std::vector<int> &dest_offset, std::vector<float> weights, int fg_ghost, int bg_ghost){

    // inicia com um zero
    dest_offset.push_back(0);


    for(int pixel = 0; pixel < img.size(); pixel ++){

        int offset = dest_offset[i];

        //if pixel in front seed -> add to vectors ghost seed 
        //elif pixel back seed -> add to vectors ghost seed

        if(std::find(seeds_fg.begin(), seeds_fg.end(), pixel) != seeds_fg.end()) {){

            offset++;
            source.push_back(fg_ghost);
    	    weights.push_back(0.0);

        }

        else if(std::find(seeds_bg.begin(), seeds_bg.end(), pixel) != seeds_bg.end())){            

            offset++;
            source.push_back(bg_ghost);
    	    weights.push_back(0.0);

        }

        // pixel de cima
        int acima = vertex - img->cols;
        if (acima >= 0) {
            offset++;
            source.push_back(acima);
            weights.push_back(get_edge_edge(img, i , acima));
        }

        // pixel de baixo
        int abaixo = vertex + img->cols;
        if (abaixo < img->total_size) {
            offset++;
            source.push_back(abaixo);
            weights.push_back(get_edge_edge(img, i , abaixo));
        }

        // pixel da esquerda
        int esquerda = vertex - 1;
        if (esquerda >= 0) {
            offset++;
            source.push_back(esquerda);
            weights.push_back(get_edge_edge(img, i , esquerda));
        }

        // pixel da direita
        int direita = vertex + 1;
        if (direita < img->total_size) {
            offset++;
            source.push_back(direita);
            weights.push_back(get_edge_edge(img, i , direita));
        }

        dest_offset.push_back(offset);
    }
    dest_offset.push_back(source.size());
    }
}

