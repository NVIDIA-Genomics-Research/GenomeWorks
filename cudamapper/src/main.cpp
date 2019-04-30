#include <string>
#include <iostream>
#include "cudamapper/index_generator.hpp"

int main(int argc, char *argv[]){

    std::cout<<"Generating index"<<std::endl;
    // TODO: pass kmer and window size as parameters
    std::unique_ptr<genomeworks::IndexGenerator> index_generator = genomeworks::IndexGenerator::create_index_generator(std::string(argv[1]), 4, 4);
    std::cout<<"Index generated"<<std::endl;
    return  0;
}
