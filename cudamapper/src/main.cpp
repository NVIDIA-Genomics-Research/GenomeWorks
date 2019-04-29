#include <string>
#include <iostream>
#include "cudamapper/index_generator.hpp"

int main(int argc, char *argv[]){

    // TODO: pass kmer and window size as parameters
    std::unique_ptr<genomeworks::IndexGenerator> index_generator = genomeworks::IndexGenerator::create_index_generator(4, 4);
    std::cout<<"Generating index"<<std::endl;
    index_generator->generate_index(std::string(argv[1]));
    std::cout<<"Index generated"<<std::endl;
    return  0;
}
