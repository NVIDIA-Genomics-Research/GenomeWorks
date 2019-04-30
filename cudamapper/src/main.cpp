#include <string>
#include <iostream>
#include "cudamapper/index.hpp"

int main(int argc, char *argv[]){

    std::cout<<"Creating index generator"<<std::endl;
    // TODO: pass kmer and window size as parameters
    std::unique_ptr<genomeworks::IndexGenerator> index_generator = genomeworks::IndexGenerator::create_index_generator(std::string(argv[1]), 4, 4);
    std::cout<<"Created index generator"<<std::endl;
    std::cout<<"Creating index"<<std::endl;
    std::unique_ptr<genomeworks::Index> index = genomeworks::Index::create_index(*index_generator);
    std::cout<<"Created index"<<std::endl;
    return  0;
}
