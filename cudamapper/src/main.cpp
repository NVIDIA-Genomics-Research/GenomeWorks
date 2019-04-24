#include <string>
#include <iostream>
#include "cudamapper/index.hpp"

int main(int argc, char *argv[]){

    // TODO: pass kmer and window size as parameters
    std::unique_ptr<genomeworks::Index> index_generator = genomeworks::Index::create_index(4, 4);
    std::cout<<"Generating index"<<std::endl;
    index_generator->generate_index(std::string(argv[1]));
    std::cout<<"Index generated"<<std::endl;
    return  0;
}
