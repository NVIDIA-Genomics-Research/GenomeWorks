#include "cudamapper/index.hpp"

int main(int argc, char *argv[]){

    cudamapper::Index index_generator = cudamapper::Index();
    index_generator.generate_index(argv[1]);

    return  0;
}