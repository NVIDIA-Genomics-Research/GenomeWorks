#include <string>
#include "cudamapper/index.hpp"

int main(int argc, char *argv[]){

    // TODO: pass k-mer and window size as parameters
    std::unique_ptr<genomeworks::Index> index_generator = genomeworks::Index::create_index(4, 4);
    index_generator->generate_index(std::string(argv[1]));
    return  0;
}
