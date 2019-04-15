#include <string>
#include "cudamapper/index.hpp"

int main(int argc, char *argv[]){

    std::unique_ptr<genomeworks::Index> index_generator = genomeworks::Index::create_index();
    index_generator->generate_index(std::string(argv[1]));
    return  0;
}
