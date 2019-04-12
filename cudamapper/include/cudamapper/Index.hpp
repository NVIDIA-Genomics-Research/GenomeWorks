#ifndef GENOMEWORKSCPP_INDEX_HPP
#define GENOMEWORKSCPP_INDEX_HPP

namespace cudamapper {
    class index {
    public:
        index();

        void generate_index(char *fasta_filename);
    };
}
#endif //GENOMEWORKSCPP_INDEX_HPP
