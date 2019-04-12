#ifndef GENOMEWORKSCPP_INDEX_HPP
#define GENOMEWORKSCPP_INDEX_HPP

namespace cudamapper {
    class Index {
    public:
        Index();

        void generate_index(char *fasta_filename);
    };
}
#endif //GENOMEWORKSCPP_INDEX_HPP
