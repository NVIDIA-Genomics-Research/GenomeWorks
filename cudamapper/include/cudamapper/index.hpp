#pragma  once

namespace cudamapper {
    class Index {
    public:
        Index();

        void generate_index(char *fasta_filename);
    };
}
