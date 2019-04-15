#pragma  once

#include <memory>

namespace genomeworks {

    class Index {
    public:
        virtual void generate_index(std::string query_filename) = 0;
        static std::unique_ptr<Index> create_index();
    };
}
