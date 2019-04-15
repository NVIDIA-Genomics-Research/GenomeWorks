#pragma once

#include <memory>
#include <string>

namespace genomeworks {

class Sequence {
public:
    ~Sequence() = default;

    const std::string &name() const {
        return name_;
    }

    const std::string &data() const {
        return data_;
    }

    Sequence(const char *name, uint32_t name_length, const char *data,
             uint32_t data_length);


    Sequence(const Sequence &) = delete;

    const Sequence &operator=(const Sequence &) = delete;

private:
    std::string name_;
    std::string data_;
};
}
