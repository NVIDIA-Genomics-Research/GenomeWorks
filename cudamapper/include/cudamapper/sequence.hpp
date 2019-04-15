#pragma once

#include <memory>
#include <string>

namespace genomeworks {

class Sequence {
public:
    virtual const std::string &name() const = 0;
    virtual const std::string &data() const = 0;
    static std::unique_ptr<Sequence> create_sequence();

private:
    std::string name_;
    std::string data_;
};
}
