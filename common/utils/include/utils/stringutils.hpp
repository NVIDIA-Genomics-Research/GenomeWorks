#pragma once

#include <string>
#include <stdint.h>


// utility function to convert an array of node ids into a readable string representation
inline std::string array_to_string(uint16_t *arr, size_t len, std::string delim = "-") {
    std::string res;
    for (size_t i = 0; i < len; i++)
    {
        res += std::to_string(arr[i]) + (i == len - 1 ? "" : delim);
    }
    return res;
}
