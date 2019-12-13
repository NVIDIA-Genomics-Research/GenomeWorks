/*
* Copyright (c) 2019, NVIDIA CORPORATION.  All rights reserved.
*
* NVIDIA CORPORATION and its licensors retain all intellectual property
* and proprietary rights in and to this software, related documentation
* and any modifications thereto.  Any use, reproduction, disclosure or
* distribution of this software and related documentation without an express
* license agreement from NVIDIA CORPORATION is strictly prohibited.
*/

#pragma once

#include <sys/types.h>
#include <sys/stat.h>
#include <string>

namespace claragenomics
{

namespace filesystem
{

bool dirExists(const std::string path)
{
    struct stat info;

    int response = stat(path.c_str(), &info);
    if (response != 0)
    {
        return false;
    }

    return (info.st_mode & S_IFDIR) ? true : false;
}


std::string resolveFileName(const std::string file_path)
{
    size_t found;
    found = file_path.find_last_of("/\\");
    return file_path.substr(found + 1);
}

} //namespace filesystem

} // namespace claragenomics
