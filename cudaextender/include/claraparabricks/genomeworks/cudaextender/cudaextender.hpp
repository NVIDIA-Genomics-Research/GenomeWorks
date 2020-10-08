/*
* Copyright 2020 NVIDIA CORPORATION.
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*/

#pragma once

namespace claraparabricks
{

namespace genomeworks
{

namespace cudaextender
{
/// \defgroup cudaextender CUDA Extender package
/// Base docs for the cudaextender package
/// \{

/// CUDA Extender status type
enum StatusType
{
    success           = 0,
    invalid_operation = 1,
    invalid_input     = 2,
    generic_error
};

/// Extension types
/// ungapped_xdrop performs ungapped extension with an x-drop threshold
///
enum ExtensionType
{
    ungapped_xdrop = 0,
};

/// Initialize CUDA Extender context.
StatusType Init();

/// \}
} // namespace cudaextender

} // namespace genomeworks

} // namespace claraparabricks
