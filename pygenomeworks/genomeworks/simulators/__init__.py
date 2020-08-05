#
# Copyright 2019-2020 NVIDIA CORPORATION.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#


"""simulator submodule wide initialization."""

NUCLEOTIDES = set(('A', 'C', 'G', 'T'))

HIGH_GC_HOMOPOLYMERIC_TRANSITIONS = {
    "A": {
        "A": 0.25 * 3,
        "C": 0.25 * 1.25,
        "G": 0.25 * 1.25,
        "T": 0.25},
    "C": {
        "A": 0.25,
        "C": 0.25 * 3,
        "G": 0.25 * 1.25,
        "T": 0.25},
    "G": {
        "A": 0.25,
        "C": 0.25 * 1.25,
        "G": 0.25 * 1.25,
        "T": 0.25 * 3},
    "T": {
        "A": 0.25,
        "C": 0.25 * 3,
        "G": 0.25 * 3,
        "T": 0.25 * 1.25}}

HOMOGENOUS_TRANSITIONS = {
    "A": {
        "A": 0.25 * 10,
        "C": 0.25,
        "G": 0.25,
        "T": 0.25},
    "C": {
        "A": 0.25,
        "C": 0.25,
        "G": 0.25,
        "T": 0.25},
    "G": {
        "A": 0.25,
        "C": 0.25,
        "G": 0.25,
        "T": 0.25},
    "T": {
        "A": 0.25,
        "C": 0.25,
        "G": 0.25,
        "T": 0.25}}
