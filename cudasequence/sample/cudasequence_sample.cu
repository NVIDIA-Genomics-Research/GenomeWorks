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


// THIS IS AN EARLY DRAFT - WORK IN PROGRESS


// Problems we want to solve:
// a) Have a representation which can be directly used as matrix index
// b) (optional) reduce size of representations

__global__ some_kernel(ScoringMatrix::device_interface* m_ptr, DeviceSequenceVector::device_interface* p)
{
    int idx = ...;
    ScoringMatrix::device_interface&  score_matrix = *m_ptr;
    SequenceView query  = p->get_seq(idx);
    SequenceView target = p->get_seq(idx + 1);

    int size = min(get_size(query), get_size(target));
    for(int i = 0; i < size; ++i)
    {
        int32_t score = score_matrix(query[i], target[j]);
        // or:
        ScoringMatrix::value_type score = score_matrix(query[i], target[j]);

        bool is_same = (query[i] == target[i]);
        char internal_representation = query[i];
    }

    // QUESTION: Do we need to modify sequences on the device?
    // -> no, but we need to be able to create sequences.
}


// FIXME:
__global__ kernel_a(ScoringMatrix::device_interface* m_ptr, const char* query_encoded, const char* target_encoded);


int main()
{

/////////////////////////////
// Host side
/////////////////////////////

// Alphabet
std::shared_ptr<Alphabet> a = make_alphabet("acgt");
const char internal_rep = a.get_internal_representation('c'); // in case you need to compare against a special base/character.

// Scoring matrix
ScoringMatrix m(a, m);
m('a','a') = score;
m('a','c') = score;

// Sequence vector - Idea 1
// Either: hidden data format, only accessible via [] operator: +may allow for tighter packing, eg. 4 bases per char.
// or just a char array with values \in [0, alphabet_size)
SequenceVector seqs(alphabet, max_elements_total);
seqs.push_back_string("acgtgtacc", is_reverse_complement);
bool appended1 = seqs.push_back(seq1);
bool appended2 = seqs.push_back(seq2);


SequenceView seq0 = seqs[0];
SequenceView seq1(forward_str, size);
std::string s = seq0.form_forward_string(); // returns std::string acgt
bool reverse  = seq0.was_reverse_complement();

// Device Sequence Vector
DeviceSequenceVector seqs_d(seqs, stream);

const char** device_strings = ...;
const int64_t** string_sizes = ...;
int32_t number_of_strings = ...;
DeviceSequenceVector seqs_d2(alphabet, number_of_strings, device_strings, string_sizes);

some_kernel<<<1,1>>>(seqs_d.get_device_interface());

}
