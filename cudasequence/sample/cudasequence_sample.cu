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

__global__ some_kernel(DeviceSequenceCollection::device_interface* p)
{
    DeviceSequenceCollection seq = p->get_view();
    const char internal_representation = seq[i];
    
    // QUESTION: Do we need to modify sequences on the device?
}


__global__ kernel_a(ScoringMatrix::device_interface* m_ptr, DeviceSequence::device_interface* query_ptr, DeviceSequence::device_interface* target_ptr)
{
    ScoringMatrix::device_interface&  score_matrix = *m_ptr;
    DeviceSequence::device_interface& query        = *query_ptr;
    DeviceSequence::device_interface& target       = *target_ptr;

    const int32_t score = score_matrix(query[i - 1], target[j - 1]);
}

// alternative:
__global__ kernel_a(ScoringMatrix::device_interface* m_ptr, const char* query_encoded, const char* target_encoded);


int main()
{

/////////////////////////////
// Host side
/////////////////////////////

// Alphabet
std::shared_ptr<Alphabet> a = make_alphabet("acgt");

// Scoring matrix
ScoringMatrix m(a, m);
m('a','a') = score;
m('a','c') = score;

// Sequence - always 5'->3' (or other norm)
Sequence seq1 = make_sequence(a, "acgtgtaccta", is_reverse_complement); // converts acgt -> [0,1,2,3]
Sequence seq2 = make_sequence(a, reverse_complement("acgtgtaccta"));

std::string s = seq1.get_forward_string(); // returns std::string acgt
bool reverse  = seq1.was_reverse_complement();

// Device Sequence 
DeviceSequence seq_d(seq, stream);

char* str_on_device = ...;

DeviceSequence seq_d2(a, str_on_device, size, is_reverse_complement);

// Either: hidden data format, only accessible via [] operator: +may allow for tighter packing, eg. 4 bases per char.
// or just a char array with values \in [0, alphabet_size)


// Sequence vector - Idea 1
SequenceCollection seqs(max_collection_size);
seqs.push_back(seq1);
seqs.push_back(seq2);


// Device Sequence Collection
DeviceSequenceCollection seqs_d(seqs, stream);

const char** device_strings = ...;
const int64_t** string_sizes = ...;
int32_t number_of_strings = ...;
DeviceSequenceCollection seqs_d2(number_of_strings, device_strings, string_sizes);

some_kernel<<<1,1>>>(seqs_d.get_device_interface());

}
