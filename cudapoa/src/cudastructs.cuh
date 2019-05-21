#pragma once

namespace genomeworks
{

namespace cudapoa
{

template <typename SeqT>
struct SeqT4
{
    SeqT r0, r1, r2, r3;
};

template <typename ScoreT>
struct ScoreT4
{
    ScoreT s0, s1, s2, s3;
};

template <>
struct __align__(4) ScoreT4<int16_t>
{
    int16_t s0, s1, s2, s3;
};
} // namespace cudapoa
} // namespace genomeworks
