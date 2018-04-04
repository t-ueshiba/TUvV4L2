/*
 *  $Id$
 */
#include "TU/simd/cvtdown_iterator.h"
#include "TU/simd/cvtup_iterator.h"
#include "TU/simd/load_store_iterator.h"

namespace TU
{
namespace simd
{
template <class S, class T> void
doJob()
{
    using siterator	= load_iterator<S>;
    using diterator	= store_iterator<T>;
    using src_iterator	= std::conditional_t<(vec<S>::size <= vec<T>::size),
					     cvtdown_iterator<T, siterator>,
					     siterator>;
    using dst_iterator	= std::conditional_t<(vec<S>::size <= vec<T>::size),
					     diterator,
					     cvtup_iterator<diterator> >;

    S	src[] = { 0,  1,  2,  3,  4,  5,  6,  7,
		  8,  9, 10, 11, 12, 13, 14, 15,
		 16, 17, 18, 19, 20, 21, 22, 23,
		 24, 25, 26, 27, 28, 29, 30, 31};
    T	dst[32];

    std::copy(src_iterator(std::cbegin(src)), src_iterator(std::cend(src)),
	      dst_iterator(std::begin(dst)));

    empty();
    
    for (auto x : dst)
	std::cout << ' ' << int(x);
    std::cout << std::endl;
}
    
}
}

int
main()
{
    using namespace	TU;
    
    simd::doJob<int8_t,   int8_t  >();
    simd::doJob<int8_t,   int16_t >();

    simd::doJob<int8_t,   int32_t >();

    simd::doJob<int16_t,  int8_t  >();
    simd::doJob<int16_t,  int16_t >();
    simd::doJob<int16_t,  int32_t >();
    simd::doJob<int16_t,  uint8_t >();

    simd::doJob<int32_t,  int8_t  >();
    simd::doJob<int32_t,  int16_t >();
    simd::doJob<int32_t,  int32_t >();
    simd::doJob<int32_t,  uint8_t >();
#if defined(SSE4) || defined(NEON)
    simd::doJob<int32_t,  uint16_t>();
#endif

    simd::doJob<uint8_t,  int16_t >();
    simd::doJob<uint8_t,  int32_t >();
    simd::doJob<uint8_t,  uint8_t >();
    simd::doJob<uint8_t,  uint16_t>();
    simd::doJob<uint8_t,  uint32_t>();

    simd::doJob<uint16_t, int32_t >();
    simd::doJob<uint16_t, uint16_t>();
    simd::doJob<uint16_t, uint32_t>();

    simd::doJob<uint32_t, uint32_t>();

#if defined(SSE2) || defined(NEON)
#  if defined(SSE4) || defined(NEON)	// 要 vec<int32_t> -> vec<int64_t>
    simd::doJob<int8_t,   int64_t >();
    simd::doJob<int16_t,  int64_t >();
    simd::doJob<int32_t,  int64_t >();
    simd::doJob<uint8_t,  int64_t >();
    simd::doJob<uint16_t, int64_t >();
    simd::doJob<uint32_t, int64_t >();
#  endif
    simd::doJob<uint8_t,  uint64_t>();
    simd::doJob<uint16_t, uint64_t>();
    simd::doJob<uint32_t, uint64_t>();
#endif

#if defined(SSE) || defined(NEON)
    simd::doJob<int8_t,   float   >();
    simd::doJob<float,    int8_t  >();
    simd::doJob<int16_t,  float   >();
    simd::doJob<float,    int16_t >();
    simd::doJob<int32_t,  float   >();
    simd::doJob<float,    int32_t >();

    simd::doJob<uint8_t,  float   >();
    simd::doJob<float,    uint8_t >();
    simd::doJob<uint16_t, float   >();
#  if defined(SSE4) && !defined(AVX)	// 要 vec<int32_t> -> vec<uint16_t>
    simd::doJob<float,    uint16_t>();
#  endif
#endif

#if defined(SSE2)
    simd::doJob<int8_t,   double  >();
    simd::doJob<double,   int8_t  >();
    simd::doJob<int16_t,  double  >();
    simd::doJob<double,   int16_t >();
    simd::doJob<int32_t,  double  >();
    simd::doJob<double,   int32_t >();

    simd::doJob<uint8_t,  double  >();
    simd::doJob<double,   uint8_t >();
    simd::doJob<uint16_t, double  >();
#  if defined(SSE4)			// 要 vec<int32_t> -> vec<uint16_t>
    simd::doJob<double,   uint16_t>();
#  endif

    simd::doJob<double,   float   >();
    simd::doJob<float,    double  >();
#endif

    return 0;
}
