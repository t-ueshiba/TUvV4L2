/*
 *  $Id$
 */
#include "TU/simd/cvtdown_iterator.h"
#include "TU/simd/cvtup_iterator.h"
#include "TU/simd/load_iterator.h"
#include "TU/simd/store_iterator.h"

namespace TU
{
namespace simd
{
template <class SRC, class DST> void
doJob()
{
    using namespace	std;
    
    typedef SRC						src_type;
    typedef DST						dst_type;
    typedef load_iterator<const src_type*>		siterator;
    typedef store_iterator<dst_type*>			diterator;
    typedef typename boost::mpl::if_c<
	vec<src_type>::size <= vec<dst_type>::size,
	cvtdown_iterator<dst_type, siterator>,
	siterator>::type				src_iterator;
    typedef typename boost::mpl::if_c<
	vec<src_type>::size <= vec<dst_type>::size,
	diterator, cvtup_iterator<diterator> >::type	dst_iterator;

    src_type	src[] = { 0,  1,  2,  3,  4,  5,  6,  7,
			  8,  9, 10, 11, 12, 13, 14, 15,
			 16, 17, 18, 19, 20, 21, 22, 23,
			 24, 25, 26, 27, 28, 29, 30, 31};
    dst_type	dst[32];

    copy(src_iterator(&src[0]), src_iterator(&src[32]), dst_iterator(&dst[0]));

    empty();
    
    for (const dst_type* q = dst; q != dst + 32; ++q)
	cout << ' ' << int(*q);
    cout << endl;
}
    
}
}

int
main()
{
    using namespace	TU;
    
    simd::doJob<int8_t,  int8_t >();
    simd::doJob<int8_t,  int16_t>();

    simd::doJob<int8_t,  int32_t>();

    simd::doJob<int16_t, int8_t  >();
    simd::doJob<int16_t, int16_t >();
    simd::doJob<int16_t, int32_t >();
    simd::doJob<int16_t, u_int8_t>();

    simd::doJob<int32_t, int8_t   >();
    simd::doJob<int32_t, int16_t  >();
    simd::doJob<int32_t, int32_t  >();
    simd::doJob<int32_t, u_int8_t >();
#if defined(SSE4) || defined(NEON)
    simd::doJob<int32_t, u_int16_t>();
#endif

    simd::doJob<u_int8_t,  int16_t  >();
    simd::doJob<u_int8_t,  int32_t  >();
    simd::doJob<u_int8_t,  u_int8_t >();
    simd::doJob<u_int8_t,  u_int16_t>();
    simd::doJob<u_int8_t,  u_int32_t>();

    simd::doJob<u_int16_t, int32_t  >();
    simd::doJob<u_int16_t, u_int16_t>();
    simd::doJob<u_int16_t, u_int32_t>();

    simd::doJob<u_int32_t, u_int32_t>();

#if defined(SSE2) || defined(NEON)
#  if defined(SSE4) || defined(NEON)	// 要 vec<int32_t> -> vec<int64_t>
    simd::doJob<int8_t,    int64_t>();
    simd::doJob<int16_t,   int64_t>();
    simd::doJob<int32_t,   int64_t>();
    simd::doJob<u_int8_t,  int64_t>();
    simd::doJob<u_int16_t, int64_t>();
    simd::doJob<u_int32_t, int64_t>();
#  endif
    simd::doJob<u_int8_t,  u_int64_t>();
    simd::doJob<u_int16_t, u_int64_t>();
    simd::doJob<u_int32_t, u_int64_t>();
#endif

#if defined(SSE) || defined(NEON)
    simd::doJob<int8_t,    float    >();
    simd::doJob<float,     int8_t   >();
    simd::doJob<int16_t,   float    >();
    simd::doJob<float,     int16_t  >();
    simd::doJob<int32_t,   float    >();
    simd::doJob<float,     int32_t  >();

    simd::doJob<u_int8_t,  float    >();
    simd::doJob<float,     u_int8_t >();
    simd::doJob<u_int16_t, float    >();
#  if defined(SSE4) && !defined(AVX)	// 要 vec<int32_t> -> vec<u_int16_t>
    simd::doJob<float,     u_int16_t>();
#  endif
#endif

#if defined(SSE2)
    simd::doJob<int8_t,    double   >();
    simd::doJob<double,    int8_t   >();
    simd::doJob<int16_t,   double   >();
    simd::doJob<double,    int16_t  >();
    simd::doJob<int32_t,   double   >();
    simd::doJob<double,    int32_t  >();

    simd::doJob<u_int8_t,  double   >();
    simd::doJob<double,    u_int8_t >();
    simd::doJob<u_int16_t, double   >();
#  if defined(SSE4)			// 要 vec<int32_t> -> vec<u_int16_t>
    simd::doJob<double,    u_int16_t>();
#  endif

    simd::doJob<double,    float    >();
    simd::doJob<float,     double   >();
#endif

    return 0;
}
