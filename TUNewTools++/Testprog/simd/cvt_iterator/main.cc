/*
 *  $Id$
 */
#include "TU/simd/cvt_iterator.h"
#include "TU/simd/load_iterator.h"
#include "TU/simd/store_iterator.h"
#include <typeinfo>

namespace TU
{
namespace simd
{
template <class SRC, class DST> void
doJob()
{
    using namespace	std;
    
    typedef SRC							src_type;
    typedef DST							dst_type;
    typedef cvt_iterator<dst_type,
			 load_iterator<const src_type*> >	siterator;
    typedef typename std::iterator_traits<siterator>::value_type
								value_type;
    
    src_type	src[] = { 0,  1,  2,  3,  4,  5,  6,  7,
			  8,  9, 10, 11, 12, 13, 14, 15,
			 16, 17, 18, 19, 20, 21, 22, 23,
			 24, 25, 26, 27, 28, 29, 30, 31};

    copy(siterator(cbegin(src)), siterator(cend(src)),
	 ostream_iterator<value_type>(cout, " "));
    cout << endl;
  /*
    cout << typeid(typename siterator::upmost_type).name()	   << ", "
	 << typeid(typename siterator::complementary_type).name()  << ", "
	 << typeid(typename siterator::integral_type).name()	   << ", "
	 << typeid(typename siterator::unsigned_lower_type).name() << endl;
  */
    empty();
}
    
}
}

int
main()
{
    using namespace	std;
    using namespace	TU;
    
    cerr << "--- src: int8_t ---" << endl;
    simd::doJob<int8_t,  int8_t >();
    simd::doJob<int8_t,  int16_t>();
    simd::doJob<int8_t,  int32_t>();

    cerr << "--- src: int16_t ---" << endl;
    simd::doJob<int16_t, int8_t  >();
    simd::doJob<int16_t, int16_t >();
    simd::doJob<int16_t, int32_t >();
    simd::doJob<int16_t, u_int8_t>();

    cerr << "--- src: int32_t ---" << endl;
    simd::doJob<int32_t, int8_t   >();
    simd::doJob<int32_t, int16_t  >();
    simd::doJob<int32_t, int32_t  >();
    simd::doJob<int32_t, u_int8_t >();
#if defined(SSE4) || defined(NEON)
    simd::doJob<int32_t, u_int16_t>();
#endif

    cerr << "--- src: u_int8_t ---" << endl;
    simd::doJob<u_int8_t,  int16_t  >();
    simd::doJob<u_int8_t,  int32_t  >();
    simd::doJob<u_int8_t,  u_int8_t >();
    simd::doJob<u_int8_t,  u_int16_t>();
    simd::doJob<u_int8_t,  u_int32_t>();

    cerr << "--- src: u_int16_t ---" << endl;
    simd::doJob<u_int16_t, int32_t  >();
    simd::doJob<u_int16_t, u_int16_t>();
    simd::doJob<u_int16_t, u_int32_t>();

    cerr << "--- src: u_int32_t ---" << endl;
    simd::doJob<u_int32_t, u_int32_t>();

#if defined(SSE2) || defined(NEON)
#  if defined(SSE4) || defined(NEON)	// 要 vec<int32_t> -> vec<int64_t>
    cerr << "--- dst: int64_t ---" << endl;
    simd::doJob<int8_t,    int64_t>();
    simd::doJob<int16_t,   int64_t>();
    simd::doJob<int32_t,   int64_t>();
    simd::doJob<u_int8_t,  int64_t>();
    simd::doJob<u_int16_t, int64_t>();
    simd::doJob<u_int32_t, int64_t>();
#  endif
    cerr << "--- dst: u_int64_t ---" << endl;
    simd::doJob<u_int8_t,  u_int64_t>();
    simd::doJob<u_int16_t, u_int64_t>();
    simd::doJob<u_int32_t, u_int64_t>();
#endif

#if defined(SSE) || defined(NEON)
    cerr << "--- src or dst: float ---" << endl;
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
    cerr << "--- src or dst: double ---" << endl;
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
