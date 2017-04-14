/*
 *  $Id$
 */
#include "TU/simd/pack.h"
#include "TU/simd/load_store.h"

namespace TU
{
namespace simd
{
template <class PACK>
inline std::enable_if_t<!is_pair<PACK>::value, PACK>
load_pack(const pack_element<PACK>* p)
{
    return load(p);
}
    
template <class PACK>
inline std::enable_if_t<is_pair<PACK>::value, PACK>
load_pack(const pack_element<PACK>* p)
{
    using L = typename PACK::first_type;
    using R = typename PACK::second_type;
    
    auto	x = load_pack<L>(p);
    return std::make_pair(x,
			  load_pack<R>(
			      p + pair_traits<R>::size * pack_vec<R>::size));
}
    
template <class SRC, class DST> void
doJob()
{
    using namespace	std;

    using src_pack = std::conditional_t<
			 (vec<SRC>::size <= vec<DST>::size),
			 pack_target<SRC, vec<DST> >, vec<SRC> >;
    
	
    SRC		src[] = { 0,  1,  2,  3,  4,  5,  6,  7,
			  8,  9, 10, 11, 12, 13, 14, 15,
			 16, 17, 18, 19, 20, 21, 22, 23,
			 24, 25, 26, 27, 28, 29, 30, 31};
    auto	x = load_pack<src_pack>(src);
    cout << "   " << x << "\n-> " << cvt_pack<DST>(x) << endl;

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

    cerr << "\n--- src: int16_t ---" << endl;
    simd::doJob<int16_t, int8_t  >();
    simd::doJob<int16_t, int16_t >();
    simd::doJob<int16_t, int32_t >();
    simd::doJob<int16_t, u_int8_t>();

    cerr << "\n--- src: int32_t ---" << endl;
    simd::doJob<int32_t, int8_t   >();
    simd::doJob<int32_t, int16_t  >();
    simd::doJob<int32_t, int32_t  >();
    simd::doJob<int32_t, u_int8_t >();
#if defined(SSE4) || defined(NEON)
    simd::doJob<int32_t, u_int16_t>();
#endif

    cerr << "\n--- src: u_int8_t ---" << endl;
    simd::doJob<u_int8_t,  int16_t  >();
    simd::doJob<u_int8_t,  int32_t  >();
    simd::doJob<u_int8_t,  u_int8_t >();
    simd::doJob<u_int8_t,  u_int16_t>();
    simd::doJob<u_int8_t,  u_int32_t>();

    cerr << "\n--- src: u_int16_t ---" << endl;
    simd::doJob<u_int16_t, int32_t  >();
    simd::doJob<u_int16_t, u_int16_t>();
    simd::doJob<u_int16_t, u_int32_t>();

    cerr << "\n--- src: u_int32_t ---" << endl;
    simd::doJob<u_int32_t, u_int32_t>();

#if defined(SSE2) || defined(NEON)
#  if defined(SSE4) || defined(NEON)	// 要 vec<int32_t> -> vec<int64_t>
    cerr << "\n--- dst: int64_t ---" << endl;
    simd::doJob<int8_t,    int64_t>();
    simd::doJob<int16_t,   int64_t>();
    simd::doJob<int32_t,   int64_t>();
    simd::doJob<u_int8_t,  int64_t>();
    simd::doJob<u_int16_t, int64_t>();
    simd::doJob<u_int32_t, int64_t>();
#  endif
    cerr << "\n--- dst: u_int64_t ---" << endl;
    simd::doJob<u_int8_t,  u_int64_t>();
    simd::doJob<u_int16_t, u_int64_t>();
    simd::doJob<u_int32_t, u_int64_t>();
#endif

#if defined(SSE) || defined(NEON)
    cerr << "\n--- src or dst: float ---" << endl;
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
    cerr << "\n--- src or dst: double ---" << endl;
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
