/*
 *  $Id$
 */
#include <iomanip>
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
					     cvtdown_mask_iterator<T,
								   siterator>,
					     siterator>;
    using dst_iterator	= std::conditional_t<(vec<S>::size <= vec<T>::size),
					     diterator,
					     cvtup_mask_iterator<diterator> >;

    constexpr S	f = S(~0);
    S		src[] = {0, f, 0, f, 0, 0, f, f,
			 0, 0, 0, 0, f, f, f, f,
			 0, f, 0, f, 0, 0, f, f,
			 0, 0, 0, 0, f, f, f, f};
    T		dst[32];

    std::copy(src_iterator(std::cbegin(src)), src_iterator(std::cend(src)),
	      dst_iterator(std::begin(dst)));
    empty();

    for (auto x : dst)
	std::cout << ' ' << std::setfill('0') << std::setw(2*sizeof(T))
		  << std::hex
		  << (u_int64_t(x) & (u_int64_t(~0) >> (64 - 8*sizeof(T))));
    std::cout << std::endl;
}
    
}
}

int
main()
{
    using namespace	std;
    using namespace	TU;

    cerr << "--- src: u_int8_t, dst: unsinged ---" << endl;
    simd::doJob<u_int8_t,  u_int8_t >();
    simd::doJob<u_int8_t,  u_int16_t>();
    simd::doJob<u_int8_t,  u_int32_t>();
    simd::doJob<u_int8_t,  u_int64_t>();

    cerr << "--- src: u_int16_t, dst: unsinged ---" << endl;
    simd::doJob<u_int16_t, u_int8_t >();
    simd::doJob<u_int16_t, u_int16_t>();
    simd::doJob<u_int16_t, u_int32_t>();
    simd::doJob<u_int16_t, u_int64_t>();

    cerr << "--- src: u_int32_t, dst: unsinged ---" << endl;
    simd::doJob<u_int32_t, u_int8_t >();
    simd::doJob<u_int32_t, u_int16_t>();
    simd::doJob<u_int32_t, u_int32_t>();
    simd::doJob<u_int32_t, u_int64_t>();

    cerr << "--- src: u_int64_t, dst: unsinged ---" << endl;
#if defined(NEON)
    simd::doJob<u_int64_t, u_int8_t >();
    simd::doJob<u_int64_t, u_int16_t>();
    simd::doJob<u_int64_t, u_int32_t>();
#endif
    simd::doJob<u_int64_t, u_int64_t>();
  /*
#if defined(SSE2) || defined(NEON)
    cerr << "--- src: unsigned, dst: float ---" << endl;
    simd::doJob<u_int8_t,  float>();
    simd::doJob<u_int16_t, float>();
#  if !defined(AVX) || defined(AVX2)
    simd::doJob<u_int32_t, float>();
#  endif
    cerr << "--- src: float, dst: unsigned ---" << endl;
    simd::doJob<float, u_int8_t >();
    simd::doJob<float, u_int16_t>();
#  if !defined(AVX) || defined(AVX2)
    simd::doJob<float, u_int32_t>();
#  endif
#endif
  */
#if defined(SSE2) || defined(NEON)
    cerr << "--- src: unsigned, dst: float ---" << endl;
    simd::doJob<u_int8_t,  float>();
    simd::doJob<u_int16_t, float>();
#  if !defined(AVX) || defined(AVX2)
    simd::doJob<u_int32_t, float>();
#  endif
    cerr << "--- src: float, dst: unsigned ---" << endl;
    simd::doJob<float, u_int8_t >();
    simd::doJob<float, u_int16_t>();
#  if !defined(AVX) || defined(AVX2)
    simd::doJob<float, u_int32_t>();
#  endif
#endif

#if defined(SSE2)
    cerr << "--- src: unsigned, dst: double ---" << endl;
    simd::doJob<u_int8_t,  double>();
    simd::doJob<u_int16_t, double>();
    simd::doJob<u_int32_t, double>();
#  if !defined(AVX) || defined(AVX2)
    simd::doJob<u_int64_t, double>();

    cerr << "--- src: double, dst: unsigned ---" << endl;
    simd::doJob<double, u_int64_t>();
#  endif
#endif

    return 0;
}
