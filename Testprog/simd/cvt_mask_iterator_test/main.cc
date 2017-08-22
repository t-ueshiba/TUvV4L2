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
		  << (uint64_t(x) & (uint64_t(~0) >> (64 - 8*sizeof(T))));
    std::cout << std::endl;
}
    
}
}

int
main()
{
    using namespace	std;
    using namespace	TU;

    cerr << "--- src: uint8_t, dst: unsinged ---" << endl;
    simd::doJob<uint8_t,  uint8_t >();
    simd::doJob<uint8_t,  uint16_t>();
    simd::doJob<uint8_t,  uint32_t>();
    simd::doJob<uint8_t,  uint64_t>();

    cerr << "--- src: uint16_t, dst: unsinged ---" << endl;
    simd::doJob<uint16_t, uint8_t >();
    simd::doJob<uint16_t, uint16_t>();
    simd::doJob<uint16_t, uint32_t>();
    simd::doJob<uint16_t, uint64_t>();

    cerr << "--- src: uint32_t, dst: unsinged ---" << endl;
    simd::doJob<uint32_t, uint8_t >();
    simd::doJob<uint32_t, uint16_t>();
    simd::doJob<uint32_t, uint32_t>();
    simd::doJob<uint32_t, uint64_t>();

    cerr << "--- src: uint64_t, dst: unsinged ---" << endl;
#if defined(NEON)
    simd::doJob<uint64_t, uint8_t >();
    simd::doJob<uint64_t, uint16_t>();
    simd::doJob<uint64_t, uint32_t>();
#endif
    simd::doJob<uint64_t, uint64_t>();
  /*
#if defined(SSE2) || defined(NEON)
    cerr << "--- src: unsigned, dst: float ---" << endl;
    simd::doJob<uint8_t,  float>();
    simd::doJob<uint16_t, float>();
#  if !defined(AVX) || defined(AVX2)
    simd::doJob<uint32_t, float>();
#  endif
    cerr << "--- src: float, dst: unsigned ---" << endl;
    simd::doJob<float, uint8_t >();
    simd::doJob<float, uint16_t>();
#  if !defined(AVX) || defined(AVX2)
    simd::doJob<float, uint32_t>();
#  endif
#endif
  */
#if defined(SSE2) || defined(NEON)
    cerr << "--- src: unsigned, dst: float ---" << endl;
    simd::doJob<uint8_t,  float   >();
    simd::doJob<uint16_t, float   >();
#  if !defined(AVX) || defined(AVX2)
    simd::doJob<uint32_t, float   >();
#  endif
    cerr << "--- src: float, dst: unsigned ---" << endl;
    simd::doJob<float,    uint8_t >();
    simd::doJob<float,    uint16_t>();
#  if !defined(AVX) || defined(AVX2)
    simd::doJob<float,    uint32_t>();
#  endif
#endif

#if defined(SSE2)
    cerr << "--- src: unsigned, dst: double ---" << endl;
    simd::doJob<uint8_t,  double  >();
    simd::doJob<uint16_t, double  >();
    simd::doJob<uint32_t, double  >();
#  if !defined(AVX) || defined(AVX2)
    simd::doJob<uint64_t, double  >();

    cerr << "--- src: double, dst: unsigned ---" << endl;
    simd::doJob<double,   uint64_t>();
#  endif
#endif

    return 0;
}
