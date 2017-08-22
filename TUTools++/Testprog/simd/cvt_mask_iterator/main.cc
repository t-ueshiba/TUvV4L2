/*
 *  $Id$
 */
#include <iomanip>
#include "TU/simd/cvt_iterator.h"
#include "TU/simd/load_store_iterator.h"

namespace TU
{
namespace simd
{
template <class S, class T> void
doJob()
{
    using value_type
	= typename cvt_mask_iterator<T, load_iterator<S> >::value_type;
    
    constexpr S	f = S(~0);
    S		src[] = {0, f, 0, f, 0, 0, f, f,
			 0, 0, 0, 0, f, f, f, f,
			 0, f, 0, f, 0, 0, f, f,
			 0, 0, 0, 0, f, f, f, f};

    std::cout << std::hex;
    std::copy(make_cvt_mask_iterator<T>(make_accessor(std::cbegin(src))),
	      make_cvt_mask_iterator<T>(make_accessor(std::cend(src))),
	      std::ostream_iterator<value_type>(std::cout, " "));
    std::cout << std::endl;
    empty();
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
    simd::doJob<double, uint64_t  >();
#  endif
#endif

    return 0;
}
