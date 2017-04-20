/*
 *  $Id$
 */
#include "TU/simd/dup.h"
#include "TU/simd/load_store.h"
#include "TU/simd/cvt.h"

namespace TU
{
namespace simd
{
template <class T> void
doJob()
{
    T		p[] = {0, 1,  2,  3,  4,  5,  6,  7,
		       8, 9, 10, 11, 12, 13, 14, 15};
    vec<T>	x = load(p);

    std::cout << x << std::endl
	      << dup<false>(x) << std::endl
	      << dup<true>(x) << std::endl
	      << std::endl;
}

template <class T> void
doMaskJob()
{
    using U = upper_type<T>;
    
    T		p[] = {0, ~0, ~0, 0, ~0, 0, 0, ~0,
		       ~0, ~0, 0, 0, 0, 0, ~0, ~0};
    vec<T>	x = load(p);

    std::cout << std::hex << x << std::endl
	      << cast<T>(cvt<U, false, true>(x)) << std::endl
	      << cast<T>(cvt<U, true,  true>(x)) << std::endl
	      << std::endl;
}

}
}

int
main()
{
    using namespace	TU::simd;

    doJob<int8_t>();
    doJob<int16_t>();
    doJob<int32_t>();
    doJob<float>();
#if defined(SSE2)
    doJob<double>();
#endif

    doMaskJob<uint8_t >();
    doMaskJob<uint16_t>();
    doMaskJob<uint32_t>();
    
    return 0;
}
