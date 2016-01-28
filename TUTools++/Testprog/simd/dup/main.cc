/*
 *  $Id$
 */
#include "TU/simd/dup.h"
#include "TU/simd/load_iterator.h"
#include "TU/simd/cvt.h"

namespace TU
{
namespace simd
{
template <class T> void
doJob()
{
    T		p[] = {1, 2, 3, 4, 5, 6, 7, 8};
    vec<T>	x = *make_load_iterator(p);

    std::cout << x << std::endl
	      << dup<false>(x) << std::endl
	      << dup<true>(x) << std::endl
	      << std::endl;
}

template <class T> void
doMaskJob()
{
    using U = upper_type<T>;
    
    T		p[] = {0, ~0, ~0, 0, ~0, 0, 0, ~0, 0, ~0, ~0, 0, ~0, 0, 0, ~0};
    vec<T>	x = *make_load_iterator(p);

    std::cout << std::hex << x << std::endl
	      << cvt<U, false, true>(x) << std::endl
	      << cvt<U, true,  true>(x) << std::endl
	      << std::endl;
}

}
}

int
main()
{
    using namespace	TU::simd;

    doJob<int>();
    doJob<float>();
    doJob<double>();

    doMaskJob<u_int>();
    
    return 0;
}
