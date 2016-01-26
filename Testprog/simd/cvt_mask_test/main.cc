/*
 *  $Id$
 */
#include <iomanip>
#include "TU/simd/cvt.h"
#include "TU/simd/load_store.h"
#include "TU/simd/compare.h"

namespace TU
{
namespace simd
{
template <class T> void
print(vec<T> x)
{
    using namespace	std;
    
    for (size_t i = 0; i != vec<T>::size; ++i)
	cout << ' ' << setfill('0') << setw(2*sizeof(T)) << hex
	     << (u_int64_t(x[i]) & (u_int64_t(~0) >> (64 - 8*sizeof(T))));
    cout << endl;
}

template <class S, size_t N, class T,
	  typename std::enable_if<N == vec<T>::size/vec<S>::size>::type*
	  = nullptr> void
cvtup_mask_all(vec<T>)
{
}

template <class S, size_t N=0, class T,
	  typename std::enable_if<N != vec<T>::size/vec<S>::size>::type*
	  = nullptr> void
cvtup_mask_all(vec<T> x)
{
    using namespace	std;

    cout << "  " << N << ':';
    print(cvt<S, N>(x));

    cvtup_mask_all<S, N+1>(x);
}
    
template <class SRC, class DST> void
doJob()
{
    using namespace	std;
    
    const SRC	src[] = {0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07,
			 0x08, 0x09, 0x0a, 0x0b, 0x0c, 0x0d, 0x0e, 0x0f,
			 0x80, 0x81, 0x82, 0x83, 0x84, 0x85, 0x86, 0x87,
			 0x88, 0x89, 0x8a, 0x8b, 0x8c, 0x8d, 0x8e, 0x8f};
    const auto	x = load(src);

    cout << "src:";
    print(x);

    cvtup_mask_all<DST>(x);
    cout << endl;

    empty();
}
    
template <class SRC> void
doJobF()
{
    using namespace	std;
    
    typedef complementary_mask_type<SRC>	DST;

    const SRC	a[] = {0,  0, 0, 0,  0,  0, 0, 0};
    const SRC	b[] = {1, -1, 1, 1, -1, -1, 1, 1};
    const auto	x = load(a);
    const auto	y = load(b);

    cvtup_mask_all<DST>(x < y);
    cout << endl;

    empty();
}
    
}
}

int
main()
{
    using namespace	std;
    using namespace	TU;
    
  //simd::doJob<u_int8_t, u_int8_t >();
    simd::doJob<u_int8_t, u_int16_t>();
    simd::doJob<u_int8_t, u_int32_t>();
    simd::doJob<u_int8_t, u_int64_t>();

  //simd::doJob<u_int16_t, u_int16_t>();
    simd::doJob<u_int16_t, u_int32_t>();
    simd::doJob<u_int16_t, u_int64_t>();

  //simd::doJob<u_int32_t, u_int32_t>();
    simd::doJob<u_int32_t, u_int64_t>();

  //simd::doJob<u_int64_t, u_int64_t>();

#if defined(SSE2) || defined(NEON)
    simd::doJobF<float>();
#endif
#if defined(SSE2)
    simd::doJobF<double>();
#endif
    
    return 0;
}
