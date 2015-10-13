/*
 *  $Id$
 */
#include <iomanip>
#include "TU/simd/cvt_mask.h"
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
    
    for (size_t i = vec<T>::size; i-- > 0; )
	cout << ' ' << setfill('0') << setw(2*sizeof(T)) << hex
	     << (u_int64_t(x[i]) & (u_int64_t(~0) >> (64 - 8*sizeof(T))));
    cout << endl;
}

template <class SRC, class DST, size_t N=vec<SRC>::size/vec<DST>::size>
struct cvtup_mask_all
{
    static void	exec(vec<SRC> x)
		{
		    using namespace	std;

		    cerr << setw(3) << N-1 << ':';
		    print(cvt_mask<DST, N-1>(x));
		    cvtup_mask_all<SRC, DST, N-1>().exec(x);
		}
};

template <class SRC, class DST>
struct cvtup_mask_all<SRC, DST, 0>
{
    static void	exec(vec<SRC> x)	{}
};

template <class SRC, class DST> void
doJob()
{
    using namespace	std;
    
    typedef SRC				src_type;
    typedef DST				dst_type;
    typedef simd::mask_type<src_type>	mask_type;

    u_char	src[] = {0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07,
			 0x08, 0x09, 0x0a, 0x0b, 0x0c, 0x0d, 0x0e, 0x0f,
			 0x80, 0x81, 0x82, 0x83, 0x84, 0x85, 0x86, 0x87,
			 0x88, 0x89, 0x8a, 0x8b, 0x8c, 0x8d, 0x8e, 0x8f};

    vec<src_type>	x = load<false>(src);

    cerr << "src:";
    print(x);
    cvtup_mask_all<mask_type, dst_type>::exec(x);
    
    empty();

    cerr << endl;
}
    
template <class SRC> void
doJobF()
{
    using namespace	std;
    
    typedef SRC					src_type;
    typedef complementary_mask_type<SRC>	dst_type;
    typedef simd::mask_type<src_type>		mask_type;

    src_type	a[] = {0,  0, 0, 0,  0,  0, 0, 0};
    src_type	b[] = {1, -1, 1, 1, -1, -1, 1, 1};

    vec<src_type>	x = load<false>(a),
			y = load<false>(b);

    cvtup_mask_all<mask_type, dst_type>::exec(x < y);
    
    empty();

    cerr << endl;
}
    
}
}

int
main()
{
    using namespace	std;
    using namespace	TU;
    
    simd::doJob<u_int8_t, u_int16_t>();
    simd::doJob<u_int8_t, u_int32_t>();
    simd::doJob<u_int8_t, u_int64_t>();

#if defined(SSE2) || defined(NEON)
    simd::doJobF<float>();
#endif
#if defined(SSE2)
    simd::doJobF<double>();
#endif
    
    return 0;
}
