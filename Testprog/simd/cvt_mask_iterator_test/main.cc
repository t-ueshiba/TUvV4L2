/*
 *  $Id$
 */
#include <iomanip>
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
    typedef load_iterator<src_type>			siterator;
    typedef store_iterator<dst_type>			diterator;
    typedef typename boost::mpl::if_c<
	vec<src_type>::size <= vec<dst_type>::size,
	cvtdown_mask_iterator<dst_type, siterator>,
	siterator>::type				src_iterator;
    typedef typename boost::mpl::if_c<
	vec<src_type>::size <= vec<dst_type>::size,
	diterator,
	cvtup_mask_iterator<diterator> >::type		dst_iterator;

    constexpr src_type	f = src_type(~0);
    src_type		src[] = {0, f, 0, f, 0, 0, f, f,
				 0, 0, 0, 0, f, f, f, f,
				 0, f, 0, f, 0, 0, f, f,
				 0, 0, 0, 0, f, f, f, f};
    dst_type		dst[32];
    size_t		n = sizeof(src)/sizeof(src[0]);

    copy(src_iterator(src), src_iterator(src + n),
	 dst_iterator(dst));

    empty();

    for (const dst_type* q = dst; q != dst + n; ++q)
	cout << ' ' << setfill('0') << setw(2*sizeof(dst_type)) << hex
	     << (u_int64_t(*q) & (u_int64_t(~0) >> (64 - 8*sizeof(dst_type))));
    cout << endl;
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
