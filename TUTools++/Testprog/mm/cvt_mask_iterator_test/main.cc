/*
 *  $Id$
 */
#include "TU/mmInstructions.h"

namespace TU
{
namespace mm
{
template <class SRC, class DST> void
doJob()
{
    using namespace	std;
    using namespace	mm;
    
    typedef SRC						src_type;
    typedef DST						dst_type;
    typedef load_iterator<const src_type*>		siterator;
    typedef store_iterator<dst_type*>			diterator;
    typedef typename boost::mpl::if_c<
	vec<src_type>::size <= vec<dst_type>::size,
	cvtdown_mask_iterator<dst_type, siterator>,
	siterator>::type				src_iterator;
    typedef typename boost::mpl::if_c<
	vec<src_type>::size <= vec<dst_type>::size,
	diterator,
	cvtup_mask_iterator<diterator> >::type		dst_iterator;

    src_type	src[] = { 0,  1,  2,  3,  4,  5,  6,  7,
			  8,  9, 10, 11, 12, 13, 14, 15,
			 16, 17, 18, 19, 20, 21, 22, 23,
			 24, 25, 26, 27, 28, 29, 30, 31};
    dst_type	dst[32];

    copy(src_iterator(src), src_iterator(src + 32), dst_iterator(dst));

    empty();
    
    for (const dst_type* q = dst; q != dst + 32; ++q)
	cout << ' ' << int(*q);
    cout << endl;
}
    
}
}

int
main()
{
    using namespace	TU;
    
    mm::doJob<u_int8_t,  u_int8_t >();
    mm::doJob<u_int8_t,  u_int16_t>();
    mm::doJob<u_int8_t,  u_int32_t>();
    mm::doJob<u_int8_t,  u_int64_t>();

    mm::doJob<u_int16_t, u_int8_t >();
    mm::doJob<u_int16_t, u_int16_t>();
    mm::doJob<u_int16_t, u_int32_t>();
    mm::doJob<u_int16_t, u_int64_t>();

    mm::doJob<u_int32_t, u_int8_t >();
    mm::doJob<u_int32_t, u_int16_t>();
    mm::doJob<u_int32_t, u_int32_t>();
    mm::doJob<u_int32_t, u_int64_t>();

    mm::doJob<u_int64_t, u_int64_t>();

#if defined(SSE2)
    mm::doJob<u_int8_t,  float    >();
    mm::doJob<u_int8_t,  double   >();

    mm::doJob<u_int16_t, float    >();
    mm::doJob<u_int16_t, double   >();

    mm::doJob<u_int32_t, double   >();

#  if defined(AVX2) || !defined(AVX)
    mm::doJob<u_int32_t, float    >();

    mm::doJob<u_int64_t, double   >();
    mm::doJob<double,    u_int64_t>();

    mm::doJob<float,     u_int32_t>();
#  endif
    mm::doJob<float,     u_int16_t>();
    mm::doJob<float,     u_int8_t >();
#endif
    
    return 0;
}
