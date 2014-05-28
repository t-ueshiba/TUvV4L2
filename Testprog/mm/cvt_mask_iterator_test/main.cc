/*
 *  $Id$
 */
#include <iomanip>
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

    u_char	p[] = {0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07,
		       0x08, 0x09, 0x0a, 0x0b, 0x0c, 0x0d, 0x0e, 0x0f,
		       0x80, 0x81, 0x82, 0x83, 0x84, 0x85, 0x86, 0x87,
		       0x88, 0x89, 0x8a, 0x8b, 0x8c, 0x8d, 0x8e, 0x8f};
    dst_type	dst[32];
    size_t	n = sizeof(p)/sizeof(src_type);
    
    copy(src_iterator((const src_type*)p),
	 src_iterator((const src_type*)p + n), dst_iterator(dst));

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

    cerr << "--- src: int8_t, dst: singed ---" << endl;
    mm::doJob<int8_t,	 int8_t >();
    mm::doJob<int8_t,	 int16_t>();
    mm::doJob<int8_t,	 int32_t>();
    mm::doJob<int8_t,	 int64_t>();

    cerr << "--- src: int8_t, dst: unsinged ---" << endl;
    mm::doJob<int8_t,	 u_int8_t >();
    mm::doJob<int8_t,	 u_int16_t>();
    mm::doJob<int8_t,	 u_int32_t>();
    mm::doJob<int8_t,	 u_int64_t>();

    cerr << "--- src: u_int8_t, dst: singed ---" << endl;
    mm::doJob<u_int8_t,  int8_t >();
    mm::doJob<u_int8_t,  int16_t>();
    mm::doJob<u_int8_t,  int32_t>();
    mm::doJob<u_int8_t,  int64_t>();

    cerr << "--- src: u_int8_t, dst: unsinged ---" << endl;
    mm::doJob<u_int8_t,  u_int8_t >();
    mm::doJob<u_int8_t,  u_int16_t>();
    mm::doJob<u_int8_t,  u_int32_t>();
    mm::doJob<u_int8_t,  u_int64_t>();

    cerr << "--- src: int16_t, dst: singed ---" << endl;
    mm::doJob<int16_t,   int8_t >();
    mm::doJob<int16_t,   int16_t>();
    mm::doJob<int16_t,   int32_t>();
    mm::doJob<int16_t,   int64_t>();

    cerr << "--- src: int16_t, dst: unsinged ---" << endl;
    mm::doJob<int16_t,   u_int8_t >();
    mm::doJob<int16_t,   u_int16_t>();
    mm::doJob<int16_t,   u_int32_t>();
    mm::doJob<int16_t,   u_int64_t>();

    cerr << "--- src: u_int16_t, dst: singed ---" << endl;
    mm::doJob<u_int16_t, int8_t >();
    mm::doJob<u_int16_t, int16_t>();
    mm::doJob<u_int16_t, int32_t>();
    mm::doJob<u_int16_t, int64_t>();

    cerr << "--- src: u_int16_t, dst: unsinged ---" << endl;
    mm::doJob<u_int16_t, u_int8_t >();
    mm::doJob<u_int16_t, u_int16_t>();
    mm::doJob<u_int16_t, u_int32_t>();
    mm::doJob<u_int16_t, u_int64_t>();

    cerr << "--- src: int32_t, dst: singed ---" << endl;
    mm::doJob<int32_t,   int8_t >();
    mm::doJob<int32_t,   int16_t>();
    mm::doJob<int32_t,   int32_t>();
    mm::doJob<int32_t,   int64_t>();

    cerr << "--- src: int32_t, dst: unsinged ---" << endl;
    mm::doJob<int32_t,   u_int8_t >();
    mm::doJob<int32_t,   u_int16_t>();
    mm::doJob<int32_t,   u_int32_t>();
    mm::doJob<int32_t,   u_int64_t>();

    cerr << "--- src: u_int32_t, dst: singed ---" << endl;
    mm::doJob<u_int32_t, int8_t >();
    mm::doJob<u_int32_t, int16_t>();
    mm::doJob<u_int32_t, int32_t>();
    mm::doJob<u_int32_t, int64_t>();

    cerr << "--- src: u_int32_t, dst: unsinged ---" << endl;
    mm::doJob<u_int32_t, u_int8_t >();
    mm::doJob<u_int32_t, u_int16_t>();
    mm::doJob<u_int32_t, u_int32_t>();
    mm::doJob<u_int32_t, u_int64_t>();

    cerr << "--- src: int64_t ---" << endl;
    mm::doJob<int64_t,   int64_t>();
    mm::doJob<int64_t,   u_int64_t>();
    cerr << "--- src: u_int64_t ---" << endl;
    mm::doJob<u_int64_t, int64_t>();
    mm::doJob<u_int64_t, u_int64_t>();

#if defined(SSE2)
    mm::doJob<int8_t,    float    >();
    mm::doJob<int16_t,	 float    >();

    mm::doJob<u_int8_t,  float    >();
    mm::doJob<u_int16_t, float    >();

    mm::doJob<float,     int16_t  >();
    mm::doJob<float,     int8_t   >();

    mm::doJob<float,     u_int16_t>();
    mm::doJob<float,     u_int8_t >();

    mm::doJob<int8_t,    double   >();
    mm::doJob<int16_t,	 double   >();
    mm::doJob<int32_t,   double   >();

    mm::doJob<u_int8_t,  double   >();
    mm::doJob<u_int16_t, double   >();
    mm::doJob<u_int32_t, double   >();

#  if defined(AVX2) || !defined(AVX)
    mm::doJob<int32_t,	 float    >();
    mm::doJob<u_int32_t, float    >();

    mm::doJob<float,     int32_t  >();
    mm::doJob<float,     u_int32_t>();

    mm::doJob<int64_t,	 double   >();
    mm::doJob<u_int64_t, double   >();

    mm::doJob<double,    int64_t  >();
    mm::doJob<double,    u_int64_t>();
#  endif
#endif

    return 0;
}
