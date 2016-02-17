#include "TU/Array++.h"
#include "TU/simd/lookup.h"

namespace TU
{
namespace simd
{
template <class P, class T> void
doJob()
{
    using	std::cout;
    using	std::endl;
    
    typedef Array2<Array<P> >	array2_type;

    P		p[] = { 0,  1,  2,  3,  4,  5,  6,  7,
			8,  9, 10, 11, 12, 13, 14, 15,
		       16, 17, 18, 19, 20, 21, 22, 23,
		       24, 25, 26, 27, 28, 29, 30, 31,
		       32, 33, 34, 35, 36, 37, 38, 39,
		       40, 41, 42, 43, 44, 45, 46, 47,
		       48, 49, 50, 51, 52, 53, 54, 55,
		       56, 57, 58, 59, 60, 61, 62, 63};
    T		i[] = {7, 6, 5, 4, 3, 2, 1, 0, 1, 3, 5, 7, 0, 2, 4, 6};
    auto	idx = load(i);
    cout << lookup(p, idx) << endl;

    array2_type	a(p, 8, 8);
    T		j[] = {7, 5, 3, 1, 6, 4, 2, 0, 1, 3, 5, 7, 0, 2, 4, 6};
    auto	col = load(j);
    cout << lookup(a.data(), idx, col, a.ncol()) << endl << endl;

}
    
}
}

int
main()
{
    using namespace	std;
    
    cerr << "--- idx: int16_t ---" << endl;
    TU::simd::doJob<u_int8_t,  int16_t>();
    TU::simd::doJob<u_int16_t, int16_t>();

#if defined(SSE4)
    cerr << "--- idx: int32_t ---" << endl;
    TU::simd::doJob<u_int8_t,  int32_t>();
    TU::simd::doJob<u_int16_t, int32_t>();
    TU::simd::doJob<u_int32_t, int32_t>();
    TU::simd::doJob<float,     int32_t>();
#endif
    
    return 0;
}
