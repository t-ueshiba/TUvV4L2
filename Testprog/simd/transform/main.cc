#include "TU/simd/transform.h"
#include "TU/simd/load_iterator.h"
#include "TU/simd/store_iterator.h"
#include "TU/simd/arithmetic.h"

namespace TU
{
struct sum
{
    template <class HEAD>
    auto	operator ()(const boost::tuples::cons<
				      HEAD, boost::tuples::null_type>& x) const
		{
		    return x.get_head();
		}
    template <class HEAD, class TAIL>
    auto	operator ()(const boost::tuples::cons<HEAD, TAIL>& x) const
		{
		    return x.get_head() + (*this)(x.get_tail());
		}
};

namespace simd
{
template <class T, class O, class S0, class S1, class S2> void
doJob()
{
    using	namespace std;

    typedef boost::tuple<vec<T>, vec<T>, vec<T> >	target_type;
    
    S0	x[] = { 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15,
	       16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31};
    S1	y[] = { 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15,
	       16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31};
    S2	z[] = { 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15,
	       16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31};
    O	w[32];

    simd::copy<T>(ostream_iterator<target_type>(cout, "\n"),
		  make_load_iterator(cbegin(x)),
		  make_load_iterator(cend(x)),
		  make_load_iterator(cbegin(y)),
		  make_load_iterator(cbegin(z)));
    cout << endl;
    
    simd::transform<T>(sum(),
		       make_store_iterator(begin(w)),
		       make_load_iterator(cbegin(x)),
		       make_load_iterator(cend(x)),
		       make_load_iterator(cbegin(y)),
		       make_load_iterator(cbegin(z)));
		       
    std::copy(make_load_iterator(cbegin(w)), make_load_iterator(cend(w)),
	      ostream_iterator<vec<O> >(cout, "\n"));
}
    
}
}

int
main()
{
    TU::simd::doJob<double, int8_t, int8_t, int16_t, int32_t>();
    return 0;
}
