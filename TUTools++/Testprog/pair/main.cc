/*
 *  $Id$
 */
#include "TU/pair.h"
#ifdef DEMANGLE
#  include <boost/core/demangle.hpp>
#endif

namespace TU
{
template <class S, class T>
struct Derived : public std::pair<S, T>
{
    Derived(S i, T d)	:std::pair<S, T>(i, d), a(10)	{};

    int	a;
};

template <class S, class T> void
arithmetic_test()
{
    using	std::cout;
    using	std::endl;
    using	ivec_t = std::pair<S, S>;
    using	dvec_t = std::pair<T, T>;
    
    cout << "*** arithmetic test ***" << endl;

    ivec_t	a(1, 2), b(10, 20);
    dvec_t	x(1.1, 2.2), y(10.1, 20.2);
    cout << a*a + b << endl;
    cout << x + 2*y << endl;

    std::pair<ivec_t, dvec_t>	ax{a, x}, by{b, y};
    cout << (ax *= by) << endl;
#ifdef DEMANGLE
    using	boost::core::demangle;
    
  //    cout << demangle(typeid(tuple_t<Derived<S, T> >).name()) << endl;
#endif
    Derived<S, T>	c(3, 30.3), d(4, 40.4);
    auto		u = 3*c + d*2;
    cout << u << endl;
}

}
    
int
main()
{
    TU::arithmetic_test<int, double>();
    
    return 0;
}
