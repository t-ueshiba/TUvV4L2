/*
 *  $Id$
 */
#include "TU/pair.h"

namespace TU
{
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
}

}
    
int
main()
{
    TU::arithmetic_test<int, double>();
    
    return 0;
}
