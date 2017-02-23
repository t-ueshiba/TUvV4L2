/*
 *  $Id$
 */
#include "TU/Array++.h"
#include <boost/core/demangle.hpp>

namespace TU
{
void
arithmetic_test()
{
    using ivec_t = std::tuple<int, int>;
    using dvec_t = std::tuple<double, double>;
    
    ivec_t	a(1, 2), b(10, 20);
    dvec_t	x(1.1, 2.2), y(10.1, 20.2);
    std::cout << a*a + b << std::endl;
    std::cout << x + 2*y << std::endl;

    std::tuple<ivec_t, dvec_t>	ax{a, x}, by{b, y};
    std::cout << (ax *= by) << std::endl;

    x += a;
    std::cout << x << std::endl;
}

void
zip_iterator_test()
{
    using	boost::core::demangle;
    
    Array2<int>		a(10, 8);
    for (size_t i = 0; i < a.nrow(); ++i)
	for (size_t j = 0; j < a.ncol(); ++j)
	    a[i][j] = 10*i + j;

    Array2<double>	b(10, 8);
    b.resize(10, 8);
    for (size_t i = 0; i < b.nrow(); ++i)
	for (size_t j = 0; j < b.ncol(); ++j)
	    b[i][j] = i + 0.1*j;

    const auto	t = std::make_tuple(std::cref(a), std::cref(b));

    using	tuple_t = decltype(t);
    std::cout << rank<tuple_t>() << std::endl;
    std::cout << demangle(typeid(tuple_t).name())
	      << std::endl;
    std::cout << demangle(typeid(decltype(std::begin(t))).name())
	      << std::endl;
    std::cout << demangle(typeid(decltype(std::begin(*std::begin(t)))).name())
	      << std::endl;
  //const auto	u = make_subrange<2, 4>(t, 1, 3);

    for (const auto& row : t)
    {
	for (const auto& col : row)
	    std::cout << ' ' << col;
	std::cout << std::endl;
    }
}

}
    
int
main()
{
    TU::arithmetic_test();
    TU::zip_iterator_test();
    
    return 0;
}
