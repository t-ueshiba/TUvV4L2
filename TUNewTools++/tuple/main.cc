/*
 *  $Id$
 */
#include "TU/tuple.h"
#include "TU/Array++.h"

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
    std::cout << (ax * by) << std::endl;

    x += a;
    std::cout << x << std::endl;
}

void
zip_iterator_test()
{
    std::tuple<Array2<int>, Array2<double> >	t;
    
    auto&	a = std::get<0>(t);
    a.resize(10, 8);
    for (size_t i = 0; i < a.nrow(); ++i)
	for (size_t j = 0; j < a.ncol(); ++j)
	    a[i][j] = 10*i + j;

    auto&	b = std::get<1>(t);
    b.resize(10, 8);
    for (size_t i = 0; i < b.nrow(); ++i)
	for (size_t j = 0; j < b.ncol(); ++j)
	    b[i][j] = i + 0.1*j;

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
