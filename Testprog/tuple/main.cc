/*
 *  $Id$
 */
#include <boost/core/demangle.hpp>
#include "TU/Array++.h"

namespace TU
{
template <class T> auto
demangle()
{
    return boost::core::demangle(typeid(T).name());
}
    
template <class S, class T> void
arithmetic_test()
{
    using	std::cout;
    using	std::endl;
    using	ivec_t = std::tuple<S, S>;
    using	dvec_t = std::tuple<T, T>;
    
    cout << "*** arithmetic test ***" << endl;

    ivec_t	a(1, 2), b(10, 20);
    dvec_t	x(1.1, 2.2), y(10.1, 20.2);
    cout << a*a + b << endl;
    cout << x + 2*y << endl;

    std::tuple<ivec_t, dvec_t>	ax{a, x}, by{b, y};
    cout << (ax *= by) << endl;
}

void
zip_iterator_test()
{
    using	std::cout;
    using	std::endl;

    cout << "*** zip iterator test ***" << endl;
    
    Array2<int>		a(10, 8);
    for (size_t i = 0; i < a.nrow(); ++i)
	for (size_t j = 0; j < a.ncol(); ++j)
	    a[i][j] = 10*i + j;

    Array2<double>	b(10, 8);
    b.resize(10, 8);
    for (size_t i = 0; i < b.nrow(); ++i)
	for (size_t j = 0; j < b.ncol(); ++j)
	    b[i][j] = i + 0.1*j;

    const auto	t = std::make_tuple(std::ref(a), std::ref(b));
  //const auto	u = make_range<3>(std::begin(t) + 2);

    auto	w = slice<2, 4>(t, 1, 3);

#ifdef DEMANGLE
    cout << endl;
    cout << "rank(tuple_t): "
	 << rank<decltype(t)>() << endl;
    cout << "tuple_t: "
	 << demangle<decltype(t)>() << endl << endl;
    cout << "begin(t): "
	 << demangle<decltype(begin(t))>() << endl << endl;
    cout << "begin(*begin(t): "
	 << demangle<decltype(begin(*begin(t)))>()
	 << endl << endl;

  //cout << demangle(typeid(decltype(u)).name()) << endl << endl;
    cout << demangle<decltype(w)>() << endl << endl;
#endif
  //cout << u << endl;
    cout << w << endl;
    
    w = std::make_tuple(100, 1000.0);

    for (const auto& row : t)
    {
	for (const auto& col : row)
	    cout << ' ' << col;
	cout << endl;
    }
}

}
    
int
main()
{
    TU::arithmetic_test<int, double>();
    TU::zip_iterator_test();
    
    return 0;
}
