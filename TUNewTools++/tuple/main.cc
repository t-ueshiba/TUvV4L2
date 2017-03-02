/*
 *  $Id$
 */
#include "TU/Array++.h"
#ifdef DEMANGLE
#  include <boost/core/demangle.hpp>
#endif

namespace TU
{
template <class S, class T>
struct Derived : public std::tuple<S, T>
{
    Derived(S i, T d)	:std::tuple<S, T>(i, d), a(10)	{};

    int	a;
};

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
#ifdef DEMANGLE
    using	boost::core::demangle;
    
    cout << demangle(typeid(tuple_t<Derived<S, T> >).name()) << endl;
#endif
    Derived<S, T>	c(3, 30.3), d(4, 40.4);
    auto		u = 3*c + d*2;
    cout << u << endl;
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
    const auto	w = slice<2, 4>(t, 1, 3);
#ifdef DEMANGLE
    using	boost::core::demangle;
    
    cout << endl;
    cout << "rank(tuple_t): "
	 << rank<decltype(t)>() << endl;
    cout << "tuple_t: "
	 << demangle(typeid(decltype(t)).name()) << endl << endl;
    cout << "begin(t): "
	 << demangle(typeid(decltype(std::begin(t))).name()) << endl << endl;
    cout << "begin(*begin(t): "
	 << demangle(typeid(decltype(std::begin(*std::begin(t)))).name())
	 << endl << endl;

  //cout << demangle(typeid(decltype(u)).name()) << endl << endl;
    cout << demangle(typeid(decltype(w)).name()) << endl << endl;
#endif
  //cout << u << endl;
    
    cout << w << endl;
    fill(w, std::make_tuple(100, 1000.0f));

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
