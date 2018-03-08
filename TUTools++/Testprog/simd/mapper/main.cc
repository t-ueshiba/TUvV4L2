/*
 *  $Id$
 */
#include "TU/simd/mapper.h"
#include "TU/simd/arithmetic.h"
#include "TU/algorithm.h"
#include <boost/core/demangle.hpp>

namespace TU
{
template <class T> inline auto
demangle()
{
    return boost::core::demangle(typeid(T).name());
}
    
namespace simd
{
template <class IN, class OUT> void
test_assign()
{
    IN	x[] = { 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15,
	       16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31};
    OUT	w[32];
    std::fill(std::begin(w), std::end(w), 0);

    auto	o = make_mapper<OUT>([](const auto& x, auto&& y){ y = x; },
				     std::make_tuple(
					 make_accessor(std::cbegin(x)),
					 make_accessor(std::begin(w))));
    for_each<size(w)/o.size>([&o]{ o(); });

    std::copy(std::cbegin(w), std::cend(w),
	      std::ostream_iterator<int>(std::cout, " "));
    std::cout << std::endl;
}
    
template <class IN0, class IN1, class T, class OUT> void
test_mapper()
{
    IN0	x[] = { 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15,
	       16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31};
    IN1	y[] = { 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15,
	       16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31};
    OUT	w[32];
    std::fill(std::begin(w), std::end(w), 0);

#if 1
    auto	f = make_mapper<T>(std::plus<>(),
				   std::make_tuple(
				       make_accessor(std::cbegin(x)),
				       make_accessor(std::cbegin(y))));
#else
    auto	f = make_mapper<T>([](const auto& x){ return 10*x; },
				   make_accessor(std::cbegin(x)));
#endif
    auto	o = make_mapper<OUT>([](const auto& x, auto&& y){ y = x; },
				     std::make_tuple(
					 f, make_accessor(std::begin(w))));
    for_each<size(w)/o.size>([&o]{ o(); });

    std::copy(std::cbegin(w), std::cend(w),
	      std::ostream_iterator<int>(std::cout, " "));
    std::cout << std::endl;
}

template <class IN0, class IN1, class T, class IN2, class OUT> void
test_pipe()
{
    IN0	x[] = { 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15,
	       16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31};
    IN1	y[] = { 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15,
	       16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31};
    IN2	z[] = { 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15,
	       16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31};
    OUT	w[32];
    std::fill(std::begin(w), std::end(w), 0);

    auto	f = make_mapper<T>(std::plus<>(),
				   std::make_tuple(
				       make_accessor(std::cbegin(x)),
				       make_accessor(std::cbegin(y))));
    auto	g = make_mapper<OUT>(std::multiplies<>(),
				     std::make_tuple(
					 f, make_accessor(std::cbegin(z))));
    auto	o = make_mapper<OUT>([](auto&& x, const auto& y){ x = y; },
				     std::make_tuple(
					 make_accessor(std::begin(w)), g));
    for_each<size(w)/o.size>([&o]{ o(); });

    std::copy(std::cbegin(w), std::cend(w),
	      std::ostream_iterator<int>(std::cout, " "));
    std::cout << std::endl;
}

}
}

int
main()
{
    TU::simd::test_assign<int32_t, int8_t>();

    std::cerr << "--------------" << std::endl;
    
  //TU::simd::test_mapper<int8_t, int32_t, float, int16_t>();
    TU::simd::test_mapper<int16_t, int16_t, int8_t, int32_t>();

    std::cerr << "--------------" << std::endl;
    
    TU::simd::test_pipe<int16_t, int32_t, int16_t, int8_t, int16_t>();
    
    return 0;
}
