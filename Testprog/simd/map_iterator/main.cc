/*
 *  $Id$
 */
#include "TU/simd/Array++.h"
#include <boost/core/demangle.hpp>

namespace TU
{
template <class T> inline auto
demangle()
{
    return boost::core::demangle(typeid(T).name());
}

struct sum
{
    template <class T>
    T	operator ()(T x, T y, T z)	const	{ return x + y + z; }
};
    
template <class IN, class OUT> void
test_assign()
{
    Array<IN, 0, simd::allocator<IN> >
	x({ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15,
	   16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31});
    Array<OUT, 0, simd::allocator<OUT> >	w;

    w = x;

    std::copy(cbegin(w), cend(w),
	      std::ostream_iterator<int>(std::cout, " "));
    std::cout << std::endl;
}
    
template <class IN0, class IN1, class T, class OUT> void
test_single_stage()
{
    Array<IN0, 0, simd::allocator<IN0> >
	x({ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15,
	   16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31});
    Array<IN1, 0, simd::allocator<IN1> >
	y({ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15,
	   16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31});
    Array<OUT, 0, simd::allocator<OUT> >	w;

    w = zip(x, y) | mapped<T>(std::plus<>());

    std::copy(cbegin(w), cend(w), std::ostream_iterator<int>(std::cout, " "));
    std::cout << std::endl;
}

template <class IN0, class IN1, class T, class IN2, class OUT> void
test_two_stages()
{
    Array<IN0, 0, simd::allocator<IN0> >
	x({ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15,
	   16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31});
    Array<IN1, 0, simd::allocator<IN1> >
	y({ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15,
	   16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31});
    Array<IN2, 0, simd::allocator<IN2> >
	z({ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15,
	   16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31});
    Array<OUT, 0, simd::allocator<OUT> >	w;

    w = zip(zip(x, y) | mapped<T>(std::plus<>()), z)
	| mapped<OUT>(std::multiplies<>());
    
    std::copy(cbegin(w), cend(w), std::ostream_iterator<int>(std::cout, " "));
    std::cout << std::endl;

    w += zip(x, y, z) | mapped<T>(sum());
    
    std::copy(cbegin(w), cend(w), std::ostream_iterator<int>(std::cout, " "));
    std::cout << std::endl;
}

template <class T, class IN0, class IN1> void
test_map_iterator()
{
    Array<IN0, 0, simd::allocator<IN0> >
	x({ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15,
	   16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31});
    Array<IN1, 0, simd::allocator<IN1> >
	y({ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15,
	   16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31});

    const auto	end = make_map_iterator<T>(std::plus<>(), cend(x), cend(y));
    for (auto iter = make_map_iterator<T>(std::plus<>(), cbegin(x), cbegin(y));
	 iter != end; ++iter)
	std::cout << *iter << std::endl;
}

}

int
main()
{
    TU::test_assign<int32_t, int8_t>();

    std::cerr << "--------------" << std::endl;
    
    TU::test_single_stage<int16_t, int16_t, int8_t, int32_t>();

    std::cerr << "--------------" << std::endl;
    
    TU::test_two_stages<int16_t, int32_t, float, int8_t, int16_t>();
    
    std::cerr << "--------------" << std::endl;
    
    TU::test_map_iterator<int16_t, int16_t, float>();

    return 0;
}
