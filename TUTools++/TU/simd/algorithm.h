/*!
  \author	Toshio UESHIBA
  \brief	SIMDベクトルを対象とした各種アルゴリズムの実装
*/
#ifndef TU_SIMD_ALGORITHM_H
#define TU_SIMD_ALGORITHM_H

#include "TU/simd/arithmetic.h"
#include "TU/simd/load_store_iterator.h"
#include "TU/algorithm.h"
#ifdef TU_DEBUG
#  include <boost/core/demangle.hpp>
#endif

namespace TU
{
namespace simd
{
template <size_t N, class ITER, class ARG, class FUNC> inline FUNC
for_each(iterator_wrapper<ITER> iter, ARG arg, FUNC func)
{
    constexpr auto	M = make_terminator<ITER>(N);
    
    return TU::for_each<M>(make_accessor(iter),
			   make_terminator(iter, arg), func);
}

template <size_t N, class ITER0, class ARG, class ITER1, class FUNC> inline FUNC
for_each(iterator_wrapper<ITER0> iter0, ARG arg,
	 iterator_wrapper<ITER1> iter1, FUNC func)
{
    constexpr auto	M = make_terminator<ITER0>(N);

    return TU::for_each<M>(make_accessor(iter0),
			   make_terminator<ITER0>(arg),
			   make_accessor(iter1), func);
}
    
template <size_t N, class ITER0, class ARG, class ITER1, class T> inline T
inner_product(iterator_wrapper<ITER0> iter0, ARG arg,
	      iterator_wrapper<ITER1> iter1, T init)
{
#ifdef TU_DEBUG
    std::cout << "(simd)inner_product<" << N << "> ["
	      << print_sizes(range<iterator_wrapper<ITER0>, N>(iter0, arg))
	      << ']' << std::endl;
#endif
    constexpr auto	M = make_terminator<ITER0>(N);
    
    return hadd(TU::inner_product<M>(make_accessor(iter0),
				     make_terminator<ITER0>(arg),
				     make_accessor(iter1), vec<T>(init)));
}
    
template <size_t N, class ITER, class ARG> inline auto
square(iterator_wrapper<ITER> iter, ARG arg)
{
    constexpr auto	M = make_terminator<ITER>(N);
    
    return hadd(TU::square<M>(make_accessor(iter),
			      make_terminator<ITER>(arg)));
}

template <class FUNC, class ITER> inline auto
make_transform_iterator1(iterator_wrapper<ITER> iter, FUNC func)
{
#ifdef TU_DEBUG
    using	boost::core::demangle;

    std::cout << "(simd)transform_iterator1:\n\t"
	      << demangle(typeid(ITER).name()) << std::endl;
#endif		  
    return make_iterator_wrapper(TU::make_transform_iterator1(
				     make_accessor(iter), func));
}

template <class FUNC, class ITER0, class ITER1> inline auto
make_transform_iterator2(iterator_wrapper<ITER0> iter0,
			 iterator_wrapper<ITER1> iter1, FUNC func)
{
#ifdef TU_DEBUG
    using	boost::core::demangle;

    std::cout << "(simd)transform_iterator2:\n\t"
	      << demangle(typeid(ITER0).name()) << "\n\t"
	      << demangle(typeid(ITER1).name()) << std::endl;
#endif  
    return make_iterator_wrapper(TU::make_transform_iterator2(
				     make_accessor(iter0),
				     make_accessor(iter1), func));
}

}	// namespace simd
}	// namespace TU
#endif	// !TU_SIMD_ALGORITHM_H
