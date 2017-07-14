/*!
  \author	Toshio UESHIBA
  \brief	SIMDベクトルを対象とした各種アルゴリズムの実装
*/
#ifndef TU_SIMD_ALGORITHM_H
#define TU_SIMD_ALGORITHM_H

#include <TU/simd/simd.h>

namespace TU
{
template <size_t N, class T, class ARG, class FUNC> inline FUNC
for_each(simd::ptr<T> p, ARG arg, FUNC func)
{
    constexpr size_t	vsize = simd::vec<std::remove_cv_t<T> >::size;
    constexpr size_t	M = (N > 0 ? (N - 1)/vsize + 1 : 0);
    
    return detail::for_each(simd::make_accessor(p), simd::end<T>(arg), func,
			    std::integral_constant<size_t, M>());
}

template <size_t N, class S, class T, class ARG, class FUNC>
inline std::enable_if_t<simd::vec<std::remove_cv_t<S> >::size ==
			simd::vec<std::remove_cv_t<T> >::size, FUNC>
for_each(simd::ptr<S> p, ARG arg, simd::ptr<T> q, FUNC func)
{
    constexpr size_t	vsize = simd::vec<std::remove_cv_t<S> >::size;
    constexpr size_t	M = (N > 0 ? (N - 1)/vsize + 1 : 0);
    
    return detail::for_each(simd::make_accessor(p), simd::end<S>(arg),
			    simd::make_accessor(q), func,
			    std::integral_constant<size_t, M>());
}
    
template <size_t N, class T, class ARG, class ITER, class FUNC>
inline std::enable_if_t<simd::is_vec<iterator_value<ITER> >::value, FUNC>
for_each(simd::ptr<T> p, ARG arg, ITER iter, FUNC func)
{
    constexpr size_t	vsize = simd::vec<std::remove_cv_t<T> >::size;
    constexpr size_t	M = (N > 0 ? (N - 1)/vsize + 1 : 0);
    
    return detail::for_each(simd::make_accessor(p), simd::end<T>(arg), iter,
			    func, std::integral_constant<size_t, M>());
}
    
template <size_t N, class ITER, class ARG, class T, class FUNC>
inline std::enable_if_t<simd::is_vec<iterator_value<ITER> >::value, FUNC>
for_each(ITER iter, ARG arg, simd::ptr<T> p, FUNC func)
{
    constexpr size_t	vsize = simd::vec<std::remove_cv_t<T> >::size;
    constexpr size_t	M = (N > 0 ? (N - 1)/vsize + 1 : 0);
    
    return detail::for_each(iter, simd::end<T>(arg), simd::make_accessor(p),
			    func, std::integral_constant<size_t, M>());
}
    
template <size_t N, class T, class ARG> inline T
inner_product(simd::ptr<const T> p, ARG arg, simd::ptr<const T> q, T init)
{
    constexpr size_t	vsize = simd::vec<T>::size;
    constexpr size_t	M = (N > 0 ? (N - 1)/vsize + 1 : 0);
    
    return detail::inner_product(simd::make_accessor(p),
				 simd::end<const T>(arg),
				 simd::make_accessor(q), simd::vec<T>(init),
				 std::integral_constant<size_t, M>());
}
    
template <size_t N, class T, class ARG> inline T
square(simd::ptr<const T> p, ARG arg)
{
    constexpr size_t	vsize = simd::vec<T>::size;
    constexpr size_t	M = (N > 0 ? (N - 1)/vsize + 1 : 0);
    
    return detail::square(simd::make_accessor(p), simd::end<const T>(arg),
			  std::integral_constant<size_t, M>());
}

template <class FUNC, class T> inline auto
make_transform_iterator1(simd::ptr<const T> p, FUNC func)
{
    return make_transform_iterator1(simd::make_accessor(p), func);
}

template <class FUNC, class S, class T> inline auto
make_transform_iterator2(simd::ptr<const S> p, simd::ptr<const T> q, FUNC func)
{
    return make_transform_iterator2(simd::make_accessor(p),
				    simd::make_accessor(q), func);
}

template <class FUNC, class T, class ITER,
	  std::enable_if_t<std::is_convertible<iterator_value<ITER>,
					       simd::vec<T> >::value>*
	  = nullptr> inline auto
make_transform_iterator2(simd::ptr<const T> p, ITER iter, FUNC func)
{
    return make_transform_iterator2(simd::make_accessor(p), iter, func);
}

template <class FUNC, class ITER, class T,
	  std::enable_if_t<std::is_convertible<iterator_value<ITER>,
					       simd::vec<T> >::value>*
	  = nullptr> inline auto
make_transform_iterator2(ITER iter, simd::ptr<const T> p, FUNC func)
{
    return make_transform_iterator2(iter, simd::make_accessor(p), func);
}

}	// namespace TU
#endif	// !TU_SIMD_ALGORITHM_H
