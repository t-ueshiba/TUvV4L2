/*!
  \file		map_iterator.h
  \author	Toshio UESHIBA
  \brief	SIMDベクトル間の型変換関数の定義
*/
#if !defined(TU_SIMD_MAP_ITERATOR_H)
#define	TU_SIMD_MAP_ITERATOR_H

#include "TU/simd/cvt.h"

namespace TU
{
namespace simd
{
/************************************************************************
*  class map_iterator<S, T, MASK, FUNC, ITER...>			*
************************************************************************/
//! 入力を適切に変換してから関数を適用し，結果をconvert downする．
/*!
  戻り値のSIMDベクトルは，vec<S>と入力のうち最下位のSIMDベクトルと同位
  \param S	FUNCの引数となるSIMDベクトルの要素型
  \param T	FUNCの結果のconvert down先のSIMDベクトルの要素型
  \param MASK
  \param FUNC	適用する関数
 */
template <class S, class T, bool MASK, class FUNC, class... ITER>
class map_iterator
    : public boost::iterator_facade<
		map_iterator<S, T, MASK, FUNC, ITER...>,
		decltype(
		    apply(std::declval<FUNC>(),
			  std::declval<
			  replace_element<std::tuple<ITER...>, vec<S> > >())),
		boost::single_pass_traversal_tag,
		decltype(
		    apply(std::declval<FUNC>(),
			  std::declval<
			  replace_element<std::tuple<ITER...>, vec<S> > >()))>
{
  private:
    using iter_tuple	= std::tuple<ITER...>;
    using ref		= decltype(
			      apply(std::declval<FUNC>(),
				    std::declval<
				    replace_element<iter_tuple, vec<S> > >()));
    using super		= boost::iterator_facade<
				map_iterator,
				ref, 
				boost::single_pass_traversal_tag,
				ref>;
    friend	class boost::iterator_core_access;
    
  public:
    using	typename super::reference;
    
  private:
    template <class IN_,
	      std::enable_if_t<(vsize<IN_>::max <= vec<S>::size)>* = nullptr>
    auto	exec(IN_&& in) const
		{
		  // 戻り値のベクトルは S と同位
		    return cvtdown<T, MASK>(apply(_func,
						  cvtdown<S, MASK>(in)));
		}
    template <class IN_,
	      std::enable_if_t<(vsize<IN_>::max > vec<S>::size)>* = nullptr>
    auto	exec(IN_&& in) const
		{
		    constexpr auto	N = vsize<IN_>::max;

		    const auto	x = exec(cvtup<S, false, MASK, N/2>(in));
		    const auto	y = exec(cvtup<S, true,  MASK, N/2>(in));

		  // 戻り値のベクトルは IN_ と同位
		    return cvtdown<T, MASK>(x, y);
		}

  public:
		map_iterator(const FUNC& func, const ITER&... iter)
		    :_iter_tuple(iter...), _func(func)	{}

    const auto&	get_iterator_tuple()		const	{ return _iter_tuple; }

  private:
    reference	dereference() const
		{
		    constexpr auto	N = std::max(vsize<iter_tuple>::max,
						     vec<S>::size);
		    
		    return exec(cvtup<S, false, MASK, N>(_iter_tuple));
		}
    void	increment()
		{
		}
    template <class... ITER_>
    bool	equal(const map_iterator<S, T, MASK,
					 FUNC, ITER_...>& iter) const
		{
		    return std::get<0>(_iter_tuple) ==
			   std::get<0>(iter.get_iterator_tuple());
		}
    
  private:
    mutable iter_tuple	_iter_tuple;
    FUNC		_func;
};

template <class S, class T=S, bool MASK=false, class FUNC, class... ITER>
inline map_iterator<S, T, MASK, FUNC, ITER...>
make_map_iterator(const FUNC& func, const ITER&... iter)
{
    return {func, iter...};
}

}	// namespace simd
}	// namespace TU
#endif	// !TU_SIMD_MAP_ITERATOR_H
