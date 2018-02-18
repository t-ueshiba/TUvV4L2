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
*  class map_iterator<ARG, T, MASK, FUNC, ITER...>			*
************************************************************************/
//! 入力を適切に変換してから関数を適用し，結果をconvert downする．
/*!
  戻り値のSIMDベクトルは，vec<ARG>と入力のうち最下位のSIMDベクトルと同位
  \param ARG	FUNCの引数となるSIMDベクトルの要素型
  \param T	FUNCの結果のconvert down先のSIMDベクトルの要素型
  \param MASK
  \param FUNC	適用する関数
 */
template <class ARG, class T, bool MASK, class FUNC, class... ITER>
class map_iterator
{
  private:
    using	iter_tuple = std::tuple<ITER...>;
    
  private:
    template <class IN_,
	      std::enable_if_t<(vsize<IN_>::max <= vec<ARG>::size)>* = nullptr>
    auto	exec(IN_&& in) const
		{
		  // 戻り値のベクトルは ARG と同位
		    return cvtdown<T, MASK>(apply(_func,
						  cvtdown<ARG, MASK>(in)));
		}
    template <class IN_,
	      std::enable_if_t<(vsize<IN_>::max > vec<ARG>::size)>* = nullptr>
    auto	exec(IN_&& in) const
		{
		    constexpr auto	N = vsize<IN_>::max;

		    const auto	x = exec(cvtup<ARG, false, MASK, N/2>(in));
		    const auto	y = exec(cvtup<ARG, true,  MASK, N/2>(in));

		  // 戻り値のベクトルは IN_ と同位
		    return cvtdown<T, MASK>(x, y);
		}

  public:
		map_iterator(const FUNC& func, const ITER&... iter)
		    :_iter_tuple(iter...), _func(func)	{}

    const auto&	get_iterator_tuple()		const	{ return _iter_tuple; }
	
    auto	operator *() const
		{
		    constexpr auto	N = std::max(vsize<iter_tuple>::max,
						     vec<ARG>::size);
		    
		    return exec(cvtup<ARG, false, MASK, N>(
				    std::forward<iter_tuple>(_iter_tuple)));
		}
    auto&	operator ++()
		{
		    return *this;
		}
    template <class... ITER_>
    bool	operator ==(const map_iterator<ARG, T, MASK, FUNC, ITER_...>&
			    iter) const
		{
		    return std::get<0>(_iter_tuple) ==
			   std::get<0>(iter.get_iterator_tuple());
		}
    template <class... ITER_>
    bool	operator !=(const map_iterator<ARG, T, MASK, FUNC, ITER_...>&
			    iter) const
		{
		    return !(*this == iter);
		}
    
  private:
    mutable iter_tuple	_iter_tuple;
    FUNC		_func;
};

template <class ARG, class T=ARG, bool MASK=false, class FUNC, class... ITER>
inline map_iterator<ARG, T, MASK, FUNC, ITER...>
make_map_iterator(const FUNC& func, const ITER&... iter)
{
    return {func, iter...};
}

}	// namespace simd
}	// namespace TU
#endif	// !TU_SIMD_MAP_ITERATOR_H
