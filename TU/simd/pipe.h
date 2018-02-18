/*!
  \file		cvt.h
  \author	Toshio UESHIBA
  \brief	SIMDベクトル間の型変換関数の定義
*/
#if !defined(TU_SIMD_PIPE_H)
#define	TU_SIMD_PIPE_H

#include "TU/simd/cvt.h"

namespace TU
{
namespace simd
{
//! 入力を適切に変換してから関数を適用し，結果をconvert downする．
/*!
  戻り値のSIMDベクトルは，vec<ARG>と入力のうち最下位のSIMDベクトルと同位
  \param FUNC	適用する関数
  \param ARG	FUNCの引数となるSIMDベクトルの要素型
  \param T	FUNCの結果のconvert down先のSIMDベクトルの要素型
  \param MASK
 */
template <class FUNC, class ARG, class T=ARG, bool MASK=false>
class downfunc
{
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
    downfunc(FUNC func=FUNC())	:_func(func)				{}

    template <class IN_>
    auto	operator ()(IN_&& in) const
		{
		    constexpr auto	N = std::max(vsize<IN_>::max,
						     vec<ARG>::size);
		    
		    return exec(cvtup<ARG, false, MASK, N>(
				    std::forward<IN_>(in)));
		}
    template <class... IN_>
    auto	operator ()(IN_&&... in) const
		{
		    return (*this)(std::make_tuple(in...));
		}
    
  private:
    FUNC	_func;
};

//! 入力をconvert upしてから関数を適用する．
/*!
  \param FUNC	適用する関数
  \param ARG	FUNCの引数となるSIMDベクトルの要素型
  \param MASK
*/
template <class FUNC, class ARG, bool MASK=false>
class upfunc
{
  private:
    template <class OP_, class IN_, class OUT_>
    std::enable_if_t<(vsize<IN_>::max <= vec<ARG>::size)>
		exec(OP_ op, OUT_&& out, IN_&& in) const
		{
		    op(*out, apply(_func, cvtdown<ARG, MASK>(in)));
		    ++out;
		}
    template <class OP_, class IN_, class OUT_>
    std::enable_if_t<(vsize<IN_>::max > vec<ARG>::size)>
		exec(OP_ op, OUT_&& out, IN_&& in) const
		{
		    constexpr auto	N = vsize<IN_>::max;
		    
		    exec(op, out, cvtup<ARG, false, MASK, N/2>(in));
		    exec(op, out, cvtup<ARG, true,  MASK, N/2>(in));
		}
    
  public:
    upfunc(FUNC func=FUNC())	:_func(func)				{}

    template <class OP_, class OUT_, class IN_>
    void	operator ()(OP_ op, OUT_&& out, IN_&& in) const
		{
		    constexpr auto	N = std::max(vsize<IN_>::max,
						     vec<ARG>::size);
		    
		    exec(op, out, cvtup<ARG, false, MASK, N>(in));
		}
    template <class OP_, class OUT_, class... IN_>
    void	operator ()(OP_ op, OUT_&& out, IN_&&... in) const
		{
		    (*this)(op, out, std::make_tuple(in...));
		}
    
  private:
    FUNC	_func;
};

}	// namespace simd
}	// namespace TU

#endif	// !TU_SIMD_PIPE_H
