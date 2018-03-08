/*!
  \file		mapper.h
  \author	Toshio UESHIBA
  \brief
*/
#if !defined(TU_SIMD_MAPPER_H)
#define	TU_SIMD_MAPPER_H

#include "TU/simd/cvt.h"
#include "TU/simd/load_store_iterator.h"

namespace TU
{
namespace simd
{
template <class T, bool MASK=false, class FUNC, class IN>
auto	make_mapper(FUNC&& func, IN&& in)				;

template <class T, bool MASK, class FUNC, class IN>
class mapper
{
  private:
    template <class ITER_, class=void>
    struct vsize_impl	// ITER_  が detail::store_proxy の場合にも有効
    {			// にするため，iterator_value<ITER_> は使わない
	constexpr static auto
		max = vsize_impl<typename ITER_::value_type>::max;
    };
    template <class T_>
    struct vsize_impl<vec<T_> >
    {
	constexpr static auto	max = vec<T_>::size;
    };
    template <class DUMMY_>
    struct vsize_impl<std::tuple<>, DUMMY_>
    {
	constexpr static size_t	max = 0;
    };
    template <class VEC_, class... VECS_>
    struct vsize_impl<std::tuple<VEC_, VECS_...> >
    {
	constexpr static auto
		max = std::max(vsize_impl<std::decay_t<VEC_>   >::max,
			       vsize_impl<std::tuple<VECS_...> >::max);
    };
    template <class T_, bool MASK_, class FUNC_, class IN_>
    struct vsize_impl<mapper<T_, MASK_, FUNC_, IN_> >
    {
	constexpr static auto
		max = std::max(vsize_impl<std::decay_t<IN_> >::max,
			       vec<T_>::size);
    };

  //! tuple中のベクトルおよび反復子が指すベクトルのうちの最大/最小要素数
    template <class VECS_>
    using vsize = vsize_impl<std::decay_t<VECS_> >;

  //! 入力の型と演算の型がそれぞれ IN_, T_ である mapper から T_ 型の出力を
    template <class IN_, class S_>
    struct result
    {
	constexpr static auto
		size = std::max(vec<S_>::size,
				std::min(vsize<IN_>::max, vec<T>::size));
    };

  private:
  /************************ convert up ************************/
  // SIMDベクトルを convert up
    template <bool HI_, size_t N_, class S_>
    static std::enable_if_t<vec<S_>::size == N_, vec<S_> >
		cvtup(vec<S_> x)	// 与えられた vec のサイズが
		{			// 要求値 _N に等しければ
		    return x;		// vec をそのまま返す
		}
    template <bool HI_, size_t N_, class S_,
	      std::enable_if_t<vec<S_>::size != N_>* = nullptr>
    static auto	cvtup(vec<S_> x)	// 与えられた vec のサイズが
		{			// 要求値 _N でなければ直上の vec を返す
		    return cvt<cvt_upper_type<T, S_, MASK>, HI_, MASK>(x);
		}

  // 反復子を convert up
    template <bool HI_, size_t N_, class ITER_,
	      std::enable_if_t<iterator_value<ITER_>::size == N_>* = nullptr>
    static auto	cvtup(ITER_& iter)	// 反復子が指す vec のサイズが
		{			// 要求値 _N に等しければ
		    return *iter++;	// vec を返し反復子をインクリメント
		}
    template <bool HI_, size_t N_, class ITER_>
    static std::enable_if_t<iterator_value<ITER_>::size != N_, ITER_&>
		cvtup(ITER_& iter)	// 反復子が指す vec のサイズが
		{			// 要求値 _N でなければ
		    return iter;	// 反復子そのものへの参照を返す
		}

  // mapper を convert up
    template <bool HI_, size_t N_,
	      class S_, bool MASK_, class FUNC_, class IN_>
    static auto	cvtup(mapper<S_, MASK_, FUNC_, IN_>& m)
		{
		    return m.template cvtup<HI_, N_>();
		}

  // SIMDベクトル|反復子|mapper の tuple を convert up
    template <bool HI_, size_t N_, class... VEC_>
    static auto	cvtup(std::tuple<VEC_...>& t)
		{
		    return tuple_transform(
				[](auto&& x) -> decltype(auto)
				{ return mapper::cvtup<HI_, N_>(x); },
				t);
		}

  /************************ convert down ************************/
  // SIMDベクトルを convert down
    template <size_t=vec<T>::size, class T_=T, class S_>
    static auto	cvtdown(vec<S_> x)
		{
		    using L = cvt_lower_type<T_, S_, MASK>;
		    using B = std::conditional_t<vec<L>::size == vec<S_>::size,
						 L, S_>;

		    return cvt<B, false, MASK>(x);
		}
    template <class T_, class S_>
    static auto	cvtdown(vec<S_> x, vec<S_> y)
		{
		    return cvt<cvt_lower_type<T_, S_, MASK>, MASK>(x, y);
		}

  // 反復子を convert down
    template <size_t N_, class ITER_,
	      std::enable_if_t<iterator_value<ITER_>::size == N_>* = nullptr>
    static auto	cvtdown(ITER_& iter)
		{
		    return cvtdown(*iter++);
		}
    template <size_t N_, class ITER_,
	      std::enable_if_t<(iterator_value<ITER_>::size < N_)>* = nullptr>
    static auto	cvtdown(ITER_& iter)
		{
		    const auto	x = cvtdown<N_/2>(iter);
		    const auto	y = cvtdown<N_/2>(iter);
    
		    return cvtdown<T>(x, y);
		}

  // mapperを convert down
    template <size_t N_, class S_, bool MASK_, class FUNC_, class IN_,
	      std::enable_if_t<result<IN_, S_>::size == N_>* = nullptr>
    static auto	cvtdown(mapper<S_, MASK_, FUNC_, IN_>& m)
		{
		    return m.template eval<T>();
		}
    template <size_t N_, class S_, bool MASK_, class FUNC_, class IN_,
	      std::enable_if_t<(result<IN_, S_>::size < N_)>* = nullptr>
    static auto	cvtdown(mapper<S_, MASK_, FUNC_, IN_>& m)
		{
		    const auto	x = cvtdown<N_/2>(m);
		    const auto	y = cvtdown<N_/2>(m);

		    return cvtdown<T>(x, y);
		}

  // detail::store_proxy を convert down
    template <size_t, bool ALIGNED_>
    static auto&
		cvtdown(detail::store_proxy<T, ALIGNED_>& x)
		{
		    return x;
		}

  // SIMDベクトル|反復子|mapper|detail::store_proxy の tuple を convert down
    template <class... VEC_>
    static auto	cvtdown(std::tuple<VEC_...>& t)
		{
		    return tuple_transform(
				[](auto&& x) -> decltype(auto)
				{ return mapper::cvtdown<vec<T>::size>(x); },
				t);
		}
  
  // IN_ をより上位の型へ convert up して FUNC を適用
    template <class IN_>
    std::enable_if_t<(vsize<IN_>::max <= vec<T>::size)>
		exec(IN_&& in) const
		{
		    apply(_func, cvtdown(in));
		}
    template <class IN_>
    std::enable_if_t<(vsize<IN_>::max > vec<T>::size)>
		exec(IN_&& in) const
		{
		    constexpr auto	N = vsize<IN_>::max;
		    
		    exec(cvtup<false, N/2>(in));
		    exec(cvtup<true,  N/2>(in));
		}

  // IN_ をより上位の型へ convert up して FUNC を適用し，
  // 結果を IN_ のレベル に convert down して返す
    template <class T_, class IN_,
	      std::enable_if_t<(vsize<IN_>::max <= vec<T>::size)>* = nullptr>
    auto	eval(IN_&& in) const
		{
		  // 戻り値は T と同位
		    return cvtdown<vec<T>::size, T_>(apply(_func, cvtdown(in)));
		}
    template <class T_, class IN_,
	      std::enable_if_t<(vsize<IN_>::max > vec<T>::size)>* = nullptr>
    auto	eval(IN_&& in) const
		{
		    constexpr auto	N = vsize<IN_>::max;

		    const auto	x = eval<T_>(cvtup<false, N/2>(in));
		    const auto	y = eval<T_>(cvtup<true,  N/2>(in));
		  // 戻り値は IN_ の最低位要素と同位
		    return cvtdown<T_>(x, y);
		}

  public:
		mapper(FUNC&& func, IN&& in)
		    :_in(in), _func(func)		{}

    void	operator ()()
		{
		    exec(cvtup<false, size>(_in));
		}

    template <class T_>
    auto	eval()
		{
		    return eval<T_>(cvtup<false, size>(_in));
		}

    template <bool HI_, size_t N_,
	      std::enable_if_t<vec<T>::size == N_>* = nullptr>
    auto	cvtup()
		{
		    return eval<T>(cvtup<HI_, N_>(_in));
		}
    template <bool HI_, size_t N_,
	      std::enable_if_t<vec<T>::size != N_>* = nullptr>
    auto	cvtup()
		{
		    return make_mapper<T, MASK>(_func, cvtup<HI_, N_>(_in));
		}
    
  private:
    IN		_in;	// 反復子または反復子の tuple
    FUNC	_func;

  public:
    constexpr static auto	size = vsize<mapper>::max;
};

template <class T, bool MASK, class FUNC, class IN> inline auto
make_mapper(FUNC&& func, IN&& in)
{
    return mapper<T, MASK, FUNC, IN>(std::forward<FUNC>(func),
				     std::forward<IN>(in));
}

}	// namespace simd
}	// namespace TU
#endif	// !TU_SIMD_MAPPER_H
