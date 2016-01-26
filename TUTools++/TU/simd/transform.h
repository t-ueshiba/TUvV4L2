/*
 *  $Id$
 */
#if !defined(__TU_SIMD_TRANSFORM_H)
#define	__TU_SIMD_TRANSFORM_H

#include "TU/iterator.h"
#include "TU/simd/cvt.h"

namespace TU
{
namespace simd
{
namespace detail
{
  //! 複数の入力反復子からのデータを関数に入力し，その結果を出力反復子に書き出す
  /*!
    入力反復子毎に異なるSIMDベクトルの型をvec<T>型に統一して変換関数に入力し，
    その結果を再び変換して出力反復子に書き出す．
    \param T		関数に入力するSIMDベクトルの要素型
    \param MASK		型変換をマスクベクトルとして行うならtrue, そうでなければfalse
    \param ITER_TUPLE	複数の入力反復子をまとめたタプル
    \param OUT		関数が出力するSIMDベクトルの変換結果を書き出す出力反復子
    \param FUNC		変換関数
  */
  template <class T, bool MASK, class ITER_TUPLE, class OUT, class FUNC>
  class transformer
  {
    public:
      using head_iterator = tuple_head<ITER_TUPLE>;
      using result_type =
		typename std::conditional<
		    std::is_void<
		        typename std::iterator_traits<OUT>::value_type>::value,
		    vec<T>,
		    typename std::iterator_traits<OUT>::value_type>::type;
      
    private:
    //! 出力反復子に書き出すSIMDベクトルの要素型
      using R = typename tuple_head<result_type>::element_type;
    //! 変換関数に入力するSIMDベクトルの要素型
      using U = typename std::conditional<std::is_void<T>::value, R, T>::type;
    //! vec<S> を vec<R> に変換する過程において vec<S> の直後の変換先の要素型
      template <class S>
      using below_type =
	  typename std::conditional<
		       (vec<cvt_lower_type<R, S, MASK> >::size > vec<S>::size),
		       S, cvt_lower_type<R, S, MASK> >::type;

    //! vec<U> よりも要素数が少ない入力SIMDベクトルを vec<U> に変換
      struct generic_cvtdown
      {
	  template <class T_=U, class ITER_>
	  typename std::enable_if<
	      (vec<T_>::size == std::iterator_traits<ITER_>::value_type::size),
	      vec<T_> >::type
			operator ()(ITER_& iter) const
			{
			    const auto	x = *iter;
			    ++iter;
			    return cvt<T_, false, MASK>(x);
			}
	  template <class T_=U, class ITER_>
	  typename std::enable_if<
	      (vec<T_>::size > std::iterator_traits<ITER_>::value_type::size),
	      vec<T_> >::type
			operator ()(ITER_& iter) const
			{
			    using S = typename std::iterator_traits<ITER_>
						  ::value_type::element_type;
			    using A = cvt_above_type<T_, S, MASK>;
	  
			    const auto	x = operator ()<A>(iter);
			    const auto	y = operator ()<A>(iter);
			    return cvt<T_, MASK>(x, y);
			}
	  template <class S_>
	  typename std::enable_if<vec<U>::size == vec<S_>::size, vec<U> >::type
			operator ()(vec<S_> x) const
			{
			    return cvt<U, false, MASK>(x);
			}
      };

    //! vec<U> よりも要素数が多い入力SIMDベクトルを vec<U> に変換
      template <size_t N_, bool HI_>
      struct generic_cvtup
      {
	  template <class ITER_>
	  typename std::enable_if<
	      std::iterator_traits<ITER_>::value_type::size == N_,
	      typename std::iterator_traits<ITER_>::value_type>::type
	  		operator ()(ITER_& iter) const
			{
			    const auto	x = *iter;
			    ++iter;
			    return x;
			}
	  template <class ITER_>
	  typename std::enable_if<
	      std::iterator_traits<ITER_>::value_type::size != N_, ITER_&>::type
			operator ()(ITER_& iter) const
			{
			    return iter;
			}
	  template <class S_>
	  typename std::enable_if<vec<cvt_upper_type<U, S_, MASK> >::size == N_,
				  vec<cvt_upper_type<U, S_, MASK> > >::type
			operator ()(vec<S_> x) const
			{
			    return cvt<cvt_upper_type<U, S_, MASK>,
				       HI_, MASK>(x);
			}
      };

    //! 変換関数が出力する vec<U> 型のSIMDベクトルをより要素数が多い vec<R> 型に変換
      struct generic_cvtbelow
      {
	  template <class S_> vec<below_type<S_> >
	  		operator ()(vec<S_> x) const
			{
			    return cvtbelow(x);
			}
	  template <class S_> vec<cvt_lower_type<R, S_, MASK> >
	  		operator ()(vec<S_> x, vec<S_> y) const
			{
			    return cvtbelow(x, y);
			}
      };

    //! tuple中の反復子が指すSIMDベクトルの最大要素数
      template <class ITER_>
      struct max_size
      {
	  static constexpr size_t
		value = std::iterator_traits<ITER_>::value_type::size;
      };
      template <class ITER_>
      struct max_size<boost::tuples::cons<ITER_, boost::tuples::null_type> >
      {
	  static constexpr size_t	value = max_size<ITER_>::value;
      };
      template <class ITER_, class TAIL_>
      struct max_size<boost::tuples::cons<ITER_, TAIL_> >
      {
	  static constexpr size_t	head_max = max_size<ITER_>::value;
	  static constexpr size_t	tail_max = max_size<TAIL_>::value;
	  static constexpr size_t	value	 = (head_max > tail_max ?
						    head_max : tail_max);
      };
      template <class... ITER_>
      struct max_size<boost::tuple<ITER_...> >
	  : max_size<typename boost::tuple<ITER_...>::inherited>
      {
      };

    private:
      template <class S_> static vec<below_type<S_> >
		cvtbelow(vec<S_> x)
		{
		    return cvt<below_type<S_>, false, MASK>(x);
		}
      template <class S_> static vec<cvt_lower_type<R, S_, MASK> >
		cvtbelow(vec<S_> x, vec<S_> y)
		{
		    return cvt<cvt_lower_type<R, S_, MASK>, MASK>(x, y);
		}
      template <class HEAD, class TAIL> static auto
		cvtbelow(const boost::tuples::cons<HEAD, TAIL>& x)
		    -> decltype(boost::tuples::cons_transform(
				    generic_cvtbelow(), x))
		{
		    return boost::tuples::cons_transform(generic_cvtbelow(), x);
		}
      template <class H1, class T1, class H2, class T2> static auto
		cvtbelow(const boost::tuples::cons<H1, T1>& x,
			 const boost::tuples::cons<H2, T2>& y)
		    -> decltype(boost::tuples::cons_transform(
				    generic_cvtbelow(), x, y))
		{
		    return boost::tuples::cons_transform(generic_cvtbelow(),
							 x, y);
		}

      template <size_t N_, class TUPLE_,
		typename std::enable_if<(N_ == vec<U>::size)>::type* = nullptr>
      auto	cvtup_down(TUPLE_&& x)
		{
		    return cvtbelow(_func(boost::tuples::cons_transform(
					    generic_cvtdown(), x)));
		}
      template <size_t N_, class TUPLE_,
		typename std::enable_if<(N_ > vec<U>::size)>::type* = nullptr>
      auto	cvtup_down(TUPLE_&& x)
		{
		    const auto	y = cvtup_down<N_/2>(
					boost::tuples::cons_transform(
					    generic_cvtup<N_/2, false>(), x));
		    const auto	z = cvtup_down<N_/2>(
					boost::tuples::cons_transform(
					    generic_cvtup<N_/2, true >(), x));
		    return cvtbelow(y, z);
		}

      template <size_t N_, class TUPLE_>
      typename std::enable_if<(N_ == vec<R>::size)>::type
		cvtup(TUPLE_&& x)
		{
		    *_out = cvtup_down<N_>(x);
		    ++_out;
		}
      template <size_t N_, class TUPLE_>
      typename std::enable_if<(N_ > vec<U>::size && N_ > vec<R>::size)>::type
		cvtup(TUPLE_&& x)
		{
		    cvtup<N_/2>(boost::tuples::cons_transform(
				    generic_cvtup<N_/2, false>(), x));
		    cvtup<N_/2>(boost::tuples::cons_transform(
				    generic_cvtup<N_/2, true >(), x));
		}

    public:
		transformer(const ITER_TUPLE& t,
			    head_iterator end, OUT out, FUNC func)
		    :_t(t), _end(end), _out(out), _func(func)		{}

      OUT	operator ()()
		{
		    while (boost::get<0>(_t) != _end)
		    {
			constexpr size_t
			    N = (max_size<ITER_TUPLE>::value > vec<R>::size ?
				 max_size<ITER_TUPLE>::value : vec<R>::size);

			cvtup<N>(boost::tuples::cons_transform(
				     generic_cvtup<N, false>(), _t));
		    }
		    return _out;
		}

    private:
      ITER_TUPLE		_t;
      const head_iterator	_end;
      OUT			_out;
      FUNC			_func;
  };
}	// namespace detail

template <class T=void, bool MASK=false,
	  class FUNC, class OUT, class IN, class... IN_EXTRA> inline OUT
transform(FUNC func, OUT out, IN in, IN end, IN_EXTRA... in_extra)
{
    detail::transformer<T, MASK, boost::tuple<IN, IN_EXTRA...>, OUT, FUNC>
	trns(boost::make_tuple(in, in_extra...), end, out, func);
    return trns();
}
    
template <class T=void, bool MASK=false,
	  class ITER_TUPLE, class OUT, class FUNC> inline OUT
transform(const ITER_TUPLE& ib, const ITER_TUPLE& ie, OUT out, FUNC func)
{
    detail::transformer<T, MASK, ITER_TUPLE, OUT, FUNC>
	trns(ib, boost::get<0>(ie), out, func);
    return trns();
}
    
template <class T=void, bool MASK=false,
	  class ITER_TUPLE, class OUT, class FUNC> inline OUT
transform(const fast_zip_iterator<ITER_TUPLE>& ib,
	  const fast_zip_iterator<ITER_TUPLE>& ie, OUT out, FUNC func)
{
    return transform<T, MASK>(ib.get_iterator_tuple(),
			      ie.get_iterator_tuple(), out, func);
}
    
template <class T=void, bool MASK=false,
	  class OUT, class IN, class... IN_EXTRA> inline OUT
copy(OUT out, IN in, IN end, IN_EXTRA... in_extra)
{
    detail::transformer<T, MASK, boost::tuple<IN, IN_EXTRA...>, OUT, identity>
	trns(boost::make_tuple(in, in_extra...), end, out, identity());
    return trns();
}
    
template <class T=void, bool MASK=false,
	  class ITER_TUPLE, class OUT> inline OUT
copy(const ITER_TUPLE& ib, const ITER_TUPLE& ie, OUT out)
{
    detail::transformer<T, MASK, ITER_TUPLE, OUT, identity>
	trns(ib, boost::get<0>(ie), out, identity());
    return trns();
}
    
template <class T=void, bool MASK=false,
	  class ITER_TUPLE, class OUT> inline OUT
copy(const fast_zip_iterator<ITER_TUPLE>& ib,
     const fast_zip_iterator<ITER_TUPLE>& ie, OUT out)
{
    return copy<T, MASK>(ib.get_iterator_tuple(), ie.get_iterator_tuple(), out);
}
    
}	// namespace simd
}	// namespace TU
#endif	// !__TU_SIMD_TRANSFORM_H
