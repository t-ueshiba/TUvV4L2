/*
 *  $Id$
 */
#if !defined(__TU_SIMD_TRANSFORM_H)
#define	__TU_SIMD_TRANSFORM_H

#include "TU/iterator.h"
#include "TU/functional.h"
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
    \param ASSIGN	変換結果を出力反復子に書き出す際に用いる代入演算子
    \param MASK		型変換をマスクベクトルとして行うならtrue, そうでなければfalse
    \param ITER_TUPLE	複数の入力反復子をまとめたタプル
    \param OUT		関数が出力するSIMDベクトルの変換結果を書き出す出力反復子
    \param FUNC		変換関数
  */
  template <class T, class ASSIGN, bool MASK,
	    class ITER_TUPLE, class OUT, class FUNC>
  class transformer
  {
    public:
      using head_iterator = tuple_head<ITER_TUPLE>;
      
    private:
      template <class T_, class=void>
      struct vec_element
      {
	  using type = typename T_::element_type;
      };
      template <class DUMMY_>
      struct vec_element<void, DUMMY_>
      {
	  using type = void;
      };
      
      using E = typename vec_element<tuple_head<iterator_value<OUT> > >::type;
      
    //! 出力反復子に書き出すSIMDベクトルの要素型
      using O = std::conditional_t<std::is_void<E>::value, T, E>;
    //! 変換関数に入力するSIMDベクトルの要素型
      using I = std::conditional_t<std::is_void<T>::value, O, T>;

    //! vec<I> よりも要素数が少ない入力SIMDベクトルを vec<I> に変換
      struct generic_downArg
      {
	  template <class T_=I, class ITER_>
	  std::enable_if_t<(vec<T_>::size == iterator_value<ITER_>::size),
			   vec<T_> >
			operator ()(ITER_& iter) const
			{
			    const auto	x = *iter;
			    ++iter;
			    return cvt<T_, false, MASK>(x);
			}
	  template <class T_=I, class ITER_>
	  std::enable_if_t<(vec<T_>::size > iterator_value<ITER_>::size),
			   vec<T_> >
			operator ()(ITER_& iter) const
			{
			    using S = typename
					  iterator_value<ITER_>::element_type;
			    using A = cvt_above_type<T_, S, MASK>;
	  
			    const auto	x = operator ()<A>(iter);
			    const auto	y = operator ()<A>(iter);
			    return cvt<T_, MASK>(x, y);
			}
	  template <class S_>
	  std::enable_if_t<vec<I>::size == vec<S_>::size, vec<I> >
			operator ()(vec<S_> x) const
			{
			    return cvt<I, false, MASK>(x);
			}
      };

    //! vec<I> よりも要素数が多い入力SIMDベクトルを vec<I> に変換
      template <size_t N_, bool HI_>
      struct generic_upArg
      {
	  template <class ITER_>
	  std::enable_if_t<iterator_value<ITER_>::size == N_,
			   iterator_value<ITER_> >
	  		operator ()(ITER_& iter) const
			{
			    const auto	x = *iter;
			    ++iter;
			    return x;
			}
	  template <class ITER_>
	  std::enable_if_t<iterator_value<ITER_>::size != N_, ITER_&>
			operator ()(ITER_& iter) const
			{
			    return iter;
			}
	  template <class S_>
	  std::enable_if_t<vec<cvt_upper_type<I, S_, MASK> >::size == N_,
			   vec<cvt_upper_type<I, S_, MASK> > >
			operator ()(vec<S_> x) const
			{
			    return cvt<cvt_upper_type<I, S_, MASK>,
				       HI_, MASK>(x);
			}
      };

    //! tuple中の反復子が指すSIMDベクトルの最大要素数
      template <class ITER_>
      struct max_size
      {
	  static constexpr size_t	value = iterator_value<ITER_>::size;
      };
      template <class ITER_>
      struct max_size<std::tuple<ITER_> >
      {
	  static constexpr size_t	value = max_size<ITER_>::value;
      };
      template <class ITER_, class... TAIL_>
      struct max_size<std::tuple<ITER_, TAIL_...> >
      {
	  static constexpr size_t	head_max = max_size<ITER_>::value;
	  static constexpr size_t	tail_max = max_size<
						   std::tuple<TAIL_...> >::value;
	  static constexpr size_t	value	 = (head_max > tail_max ?
						    head_max : tail_max);
      };

    private:
    // 変換結果をconvert up
      template <class TUPLE_>
      std::enable_if_t<(vec<O>::size == tuple_head<TUPLE_>::size)>
		upResult(const TUPLE_& x)
		{
		    ASSIGN()(*_out, cvt<O, false, MASK>(x));
		    ++_out;
		}
      template <class TUPLE_>
      std::enable_if_t<(vec<O>::size < tuple_head<TUPLE_>::size)>
		upResult(const TUPLE_& x)
		{
		    using U = cvt_upper_type<
				  O, typename tuple_head<TUPLE_>::element_type,
				  MASK>;
		    
		    upResult(cvt<U, false, MASK>(x));
		    upResult(cvt<U, true,  MASK>(x));
		}
	  
    // 変換結果をconvert down
      template <class TUPLE_> static auto
		downResult(const TUPLE_& x)
		{
		    using S = typename tuple_head<TUPLE_>::element_type;
		    using L = std::conditional_t<
				  (vec<cvt_lower_type<O, S, MASK> >::size >
				   vec<S>::size),
				  S, cvt_lower_type<O, S, MASK> >;
		    
		    return cvt<L, false, MASK>(x);
		}
      template <class TUPLE_> static auto
		downResult(const TUPLE_&x, const TUPLE_& y)
		{
		    using S = typename tuple_head<TUPLE_>::element_type;

		    return cvt<cvt_lower_type<O, S, MASK>, MASK>(x, y);
		}

    // 入力をconvert upし，変換結果をconvert down
      template <size_t N_, class TUPLE_,
		std::enable_if_t<(N_ == vec<I>::size)>* = nullptr>
      auto	upArg_downResult(TUPLE_&& x)
		{
		    return downResult(_func(tuple_transform(generic_downArg(),
							    x)));
		}
      template <size_t N_, class TUPLE_,
		std::enable_if_t<(N_ > vec<I>::size)>* = nullptr>
      auto	upArg_downResult(TUPLE_&& x)
		{
		    const auto	y = upArg_downResult<N_/2>(
					tuple_transform(
					    generic_upArg<N_/2, false>(), x));
		    const auto	z = upArg_downResult<N_/2>(
					tuple_transform(
					    generic_upArg<N_/2, true >(), x));
		    return downResult(y, z);
		}

    // 入力をconvert up
      template <size_t N_, class TUPLE_>
      std::enable_if_t<(N_ == vec<I>::size && N_ > vec<O>::size)>
		upArg(TUPLE_&& x)
		{
		    upResult(_func(tuple_transform(generic_downArg(), x)));
		}
      template <size_t N_, class TUPLE_>
      std::enable_if_t<(N_ == vec<O>::size)>
		upArg(TUPLE_&& x)
		{
		    ASSIGN()(*_out, upArg_downResult<N_>(x));
		    ++_out;
		}
      template <size_t N_, class TUPLE_>
      std::enable_if_t<(N_ > vec<I>::size && N_ > vec<O>::size)>
		upArg(TUPLE_&& x)
		{
		    upArg<N_/2>(tuple_transform(generic_upArg<N_/2, false>(),
						x));
		    upArg<N_/2>(tuple_transform(generic_upArg<N_/2, true >(),
						x));
		}

    public:
		transformer(ITER_TUPLE t, head_iterator end, OUT out, FUNC func)
		    :_t(t), _end(end), _out(out), _func(func)		{}

      OUT	operator ()()
		{
		    while (std::get<0>(_t) != _end)
		    {
			constexpr size_t
			    N = (max_size<ITER_TUPLE>::value > vec<O>::size ?
				 max_size<ITER_TUPLE>::value : vec<O>::size);

			upArg<N>(tuple_transform(generic_upArg<N, false>(),
						 _t));
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

template <class T=void, class ASSIGN=assign, bool MASK=false,
	  class FUNC, class OUT, class IN, class... IN_EXTRA> inline OUT
transform(FUNC func, OUT out, IN in, IN end, IN_EXTRA... in_extra)
{
    detail::transformer<T, ASSIGN, MASK,
			std::tuple<IN, IN_EXTRA...>, OUT, FUNC>
	trns(std::make_tuple(in, in_extra...), end, out, func);
    return trns();
}
    
template <class T=void, class ASSIGN=assign, bool MASK=false,
	  class ITER_TUPLE, class OUT, class FUNC> inline OUT
transform(ITER_TUPLE ib, ITER_TUPLE ie, OUT out, FUNC func)
{
    detail::transformer<T, ASSIGN, MASK, ITER_TUPLE, OUT, FUNC>
	trns(ib, std::get<0>(ie), out, func);
    return trns();
}
    
template <class T=void, class ASSIGN=assign, bool MASK=false,
	  class ITER_TUPLE, class OUT, class FUNC> inline OUT
transform(zip_iterator<ITER_TUPLE> ib,
	  zip_iterator<ITER_TUPLE> ie, OUT out, FUNC func)
{
    return transform<T, ASSIGN, MASK>(ib.get_iterator_tuple(),
				      ie.get_iterator_tuple(), out, func);
}
    
template <class T=void, class ASSIGN=assign, bool MASK=false,
	  class OUT, class IN, class... IN_EXTRA> inline OUT
copy(OUT out, IN in, IN end, IN_EXTRA... in_extra)
{
    detail::transformer<T, ASSIGN, MASK,
			std::tuple<IN, IN_EXTRA...>, OUT, identity>
	trns(std::make_tuple(in, in_extra...), end, out, identity());
    return trns();
}
    
template <class T=void, class ASSIGN=assign, bool MASK=false,
	  class ITER_TUPLE, class OUT> inline OUT
copy(ITER_TUPLE ib, ITER_TUPLE ie, OUT out)
{
    detail::transformer<T, ASSIGN, MASK, ITER_TUPLE, OUT, identity>
	trns(ib, std::get<0>(ie), out, identity());
    return trns();
}
    
template <class T=void, class ASSIGN=assign, bool MASK=false,
	  class ITER_TUPLE, class OUT> inline OUT
copy(const zip_iterator<ITER_TUPLE>& ib,
     const zip_iterator<ITER_TUPLE>& ie, OUT out)
{
    return copy<T, ASSIGN, MASK>(ib.get_iterator_tuple(),
				 ie.get_iterator_tuple(), out);
}
    
}	// namespace simd
}	// namespace TU
#endif	// !__TU_SIMD_TRANSFORM_H
