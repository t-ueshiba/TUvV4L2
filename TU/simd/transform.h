/*!
  \file		transform.h
  \author	Toshio UESHIBA
  \brief	複数のSIMDベクトルを成分数が等しくなるように同時に変換する関数の定義
*/
#if !defined(TU_SIMD_TRANSFORM_H)
#define	TU_SIMD_TRANSFORM_H

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
	  using type = typename tuple_head<T_>::element_type;
      };
      template <class DUMMY_>
      struct vec_element<void, DUMMY_>
      {
	  using type = void;
      };
      
      using E = typename vec_element<iterator_value<OUT> >::type;
      
    //! 出力反復子に書き出すSIMDベクトルの要素型
      using O = std::conditional_t<std::is_void<E>::value, T, E>;
    //! 変換関数に入力するSIMDベクトルの要素型
      using I = std::conditional_t<std::is_void<T>::value, O, T>;

    //! tuple中のベクトルおよび反復子が指すベクトルのうちの最大要素数
      template <class ITER_>
      struct max_size_impl
      {
	  constexpr static auto
		value = max_size_impl<iterator_value<ITER_> >::value;
      };
      template <class T_>
      struct max_size_impl<vec<T_> >
      {
	  constexpr static auto	value = vec<T_>::size;
      };
      template <class VEC_>
      struct max_size_impl<std::tuple<VEC_> >
      {
	  constexpr static auto	value = max_size_impl<
					    std::decay_t<VEC_> >::value;
      };
      template <class VEC_, class... VECS_>
      struct max_size_impl<std::tuple<VEC_, VECS_...> >
      {
	private:
	  constexpr static auto	max0  = max_size_impl<
					    std::decay_t<VEC_> >::value;
	  constexpr static auto	max1  = max_size_impl<
					    std::tuple<VECS_...> >::value;
	public:
	  constexpr static auto	value = (max0 > max1 ? max0 : max1);
      };

      template <class VECS_>
      using max_size = max_size_impl<std::decay_t<VECS_> >;
      
    //! vec<I> よりも上位の複数のベクトルを vec<I> にconvert down して返す
    /*!
      - 引数として反復子が与えられた場合は，それが指す上位のベクトルを
        複数読み込み，vec<I> にconvert downして返す．
      - 引数として vec<I> と同サイズのベクトルが与えられた場合は，
        それを vec<I> に変換して返す．
     */
      struct cvtdown
      {
	//! iter が指す vec<T_> と同サイズのベクトルを vec<T_> に変換
	  template <class T_=I, class ITER_>
	  std::enable_if_t<(iterator_value<ITER_>::size == vec<T_>::size),
			   vec<T_> >
		operator ()(ITER_& iter) const
		{
		    return cvt<T_, false, MASK>(*iter++);
		}

	//! iter の位置を先頭とする vec<T_> より上位の複数のベクトルを vec<T_> に変換
	  template <class T_=I, class ITER_>
	  std::enable_if_t<(iterator_value<ITER_>::size < vec<T_>::size),
			   vec<T_> >
		operator ()(ITER_& iter) const
		{
		    using S = typename iterator_value<ITER_>::element_type;
		    using A = cvt_above_type<T_, S, MASK>;

		  // iterの位置を先頭とする複数の vec<S> を
		  // vec<T_> の直上型 vec<A> に再帰的に変換
		    const auto	x = operator ()<A>(iter);
		    const auto	y = operator ()<A>(iter);
		    return cvt<T_, MASK>(x, y);
		}

	//! vec<T> と同サイズの vec<S_> を vec<I> に変換
	  template <class S_>
	  auto	operator ()(vec<S_> x) const
		{
		    return cvt<I, false, MASK>(x);
		}

	//! vec<I> と同サイズのベクトルまたは上位の反復子を vec<I> のtupleに変換
	  template <class... VECS>
	  auto	operator ()(const std::tuple<VECS...>& x) const
		{
		    return tuple_transform(*this, x);
		}
      };
      
    //! vec<I> よりも下位のベクトルを，一段上位のベクトルにconvert up
    /*!
      サイズ N_ のベクトルを指す反復子はdereference & incrementし，
      そうでない反復子は何もしない．convert upしてサイズ N_ になる
      ベクトルは HI_ に従って上半または下半をconvert upする．
      \param N_		convert upされたベクトルのサイズ
      \param HI_	trueならば入力ベクトルの上半を，falseならば下半を変換
    */
      template <size_t N_, bool HI_>
      struct cvtup
      {
	//! 反復子が指すサイズ N_ のベクトルを読み込んで返し，反復子を進める
	  template <class ITER_>
	  std::enable_if_t<iterator_value<ITER_>::size == N_,
			   iterator_value<ITER_> >
	  	operator ()(ITER_& iter) const
		{
		    return *iter++;
		}

	//! サイズが N_ でないベクトルを指す反復子は，それへの参照をそのまま返す
	  template <class ITER_>
	  std::enable_if_t<iterator_value<ITER_>::size != N_, ITER_&>
		operator ()(ITER_& iter) const
		{
		    return iter;
		}

	//! サイズ N_ のベクトルに一段convert up
	  template <class S_>
	  auto	operator ()(vec<S_> x) const
		{
		    return cvt<cvt_upper_type<I, S_, MASK>, HI_, MASK>(x);
		}

	//! ベクトルまたは反復子のtupleをサイズ N_ のベクトルまたは反復子のtupleに変換
	  template <class... VEC_>
	  auto	operator ()(std::tuple<VEC_...>& x) const
		{
		    return tuple_transform(*this, x);
		}
      };

    private:
    // _funcの適用結果を vec<O> (のtuple)にconvert upしてstoreする
      template <class TUPLE_>
      std::enable_if_t<(vec<O>::size == tuple_head<TUPLE_>::size)>
		upResult_store(const TUPLE_& x)
		{
		    ASSIGN()(*_out, cvt<O, false, MASK>(x));
		    ++_out;
		}
      template <class TUPLE_>
      std::enable_if_t<(vec<O>::size < tuple_head<TUPLE_>::size)>
		upResult_store(const TUPLE_& x)
		{
		    using U = cvt_upper_type<
				  O, typename tuple_head<TUPLE_>::element_type,
				  MASK>;
		    
		    upResult_store(cvt<U, false, MASK>(x));
		    upResult_store(cvt<U, true,  MASK>(x));
		}
	  
    // 既に vec<O> と同位に到達している入力をさらに vec<I> (のtuple)までconvert up
    // して_funcを適用し，その結果を vec<O> (のtuple)までconvert downして返す
      template <class TUPLE_,
		std::enable_if_t<(max_size<TUPLE_>::value == vec<I>::size)>*
		= nullptr>
      auto	upArg_downResult(TUPLE_&& x)
		{
		    const auto	y = _func(cvtdown()(x));

		    using S = typename tuple_head<decltype(y)>::element_type;
		    using L = std::conditional_t<
				  (vec<cvt_lower_type<O, S, MASK> >::size >
				   vec<S>::size),
				  S, cvt_lower_type<O, S, MASK> >;
		    
		    return cvt<L, false, MASK>(y);
		}
      template <class TUPLE_,
		std::enable_if_t<(max_size<TUPLE_>::value > vec<I>::size)>*
		= nullptr>
      auto	upArg_downResult(TUPLE_&& x)
		{
		    constexpr auto	N = max_size<TUPLE_>::value;

		    const auto	y = upArg_downResult(cvtup<N/2, false>()(x));
		    const auto	z = upArg_downResult(cvtup<N/2, true >()(x));

		    using S = typename tuple_head<decltype(y)>::element_type;

		  // 戻り値のベクトルは TUPLE_ と同位
		    return cvt<cvt_lower_type<O, S, MASK>, MASK>(y, z);
		}

    // vec<O> と同位に達した入力をさらに vec<I> にconvert upして _funcを適用し，
    // その結果をさらに vec<O> にconvert upしてstoreする．
      template <class TUPLE_>
      std::enable_if_t<(max_size<TUPLE_>::value == vec<I>::size &&
			max_size<TUPLE_>::value >  vec<O>::size)>
		exec(TUPLE_&& x)
		{
		    upResult_store(_func(cvtdown()(x)));
		}

    // vec<O> と同位に達した入力をさらに vec<I> にconvert upして _funcを適用し，
    // その結果を vec<O> にconvert downしてstoreする．
      template <class TUPLE_>
      std::enable_if_t<(max_size<TUPLE_>::value == vec<O>::size)>
		exec(TUPLE_&& x)
		{
		    ASSIGN()(*_out, upArg_downResult(x));
		    ++_out;
		}

    // 入力を vec<I> または vec<O> と同位になるまでconvert upする．
      template <class TUPLE_>
      std::enable_if_t<(max_size<TUPLE_>::value > vec<I>::size &&
			max_size<TUPLE_>::value > vec<O>::size)>
		exec(TUPLE_&& x)
		{
		    constexpr auto	N = max_size<TUPLE_>::value;
		    
		    exec(cvtup<N/2, false>()(x));
		    exec(cvtup<N/2, true >()(x));
		}

    public:
		transformer(ITER_TUPLE t, head_iterator end, OUT out, FUNC func)
		    :_t(t), _end(end), _out(out), _func(func)		{}

      OUT	operator ()()
		{
		    while (std::get<0>(_t) != _end)
		    {
		      // 入力ベクトルと出力ベクトルの中で最下位のベクトルの要素数
			constexpr auto
			    N = (max_size<ITER_TUPLE>::value > vec<O>::size ?
				 max_size<ITER_TUPLE>::value : vec<O>::size);

			exec(cvtup<N, false>()(_t));
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
#endif	// !TU_SIMD_TRANSFORM_H
