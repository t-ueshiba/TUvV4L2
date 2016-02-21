/*
 *  平成14-24年（独）産業技術総合研究所 著作権所有
 *  
 *  創作者：植芝俊夫
 *
 *  本プログラムは（独）産業技術総合研究所の職員である植芝俊夫が創作し，
 *  （独）産業技術総合研究所が著作権を所有する秘密情報です．著作権所有
 *  者による許可なしに本プログラムを使用，複製，改変，第三者へ開示する
 *  等の行為を禁止します．
 *  
 *  このプログラムによって生じるいかなる損害に対しても，著作権所有者お
 *  よび創作者は責任を負いません。
 *
 *  Copyright 2002-2012.
 *  National Institute of Advanced Industrial Science and Technology (AIST)
 *
 *  Creator: Toshio UESHIBA
 *
 *  [AIST Confidential and all rights reserved.]
 *  This program is confidential. Any using, copying, changing or
 *  giving any information concerning with this program to others
 *  without permission by the copyright holder are strictly prohibited.
 *
 *  [No Warranty.]
 *  The copyright holder or the creator are not responsible for any
 *  damages caused by using this program.
 *  
 *  $Id$
 */
/*!
  \file		functional.h
  \brief	関数オブジェクトの定義と実装
*/
#ifndef __TU_FUNCTIONAL_H
#define __TU_FUNCTIONAL_H

#include <cstddef>				// for including size_t
#include <cmath>				// std::sqrt()
#include <functional>				// std::bind()
#include <type_traits>				// std::integral_constant
#include <numeric>				// std::accumulate()
#include <cassert>
#include <boost/iterator/iterator_adaptor.hpp>

namespace std
{
#if __cplusplus <= 201103L
/************************************************************************
*  struct index_sequence<size_t...>					*
************************************************************************/
template <size_t...> struct index_sequence			{};

namespace detail
{
  template <size_t N, size_t... IDX>
  struct make_index_sequence : make_index_sequence<N - 1, N - 1, IDX...>
  {
  };
  template <size_t... IDX>
  struct make_index_sequence<0, IDX...>
  {
      typedef index_sequence<IDX...>	type;
  };
}
    
template <size_t N>
using make_index_sequence = typename detail::make_index_sequence<N>::type;
#endif
}

namespace TU
{
/************************************************************************
*  struct is_convertible<T, C<ARGS...> >				*
************************************************************************/
namespace detail
{
  template <template <class...> class C>
  struct is_convertible
  {
      template <class... ARGS>
      static std::true_type	check(C<ARGS...>)			;
      static std::false_type	check(...)				;
  };
}	// namespace detail

template <class T, template <class...> class C>
struct is_convertible
    : decltype(detail::is_convertible<C>::check(std::declval<T>()))	{};
	
/************************************************************************
*  struct generic_function<FUNC>					*
************************************************************************/
//! テンプレート引数を1つ持つ関数オブジェクトをgenericにするアダプタ
template <template <class> class FUNC>
struct generic_function
{
    template <class T_> auto
    operator ()(T_&& x) const -> decltype(FUNC<T_>()(std::forward<T_>(x)))
    {
	return FUNC<T_>()(std::forward<T_>(x));
    }

    template <class T_> auto
    operator ()(const T_& x, const T_& y) const -> decltype(FUNC<T_>()(x, y))
    {
	return FUNC<T_>()(x, y);
    }
};

/************************************************************************
*  struct generic_binary_function<FUNC>					*
************************************************************************/
//! テンプレート引数を2つ持つ関数オブジェクトをgenericにするアダプタ
template <template <class, class> class FUNC>
struct generic_binary_function
{
    template <class S_, class T_> auto
    operator ()(S_&& x, T_&& y) const
	-> decltype(FUNC<S_, T_>()(std::forward<S_>(x), std::forward<T_>(y)))
    {
	return FUNC<S_, T_>()(std::forward<S_>(x), std::forward<T_>(y));
    }
};

/************************************************************************
*  struct plus								*
************************************************************************/
//! 加算
struct plus
{
    template <class S_, class T_>
    auto	operator ()(const S_& x, const T_& y) const -> decltype(x + y)
		{
		    return x + y;
		}
};
    
/************************************************************************
*  struct minus								*
************************************************************************/
//! 減算
struct minus
{
    template <class S_, class T_>
    auto	operator ()(const S_& x, const T_& y) const -> decltype(x - y)
		{
		    return x - y;
		}
};
    
/************************************************************************
*  struct multiplies							*
************************************************************************/
//! 乗算
struct multiplies
{
    template <class S_, class T_>
    auto	operator ()(const S_& x, const T_& y) const -> decltype(x * y)
		{
		    return x * y;
		}
};
    
/************************************************************************
*  struct divides							*
************************************************************************/
//! 除算
struct divides
{
    template <class S_, class T_>
    auto	operator ()(const S_& x, const T_& y) const -> decltype(x / y)
		{
		    return x / y;
		}
};
    
/************************************************************************
*  struct identity							*
************************************************************************/
//! 恒等関数
struct identity
{
    template <class T_>
    T_&	operator ()(T_&& x)			const	{ return x; }
};

/************************************************************************
*  struct assign							*
************************************************************************/
//! 代入
struct assign
{
    template <class T_, class S_>
    T_&	operator ()(T_&& y, const S_& x)	const	{ y = x; return y; }
};

/************************************************************************
*  struct plus_assign							*
************************************************************************/
//! 引数を加算
struct plus_assign
{
    template <class T_, class S_>
    T_&	operator ()(T_&& y, const S_& x)	const	{ y += x; return y; }
};

/************************************************************************
*  struct minus_assign							*
************************************************************************/
//! 引数を減算
struct minus_assign
{
    template <class T_, class S_>
    T_&	operator ()(T_&& y, const S_& x)	const	{ y -= x; return y; }
};

/************************************************************************
*  struct multiplies_assign						*
************************************************************************/
//! 引数を乗算
struct multiplies_assign
{
    template <class T_, class S_>
    T_&	operator ()(T_&& y, const S_& x)	const	{ y *= x; return y; }
};

/************************************************************************
*  struct divides_assign						*
************************************************************************/
//! 引数を除算
struct divides_assign
{
    template <class T_, class S_>
    T_&	operator ()(T_&& y, const S_& x)	const	{ y /= x; return y; }
};

/************************************************************************
*  struct modulus_assign						*
************************************************************************/
//! 引数で割った時の剰余を代入
struct modulus_assign
{
    template <class T_, class S_>
    T_&	operator ()(T_&& y, const S_& x)	const	{ y %= x; return y; }
};

/************************************************************************
*  struct bit_and_assign						*
************************************************************************/
//! 引数とのAND
struct bit_and_assign
{
    template <class T_, class S_>
    T_&	operator ()(T_&& y, const S_& x)	const	{ y &= x; return y; }
};

/************************************************************************
*  struct bit_or_assign							*
************************************************************************/
//! 引数とのOR
struct bit_or_assign
{
    template <class T_, class S_>
    T_&	operator ()(T_&& y, const S_& x)	const	{ y |= x; return y; }
};

/************************************************************************
*  struct bit_xor_assign						*
************************************************************************/
//! 引数とのXOR
struct bit_xor_assign
{
    template <class T_, class S_>
    T_&	operator ()(T_&& y, const S_& x)	const	{ y ^= x; return y; }
};

/************************************************************************
*  struct equal_to							*
************************************************************************/
//! 等しい
struct equal_to
{
    template <class S_, class T_>
    auto	operator ()(const S_& x, const T_& y) const -> decltype(x == y)
		{
		    return x == y;
		}
};
    
/************************************************************************
*  struct not_equal_to							*
************************************************************************/
//! 等しくない
struct not_equal_to
{
    template <class S_, class T_>
    auto	operator ()(const S_& x, const T_& y) const -> decltype(x != y)
		{
		    return x != y;
		}
};
    
/************************************************************************
*  struct less								*
************************************************************************/
//! より小さい
struct less
{
    template <class S_, class T_>
    auto	operator ()(const S_& x, const T_& y) const -> decltype(x < y)
		{
		    return x < y;
		}
};
    
/************************************************************************
*  struct greater							*
************************************************************************/
//! より大きい
struct greater
{
    template <class S_, class T_>
    auto	operator ()(const S_& x, const T_& y) const -> decltype(x > y)
		{
		    return x > y;
		}
};
    
/************************************************************************
*  struct less_equal							*
************************************************************************/
//! より小さいか等しい
struct less_equal
{
    template <class S_, class T_>
    auto	operator ()(const S_& x, const T_& y) const -> decltype(x <= y)
		{
		    return x <= y;
		}
};
    
/************************************************************************
*  struct greater_equal							*
************************************************************************/
//! より大きいか等しい
struct greater_equal
{
    template <class S_, class T_>
    auto	operator ()(const S_& x, const T_& y) const -> decltype(x >= y)
		{
		    return x >= y;
		}
};
    
/************************************************************************
*  Selection								*
************************************************************************/
template <class S, class T> inline auto
select(bool s, const S& x, const T& y) -> decltype(s ? x : y)
{
    return (s ? x : y);
}
    
/************************************************************************
*  struct generic_select						*
************************************************************************/
struct generic_select
{
    template <class R_, class S_, class T_> auto
    operator ()(const R_& s, const S_& x, const T_& y) const
	-> decltype(select(s, x, y))
    {
	return select(s, x, y);
    }
};

//! 実装の詳細を収める名前空間
namespace detail
{
  /**********************************************************************
  *  struct opnode							*
  **********************************************************************/
  //! 演算子のノードを表すクラス
  struct opnode		{};
    
  /**********************************************************************
  *  type aliases							*
  **********************************************************************/
  //! 式が演算子であるか判定する
  template <class E>
  using is_opnode = std::is_convertible<E, opnode>;

  namespace impl
  {
    template <class T>	struct identity	{ typedef T	type; };

    struct is_range
    {
	template <class E_> static auto
	check(const E_& x) -> decltype(x.begin(), x.end(),
				       std::true_type())	;
	static std::false_type
	check(...)						;
    };

    struct has_const_iterator
    {
	template <class E_> static typename E_::const_iterator
	check(typename E_::const_iterator*)			;
	template <class E_> static void
	check(...)						;
    };
  }	// namespace impl
    
  //! 式がメンバ関数begin()とend()を持つか判定する
  template <class E>
  using is_range = decltype(detail::impl::is_range::check(std::declval<E>()));

  //! 式が持つ定数反復子の型を返す
  /*!
    \param E	式の型
    \return	E が定数反復子を持てばその型，持たなければ void
  */
  template <class E>
  using const_iterator_t
	    = decltype(impl::has_const_iterator::check<E>(nullptr));

  namespace impl
  {
    template <class E>
    struct value_t
    {
	typedef typename std::iterator_traits<
		    decltype(has_const_iterator::
			     check<E>(nullptr))>::value_type	type;
    };
      
    template <class E, class=const_iterator_t<E> >
    struct element_t
    {
	typedef typename value_t<E>::type			F;
	typedef typename element_t<F,
				   const_iterator_t<F> >::type	type;
    };
    template <class E>
    struct element_t<E, void> : identity<E>			{};

    template <class E>
    struct result_t
    {
	typedef typename E::result_type				type;
    };
  }

  //! 式が持つ定数反復子が指す型を返す
  /*!
    定数反復子を持たない式を与えるとコンパイルエラーとなる.
    \param E	定数反復子を持つ式の型
    \return	E の定数反復子が指す型
  */
  template <class E>
  using value_t = typename impl::value_t<E>::type;

  //! 式が持つ定数反復子が指す型を再帰的に辿って到達する型を返す
  /*!
    \param E	式の型
    \return	E が定数反復子を持てばそれが指す型を再帰的に辿って到達する型，
		持たなければ E 自身
  */
  template <class E>
  using element_t = typename impl::element_t<E>::type;

  //! 式の評価結果の型を返す
  /*!
    \param E	式の型
    \return	E が演算子ならば E::result_type, そうでなければ E
  */
  template <class E>
  using	result_t = typename std::conditional<is_opnode<E>::value,
					     impl::result_t<E>,
					     impl::identity<E> >::type::type;
    
  //! 演算子の仮引数として与えられた式の型を返す
  /*!
    \param E	式の型
    \param EVAL	式の評価結果の型を得るならtrue, そうでなければfalse
    \return	E が演算子でなければ const E&, 演算子かつ EVAL == true なら
		E::result_type, 演算子かつ EVAL == false なら const E
  */
  template <class E, bool EVAL>
  using argument_t = typename std::conditional<
			 is_opnode<E>::value,
			 typename std::conditional<
			     EVAL,
			     impl::result_t<E>,
			     impl::identity<const E> >::type,
			 impl::identity<const E&> >::type::type;

#if !defined(__NVCC__)
  /**********************************************************************
  *  class unary_operator<OP, E>					*
  **********************************************************************/
  //! コンテナ式に対する単項演算子を表すクラス
  /*!
    \param OP	各成分に適用される単項演算子の型
    \param E	単項演算子の引数となる式の実体の型
  */
  template <class OP, class E>
  class unary_operator : public opnode
  {
    private:
      typedef value_t<E>		evalue_type;
      typedef is_range<evalue_type>	evalue_is_range;
      typedef argument_t<E, false>	argument_type;
      typedef const_iterator_t<E>	base_iterator;

    public:
    //! 評価結果の型
      typedef result_t<E>		result_type;
    //! 要素の型
      typedef typename std::conditional<
	  evalue_is_range::value,
	  unary_operator<OP, evalue_type>,
	  evalue_type>::type		value_type;
    //! 定数反復子
      class const_iterator
	  : public boost::iterator_adaptor<const_iterator,
					   base_iterator,
					   value_type,
					   boost::use_default,
					   value_type>
      {
	private:
	  typedef boost::iterator_adaptor<const_iterator,
					  base_iterator,
					  value_type,
					  boost::use_default,
					  value_type>	super;

	public:
	  typedef typename super::reference	reference;

	  friend class	boost::iterator_core_access;
	
	public:
			const_iterator(base_iterator iter, const OP& op)
			    :super(iter), _op(op)			{}
	
	private:
	  reference	dereference(std::true_type) const
			{
			    return reference(*super::base(), _op);
			}
	  reference	dereference(std::false_type) const
			{
			    return _op(*super::base());
			}
	  reference	dereference() const
			{
			    return dereference(evalue_is_range());
			}

	private:
	  const OP&	_op;	//!< 単項演算子
      };

      typedef const_iterator	iterator;	//!< 定数反復子の別名
    
    public:
    //! 単項演算子を生成する.
			unary_operator(const E& expr, const OP& op)
			    :_e(expr), _op(op)				{}
      
    //! 演算結果の先頭要素を指す定数反復子を返す.
      const_iterator	begin() const
			{
			    return const_iterator(_e.begin(), _op);
			}
    //! 演算結果の末尾を指す定数反復子を返す.
      const_iterator	end() const
			{
			    return const_iterator(_e.end(), _op);
			}
    //! 演算結果の要素数を返す.
      size_t		size() const
			{
			    return _e.size();
			}
	    
    private:
    // make_unary_operator() に渡される op は temporary object で
    // あり得るので，_op は参照ではなく実体でなければならない．
      argument_type	_e;	//!< 引数となる式の実体
      const OP		_op;	//!< 単項演算子
  };

  template <class OP, class E>
  inline typename std::enable_if<is_range<E>::value,
				 unary_operator<OP, E> >::type
  make_unary_operator(const E& expr, const OP& op)
  {
      return unary_operator<OP, E>(expr, op);
  }
    
  template <class ASSIGN, class E>
  inline typename std::enable_if<is_range<E>::value, E&>::type
  op_assign(E& expr, const element_t<E>& c)
  {
      for (auto dst = expr.begin(); dst != expr.end(); ++dst)
	  ASSIGN()(*dst, c);

      return expr;
  }

  /**********************************************************************
  *  class binary_operator<OP, L, R>					*
  **********************************************************************/
  //! コンテナ式に対する2項演算子を表すクラス
  /*!
    \param OP	各成分に適用される2項演算子の型
    \param L	2項演算子の第1引数となる式の実体の型
    \param R	2項演算子の第2引数となる式の実体の型
  */
  template <class OP, class L, class R>
  class binary_operator : public opnode
  {
    private:
      typedef value_t<L>		lvalue_type;
      typedef value_t<R>		rvalue_type;
      typedef is_range<rvalue_type>	rvalue_is_range;
      typedef argument_t<L, false>	largument_type;
      typedef argument_t<R, false>	rargument_type;
      typedef const_iterator_t<L>	lbase_iterator;
      typedef const_iterator_t<R>	rbase_iterator;

    public:
    //! 評価結果の型
      typedef result_t<R>		result_type;
    //! 要素の型
      typedef typename std::conditional<
	  rvalue_is_range::value,
	  binary_operator<
	      OP, lvalue_type, rvalue_type>,
	  rvalue_type>::type		value_type;
    //! 定数反復子
      class const_iterator
	  : public boost::iterator_adaptor<const_iterator,
					   lbase_iterator,
					   value_type,
					   boost::use_default,
					   value_type>
      {
	private:
	  typedef boost::iterator_adaptor<const_iterator,
					  lbase_iterator,
					  value_type,
					  boost::use_default,
					  value_type>	super;
	
	public:
	  typedef typename super::difference_type	difference_type;
	  typedef typename super::reference		reference;

	  friend class	boost::iterator_core_access;
	
	public:
			const_iterator(lbase_iterator liter,
				       rbase_iterator riter, const OP& op)
			    :super(liter), _riter(riter), _op(op)	{}
	
	private:
	  reference	dereference(std::true_type) const
			{
			    return reference(*super::base(), *_riter, _op);
			}
	  reference	dereference(std::false_type) const
			{
			    return _op(*super::base(), *_riter);
			}
	  reference	dereference() const
			{
			    return dereference(rvalue_is_range());
			}
	  void		advance(difference_type n)
			{
			    super::base_reference() += n;
			    _riter += n;
			}
	  void		increment()
			{
			    ++super::base_reference();
			    ++_riter;
			}
	  void		decrement()
			{
			    --super::base_reference();
			    --_riter;
			}
	
	private:
	  rbase_iterator	_riter;	//!< 第2引数となる式の実体を指す反復子
	  const OP&		_op;	//!< 2項演算子
      };

      typedef const_iterator	iterator;	//!< 定数反復子の別名
    
    public:
    //! 2項演算子を生成する.
			binary_operator(const L& l, const R& r, const OP& op)
			    :_l(l), _r(r), _op(op)
			{
			    assert(_l.size() == _r.size());
			}

    //! 演算結果の先頭要素を指す定数反復子を返す.
      const_iterator	begin() const
			{
			    return const_iterator(_l.begin(), _r.begin(), _op);
			}
    //! 演算結果の末尾を指す定数反復子を返す.
      const_iterator	end() const
			{
			    return const_iterator(_l.end(), _r.end(), _op);
			}
    //! 演算結果の要素数を返す.
      size_t		size() const
			{
			    return _l.size();
			}
    
    private:
      largument_type	_l;	//!< 第1引数となる式の実体
      rargument_type	_r;	//!< 第2引数となる式の実体
      const OP		_op;	//!< 2項演算子
  };

  template <class OP, class L, class R>
  inline typename std::enable_if<(is_range<L>::value && is_range<R>::value),
				 binary_operator<OP, L, R> >::type
  make_binary_operator(const L& l, const R& r, const OP& op)
  {
      return binary_operator<OP, L, R>(l, r, op);
  }

  template <class ASSIGN, class L, class R>
  inline typename std::enable_if<(is_range<L>::value && is_range<R>::value),
				 L&>::type
  op_assign(L& l, const R& r)
  {
      assert(l.size() == r.size());
      auto	src = r.begin();
      for (auto dst = l.begin(); dst != l.end(); ++dst)
      {
	  ASSIGN()(*dst, *src);
	  ++src;
      }

      return l;
  }
#endif	// !__NVCC__
}	// End of namespace TU::detail

template <class E>	using is_range  = detail::is_range<E>;
template <class E>	using element_t = detail::element_t<E>;

#if !defined(__NVCC__)
//! 与えられた式の各要素の符号を反転する.
/*!
  \param expr	式
  \return	符号反転演算子ノード
*/
template <class E, typename std::enable_if<is_range<E>::value>::type* = nullptr>
inline auto
operator -(const E& expr)
    -> decltype(detail::make_unary_operator(
		    expr, std::negate<element_t<E> >()))
{
    return detail::make_unary_operator(expr,
				       std::negate<element_t<E> >());
}

//! 与えられた式の各要素に定数を掛ける.
/*!
  \param expr	式
  \param c	乗数
  \return	乗算演算子ノード
*/
template <class E,
	  typename std::enable_if<is_range<E>::value>::type* = nullptr>
inline auto
operator *(const E& expr, const element_t<E>& c)
    -> decltype(detail::make_unary_operator(
		    expr, std::bind(std::multiplies<element_t<E> >(),
				    std::placeholders::_1, c)))
{
    return detail::make_unary_operator(
	       expr, std::bind(std::multiplies<element_t<E> >(),
			       std::placeholders::_1, c));
}

//! 与えられた式の各要素に定数を掛ける.
/*!
  \param c	乗数
  \param expr	式
  \return	乗算演算子ノード
*/
template <class E, typename std::enable_if<is_range<E>::value>::type* = nullptr>
inline auto
operator *(const element_t<E>& c, const E& expr)
    -> decltype(detail::make_unary_operator(
		    expr, std::bind(std::multiplies<element_t<E> >(),
				    c, std::placeholders::_1)))
{
    return detail::make_unary_operator(
	       expr, std::bind(std::multiplies<element_t<E> >(),
			       c, std::placeholders::_1));
}

//! 与えられた式の各要素を定数で割る.
/*!
  \param expr	式
  \param c	除数
  \return	除算演算子ノード
*/
template <class E,
	  typename std::enable_if<is_range<E>::value>::type* = nullptr>
inline auto
operator /(const E& expr, const element_t<E>& c)
    -> decltype(detail::make_unary_operator(
		    expr, std::bind(std::divides<element_t<E> >(),
				    std::placeholders::_1, c)))
{
    return detail::make_unary_operator(
	       expr, std::bind(std::divides<element_t<E> >(),
			       std::placeholders::_1, c));
}

//! 与えられた式の各要素に定数を掛ける.
/*!
  \param expr	式
  \param c	乗数
  \return	各要素にcが掛けられた結果の式
*/
template <class E>
inline typename std::enable_if<
    is_range<typename std::decay<E>::type>::value, E&>::type
operator *=(E&& expr, const element_t<typename std::decay<E>::type>& c)
{
    return detail::op_assign<multiplies_assign>(expr, c);
}

//! 与えられた式の各要素を定数で割る.
/*!
  \param expr	式
  \param c	除数
  \return	各要素がcで割られた結果の式
*/
template <class E>
inline typename std::enable_if<
    is_range<typename std::decay<E>::type>::value, E&>::type
operator /=(E&& expr, const element_t<typename std::decay<E>::type>& c)
{
    return detail::op_assign<divides_assign>(expr, c);
}

//! 与えられた2つの式の各要素の和をとる.
/*!
  \param l	左辺の式
  \param r	右辺の式
  \return	加算演算子ノード
*/
template <class L, class R,
	  typename std::enable_if<
	      (is_range<L>::value && is_range<R>::value)>::type* = nullptr>
inline auto
operator +(const L& l, const R& r)
    -> decltype(detail::make_binary_operator(l, r, std::plus<element_t<R> >()))
{
    return detail::make_binary_operator(l, r,
					std::plus<element_t<R> >());
}

//! 与えられた2つの式の各要素の差をとる.
/*!
  \param l	左辺の式
  \param r	右辺の式
  \return	減算演算子ノード
*/
template <class L, class R,
	  typename std::enable_if<
	      (is_range<L>::value && is_range<R>::value)>::type* = nullptr>
inline auto
operator -(const L& l, const R& r)
    -> decltype(detail::make_binary_operator(l, r, std::minus<element_t<R> >()))
{
    return detail::make_binary_operator(l, r, std::minus<element_t<R> >());
}

//! 与えられた左辺の式の各要素に右辺の式の各要素を加える.
/*!
  \param l	左辺の式
  \param r	右辺の式
  \return	各要素が加算された左辺の式
*/
template <class L, class R>
inline typename std::enable_if<
    (is_range<typename std::decay<L>::type>::value && is_range<R>::value),
    L&>::type
operator +=(L&& l, const R& r)
{
    return detail::op_assign<plus_assign>(l, r);
}

//! 与えられた左辺の式の各要素から右辺の式の各要素を減じる.
/*!
  \param l	左辺の式
  \param r	右辺の式
  \return	各要素が減じられた左辺の式
*/
template <class L, class R>
inline typename std::enable_if<
    (is_range<typename std::decay<L>::type>::value && is_range<R>::value),
    L&>::type
operator -=(L&& l, const R& r)
{
    return detail::op_assign<minus_assign>(l, r);
}

/************************************************************************
*  various numeric functions						*
************************************************************************/
//! 与えられた式の各要素の自乗和を求める.
/*!
  \param x	式
  \return	式の各要素の自乗和
*/
template <class T>
constexpr inline typename std::enable_if<!is_range<T>::value, T>::type
square(const T& x)
{
    return x * x;
}
template <class E>
inline typename std::enable_if<is_range<E>::value, element_t<E> >::type
square(const E& expr)
{
    typedef element_t<E>		element_type;
    typedef typename E::value_type	value_type;
    
    return std::accumulate(expr.begin(), expr.end(), element_type(0),
			   [](const element_type& init, const value_type& val)
			   { return init + square(val); });
}

//! 与えられた式の各要素の自乗和の平方根を求める.
/*!
  \param x	式
  \return	式の各要素の自乗和の平方根
*/
template <class T> inline auto
length(const T& x) -> decltype(std::sqrt(square(x)))
{
    return std::sqrt(square(x));
}
    
//! 与えられた二つの式の各要素の差の自乗和を求める.
/*!
  \param x	第1の式
  \param y	第2の式
  \return	xとyの各要素の差の自乗和
*/
template <class L, class R> inline auto
sqdist(const L& x, const R& y) -> decltype(square(x - y))
{
    return square(x - y);
}
    
//! 与えられた二つの式の各要素の差の自乗和の平方根を求める.
/*!
  \param x	第1の式
  \param y	第2の式
  \return	xとyの各要素の差の自乗和の平方根
*/
template <class L, class R> inline auto
dist(const L& x, const R& y) -> decltype(std::sqrt(sqdist(x, y)))
{
    return std::sqrt(sqdist(x, y));
}
#endif	// !__NVCC__

//! 与えられた二つの整数の最大公約数を求める．
/*!
  \param m	第1の整数
  \param n	第2の整数
  \return	mとnの最大公約数
*/
template <class S, class T> constexpr typename std::common_type<S, T>::type
gcd(S m, T n)
{
    return (n == 0 ? m : gcd(n, m % n));
}

//! 与えられた三つ以上の整数の最大公約数を求める．
/*!
  \param m	第1の整数
  \param n	第2の整数
  \param args	第3, 第4,...の整数
  \return	m, n, args...の最大公約数
*/
template <class S, class T, class... ARGS>
constexpr typename std::common_type<S, T, ARGS...>::type
gcd(S m, T n, ARGS... args)
{
    return gcd(gcd(m, n), args...);
}

//! 与えられた二つの整数の最小公倍数を求める．
/*!
  \param m	第1の整数
  \param n	第2の整数
  \return	mとnの最小公倍数
*/
template <class S, class T> constexpr typename std::common_type<S, T>::type
lcm(S m, T n)
{
    return (m*n == 0 ? 0 : (m / gcd(m, n)) * n);
}

//! 与えられた三つ以上の整数の最小公倍数を求める．
/*!
  \param m	第1の整数
  \param n	第2の整数
  \param args	第3, 第4,...の整数
  \return	m, n, args...の最小公倍数
*/
template <class S, class T, class... ARGS>
constexpr typename std::common_type<S, T, ARGS...>::type
lcm(S m, T n, ARGS... args)
{
    return lcm(lcm(m, n), args...);
}

}	// End of namespace TU
#endif	// !__TU_FUNCTIONAL_H
