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

#include <functional>
#include <cassert>
#include <boost/tuple/tuple.hpp>
#include <boost/type_traits.hpp>

namespace TU
{
/************************************************************************
*  class op_node							*
************************************************************************/
//! 演算子のノードを表すクラス
struct op_node
{
};
    
/************************************************************************
*  class container<E>							*
************************************************************************/
//! コンテナを表すクラス
/*!
  \param C	コンテナの実体の型
*/
template <class C>
struct container
{
  //! この式の実体への参照を返す.
    const C&	operator ()() const
		{
		    return static_cast<const C&>(*this);
		}
};

//! 実装の詳細を収める名前空間
namespace detail
{
  /**********************************************************************
  *  meta template predicates						*
  **********************************************************************/
  //! 与えられた型がコンテナであるかを判別するテンプレート述語
  /*!
    \param T	任意の型
  */
    template <class T>
    class is_container
    {
      private:
	typedef char	Small;
	struct Big	{ Small dummy[2]; };
	
	template <class _C>
	static Small	test(container<_C>)	;
	static Big	test(...)		;
	static T	makeT()			;
    
      public:
	enum		{ value = (sizeof(test(makeT())) == sizeof(Small)) };
    };

  //! 与えられた式の評価結果の型を返すテンプレート述語
  /*!
    \param E	式を表す型
  */
    template <class E>
    struct ResultType
    {
	typedef typename E::result_type			type;
    };
    
  //! 式の要素型が演算子でない配列ならばその基底配列型, そうでなければ要素型そのもの
  /*!
    \param E	value_typeを持つ式を表す型
  */
    template <class E>
    class ValueType
    {
      private:
	typedef typename E::value_type			value_type;

      public:
	typedef typename boost::mpl::eval_if_c<
	    (is_container<value_type>::value &&
	     !boost::is_convertible<value_type, op_node>::value),
	    ResultType<value_type>,
	    boost::mpl::identity<value_type> >::type	type;
    };

  //! 式が演算子ノードならばそれ自体もしくはその評価結果，そうでなければそれへの参照
  /*!
    \param E	コンテナ式を表す型
    \param EVAL	演算子ノードを評価するならtrue, そうでなければfalse
  */
    template <class E, bool EVAL=false>
    struct ArgumentType
    {
	typedef typename boost::mpl::if_<
	    boost::is_convertible<E, op_node>,
	    typename boost::mpl::if_c<
		EVAL, const typename E::result_type, const E>::type,
	    const E&>::type				type;
    };
}
    
/************************************************************************
*  class unary_operator<OP, E>						*
************************************************************************/
//! コンテナ式に対する単項演算子を表すクラス
/*!
  \param OP	各成分に適用される単項演算子の型
  \param E	単項演算子の引数となる式の実体の型
*/
template <class OP, class E>
class unary_operator : public container<unary_operator<OP, E> >,
		       public op_node
{
  private:
    typedef typename detail::ArgumentType<E>::type	argument_type;
    typedef typename detail::ValueType<E>::type		evalue_type;
    typedef boost::mpl::bool_<
	detail::is_container<evalue_type>::value>	evalue_is_expr;

  public:
  //! 評価結果の型
    typedef typename E::result_type			result_type;
  //! 成分の型
    typedef typename result_type::element_type		element_type;
  //! 要素の型
    typedef typename boost::mpl::if_<
	evalue_is_expr, unary_operator<OP, evalue_type>,
	element_type>::type				value_type;
  //! 定数反復子
    class const_iterator
	: public boost::iterator_adaptor<const_iterator,
					 typename E::const_iterator,
					 value_type,
					 boost::use_default,
					 value_type>
    {
      private:
	typedef boost::iterator_adaptor<
		    const_iterator,
		    typename E::const_iterator,
		    value_type,
		    boost::use_default,
		    value_type>				super;
	typedef typename super::base_type		base_type;

      public:
	typedef typename super::difference_type		difference_type;
	typedef typename super::pointer			pointer;
	typedef typename super::reference		reference;
	typedef typename super::iterator_category	iterator_category;

	friend class	boost::iterator_core_access;
	
      public:
	const_iterator(base_type iter, const OP& op)
	    :super(iter), _op(op)					{}
	
      private:
	reference	dereference(boost::mpl::true_) const
			{
			    return reference(*super::base(), _op);
			}
	reference	dereference(boost::mpl::false_) const
			{
			    return _op(*super::base());
			}
	reference	dereference() const
			{
			    return dereference(evalue_is_expr());
			}

      private:
	const OP&	_op;	//!< 単項演算子
    };

    typedef const_iterator	iterator;	//!< 定数反復子の別名
    
  public:
  //! 単項演算子を生成する.
    unary_operator(const container<E>& expr, const OP& op)
	:_e(expr()), _op(op)						{}

  //! 演算結果の先頭要素を指す定数反復子を返す.
    const_iterator	begin() const
			{
			    return const_iterator(_e.begin(), _op);
			}
  //! 演算結果の先頭要素を指す定数反復子を返す.
    const_iterator	cbegin() const
			{
			    return begin();
			}
  //! 演算結果の末尾を指す定数反復子を返す.
    const_iterator	end() const
			{
			    return const_iterator(_e.end(), _op);
			}
  //! 演算結果の末尾を指す定数反復子を返す.
    const_iterator	cend() const
			{
			    return end();
			}
  //! 演算結果の要素数を返す.
    size_t		size() const
			{
			    return _e.size();
			}
  //! 演算結果の列数を返す.
    size_t		ncol() const
			{
			    return _e.ncol();
			}
	    
  private:
    argument_type	_e;	//!< 引数となる式の実体
    const OP		_op;	//!< 単項演算子
};

//! 与えられたコンテナ式の各要素の符号を反転する.
/*!
  \param expr	コンテナ式
  \return	符号反転演算子ノード
*/
template <class E>
inline unary_operator<std::negate<typename E::element_type>, E>
operator -(const container<E>& expr)
{
    typedef std::negate<typename E::element_type>	op_type;
    
    return unary_operator<op_type, E>(expr, op_type());
}

//! 与えられたコンテナ式の各要素に定数を掛ける.
/*!
  \param expr	コンテナ式
  \param c	乗数
  \return	乗算演算子ノード
*/
template <class E> inline
unary_operator<std::binder2nd<std::multiplies<typename E::element_type> >, E>
operator *(const container<E>& expr, typename E::element_type c)
{
    typedef std::multiplies<typename E::element_type>	op_type;
    typedef std::binder2nd<op_type>			binder_type;
    
    return unary_operator<binder_type, E>(expr, binder_type(op_type(), c));
}

//! 与えられたコンテナ式の各要素に定数を掛ける.
/*!
  \param c	乗数
  \param expr	コンテナ式
  \return	乗算演算子ノード
*/
template <class E> inline
unary_operator<std::binder1st<std::multiplies<typename E::element_type> >, E>
operator *(typename E::element_type c, const container<E>& expr)
{
    typedef std::multiplies<typename E::element_type>	op_type;
    typedef std::binder1st<op_type>			binder_type;
    
    return unary_operator<binder_type, E>(expr, binder_type(op_type(), c));
}

//! 与えられたコンテナ式の各要素を定数で割る.
/*!
  \param expr	コンテナ式
  \param c	除数
  \return	除算演算子ノード
*/
template <class E> inline
unary_operator<std::binder2nd<std::divides<typename E::element_type> >, E>
operator /(const container<E>& expr, typename E::element_type c)
{
    typedef std::divides<typename E::element_type>	op_type;
    typedef std::binder2nd<op_type>			binder_type;
    
    return unary_operator<binder_type, E>(expr, binder_type(op_type(), c));
}

/************************************************************************
*  class binary_operator<OP, L, R>					*
************************************************************************/
//! コンテナ式に対する2項演算子を表すクラス
/*!
  \param OP	各成分に適用される2項演算子の型
  \param L	2項演算子の第1引数となる式の実体の型
  \param R	2項演算子の第2引数となる式の実体の型
*/
template <class OP, class L, class R>
class binary_operator : public container<binary_operator<OP, L, R> >,
			public op_node
{
  private:
    typedef typename detail::ArgumentType<L>::type	largument_type;
    typedef typename detail::ArgumentType<R>::type	rargument_type;
    typedef typename detail::ValueType<L>::type		lvalue_type;
    typedef typename detail::ValueType<R>::type		rvalue_type;
    typedef boost::mpl::bool_<
	detail::is_container<lvalue_type>::value>	lvalue_is_expr;

  public:
  //! 評価結果の型
    typedef typename R::result_type			result_type;
  //! 成分の型
    typedef typename result_type::element_type		element_type;
  //! 要素の型
    typedef typename boost::mpl::if_<
	lvalue_is_expr,
	binary_operator<OP, lvalue_type, rvalue_type>,
	element_type>::type				value_type;
  //! 定数反復子
    class const_iterator
	: public boost::iterator_adaptor<const_iterator,
					 typename L::const_iterator,
					 value_type,
					 boost::use_default,
					 value_type>
    {
      private:
	typedef boost::iterator_adaptor<
		    const_iterator,
		    typename L::const_iterator,
		    value_type,
		    boost::use_default,
		    value_type>				super;
	typedef typename super::base_type		lbase_type;
	typedef typename R::const_iterator		rbase_type;
	
      public:
	typedef typename super::difference_type		difference_type;
	typedef typename super::pointer			pointer;
	typedef typename super::reference		reference;
	typedef typename super::iterator_category	iterator_category;

	friend class	boost::iterator_core_access;
	
      public:
	const_iterator(lbase_type liter, rbase_type riter, const OP& op)
	    :super(liter), _riter(riter), _op(op)			{}
	
      private:
	reference	dereference(boost::mpl::true_) const
			{
			    return reference(*super::base(), *_riter, _op);
			}
	reference	dereference(boost::mpl::false_) const
			{
			    return _op(*super::base(), *_riter);
			}
	reference	dereference() const
			{
			    return dereference(lvalue_is_expr());
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
	rbase_type	_riter;	//!< 第2引数となる式の実体を指す反復子
	const OP&	_op;	//!< 2項演算子
    };

    typedef const_iterator	iterator;	//!< 定数反復子の別名
    
  public:
  //! 2項演算子を生成する.
    binary_operator(const container<L>& l,
		   const container<R>& r, const OP& op)
	:_l(l()), _r(r()), _op(op)
			{
			    assert(_l.size() == _r.size());
			}

  //! 演算結果の先頭要素を指す定数反復子を返す.
    const_iterator	begin() const
			{
			    return const_iterator(_l.begin(),
						  _r.begin(), _op);
			}
  //! 演算結果の先頭要素を指す定数反復子を返す.
    const_iterator	cbegin() const
			{
			    return begin();
			}
  //! 演算結果の末尾を指す定数反復子を返す.
    const_iterator	end() const
			{
			    return const_iterator(_l.end(), _r.end(), _op);
			}
  //! 演算結果の末尾を指す定数反復子を返す.
    const_iterator	cend() const
			{
			    return end();
			}
  //! 演算結果の要素数を返す.
    size_t		size() const
			{
			    return _l.size();
			}
  //! 演算結果の列数を返す.
    size_t		ncol() const
			{
			    return _l.ncol();
			}
    
  private:
    largument_type	_l;	//!< 第1引数となる式の実体
    rargument_type	_r;	//!< 第2引数となる式の実体
    const OP		_op;	//!< 2項演算子
};

//! 与えられた2つのコンテナ式の各要素の和をとる.
/*!
  \param l	左辺のコンテナ式
  \param r	右辺のコンテナ式
  \return	加算演算子ノード
*/
template <class L, class R>
inline binary_operator<std::plus<typename R::element_type>, L, R>
operator +(const container<L>& l, const container<R>& r)
{
    typedef std::plus<typename R::element_type>		op_type;

    return binary_operator<op_type, L, R>(l, r, op_type());
}

//! 与えられた2つのコンテナ式の各要素の差をとる.
/*!
  \param l	左辺のコンテナ式
  \param r	右辺のコンテナ式
  \return	減算演算子ノード
*/
template <class L, class R>
inline binary_operator<std::minus<typename R::element_type>, L, R>
operator -(const container<L>& l, const container<R>& r)
{
    typedef std::minus<typename R::element_type>	op_type;

    return binary_operator<op_type, L, R>(l, r, op_type());
}

/************************************************************************
*  struct identity<T>							*
************************************************************************/
//! 恒等関数
/*!
  \param T	引数と結果の型
*/
template <class T>
struct identity
{
    typedef T	argument_type;
    typedef T	result_type;
    
    T&	operator ()(T& x)			const	{ return x; }
};

/************************************************************************
*  struct assign<S, T>							*
************************************************************************/
//! 代入
/*!
  \param S	引数の型
  \param T	代入先の型
*/
template <class S, class T>
struct assign
{
    typedef S						first_argument_type;
    typedef typename boost::remove_reference<T>::type	second_argument_type;
    typedef void					result_type;
    
    void	operator ()(const S& x, T y)	const	{ y = x; }
};

/************************************************************************
*  struct plus_assign<S, T>						*
************************************************************************/
//! 引数を加算
/*!
  \param S	引数の型
  \param T	加算元の型
*/
template <class S, class T>
struct plus_assign
{
    typedef S						first_argument_type;
    typedef typename boost::remove_reference<T>::type	second_argument_type;
    typedef void					result_type;
    
    void	operator ()(const S& x, T y)	const	{ y += x; }
};

/************************************************************************
*  struct minus_assign<S, T>						*
************************************************************************/
//! 引数を減算
/*!
  \param S	引数の型
  \param T	減算元の型
*/
template <class S, class T>
struct minus_assign
{
    typedef S						first_argument_type;
    typedef typename boost::remove_reference<T>::type	second_argument_type;
    typedef void					result_type;
    
    void	operator ()(const S& x, T y)	const	{ y -= x; }
};

/************************************************************************
*  struct multiplies_assign<S, T>					*
************************************************************************/
//! 引数を乗算
/*!
  \param S	引数の型
  \param T	乗算元の型
*/
template <class S, class T>
struct multiplies_assign
{
    typedef S						first_argument_type;
    typedef typename boost::remove_reference<T>::type	second_argument_type;
    typedef void					result_type;
    
    void	operator ()(const S& x, T y)	const	{ y *= x; }
};

/************************************************************************
*  struct divides_assign<S, T>						*
************************************************************************/
//! 引数を除算
/*!
  \param S	引数の型
  \param T	除算元の型
*/
template <class S, class T>
struct divides_assign
{
    typedef S						first_argument_type;
    typedef typename boost::remove_reference<T>::type	second_argument_type;
    typedef void					result_type;
    
    void	operator ()(const S& x, T y)	const	{ y /= x; }
};

/************************************************************************
*  struct modulus_assign<S, T>						*
************************************************************************/
//! 引数で割った時の剰余を代入
/*!
  \param S	引数の型
  \param T	剰余をとる元の型
*/
template <class S, class T>
struct modulus_assign
{
    typedef S						first_argument_type;
    typedef typename boost::remove_reference<T>::type	second_argument_type;
    typedef void					result_type;
    
    void	operator ()(const S& x, T y)	const	{ y %= x; }
};

/************************************************************************
*  struct bit_and_assign<S, T>						*
************************************************************************/
//! 引数とのAND
/*!
  \param S	引数の型
  \param T	ANDをとる元の型
*/
template <class S, class T>
struct bit_and_assign
{
    typedef S						first_argument_type;
    typedef typename boost::remove_reference<T>::type	second_argument_type;
    typedef void					result_type;
    
    void	operator ()(const S& x, T y)	const	{ y &= x; }
};

/************************************************************************
*  struct bit_or_assign<S, T>						*
************************************************************************/
//! 引数とのOR
/*!
  \param S	引数の型
  \param T	ORをとる元の型
*/
template <class S, class T>
struct bit_or_assign
{
    typedef S						first_argument_type;
    typedef typename boost::remove_reference<T>::type	second_argument_type;
    typedef void					result_type;
    
    void	operator ()(const S& x, T y)	const	{ y |= x; }
};

/************************************************************************
*  struct bit_xor_assign<S, T>						*
************************************************************************/
//! 引数とのXOR
/*!
  \param S	引数の型
  \param T	XORをとる元の型
*/
template <class S, class T>
struct bit_xor_assign
{
    typedef S						first_argument_type;
    typedef typename boost::remove_reference<T>::type	second_argument_type;
    typedef void					result_type;
    
    void	operator ()(const S& x, T y)	const	{ y ^= x; }
};

/************************************************************************
*  class unarize2<FUNC>							*
************************************************************************/
//! 引数をtupleにまとめることによって2変数関数を1変数関数に変換
template <class FUNC>
class unarize2
{
  public:
    typedef FUNC						functor_type;
    typedef boost::tuple<typename FUNC::first_argument_type,
			 typename FUNC::second_argument_type>	argument_type;
    typedef typename FUNC::result_type				result_type;

    class mm : public unarize2<typename FUNC::mm>
    {
      private:
	typedef unarize2<typename FUNC::mm>		super;

      public:
	mm(const unarize2& u)	:super(u.functor())	{}
    };

  public:
    unarize2(const FUNC& func=FUNC())	:_func(func)	{}

    result_type	operator ()(const argument_type& arg) const
		{
		    return _func(boost::get<0>(arg), boost::get<1>(arg));
		}

    const FUNC&	functor()			const	{return _func;}

  private:
    const FUNC	_func;
};

template <class FUNC> inline unarize2<FUNC>
make_unary_function2(const FUNC& func)
{
    return unarize2<FUNC>(func);
}
    
/************************************************************************
*  class unarize3<FUNC>							*
************************************************************************/
//! 引数をtupleにまとめることによって3変数関数を1変数関数に変換
template <class FUNC>
class unarize3
{
  public:
    typedef FUNC						functor_type;
    typedef typename FUNC::result_type				result_type;
    typedef boost::tuple<typename FUNC::first_argument_type,
			 typename FUNC::second_argument_type,
			 typename FUNC::third_argument_type>	argument_type;
    
    class mm_ : public unarize3<typename FUNC::mm_>
    {
      private:
	typedef unarize3<typename FUNC::mm_>		super;

      public:
	mm_(const unarize3& u)	:super(u.functor())	{}
    };

  public:
    unarize3(const FUNC& func=FUNC())	:_func(func)	{}

    result_type	operator ()(const argument_type& arg) const
		{
		    return _func(boost::get<0>(arg),
				 boost::get<1>(arg), boost::get<2>(arg));
		}

    const FUNC&	functor()			const	{return _func;}

  private:
    const FUNC	_func;
};

template <class FUNC> inline unarize3<FUNC>
make_unary_function3(const FUNC& func)
{
    return unarize3<FUNC>(func);
}
    
/************************************************************************
*  class unarize4<FUNC>							*
************************************************************************/
//! 引数をtupleにまとめることによって4変数関数を1変数関数に変換
template <class FUNC>
class unarize4
{
  public:
    typedef FUNC						functor_type;
    typedef typename FUNC::result_type				result_type;
    typedef boost::tuple<typename FUNC::first_argument_type,
			 typename FUNC::second_argument_type,
			 typename FUNC::third_argument_type,
			 typename FUNC::fourth_argument_type>	argument_type;
    
    class mm_ : public unarize4<typename FUNC::mm_>
    {
      private:
	typedef unarize3<typename FUNC::mm_>		super;

      public:
	mm_(const unarize4& u)	:super(u.functor())	{}
    };

  public:
    unarize4(const FUNC& func=FUNC())	:_func(func)	{}

    result_type	operator ()(const argument_type& arg) const
		{
		    return _func(boost::get<0>(arg), boost::get<1>(arg),
				 boost::get<2>(arg), boost::get<3>(arg));
		}

    const FUNC&	functor()			const	{return _func;}

  private:
    const FUNC	_func;
};

template <class FUNC> inline unarize4<FUNC>
make_unary_function4(const FUNC& func)
{
    return unarize4<FUNC>(func);
}
    
/************************************************************************
*  class mem_var_t							*
************************************************************************/
//! S型のメンバ変数を持つT型オブジェクトへのポインタからそのメンバに直接アクセス(R/W)する関数オブジェクト
/*
  \param S	T型オブジェクトのメンバ変数の型
  \param T	S型メンバ変数を所有するオブジェクトの型
*/ 
template <class S, class T>
class mem_var_t : public std::unary_function<T*, S>
{
  public:
    typedef S T::*	mem_ptr;
    
    explicit mem_var_t(mem_ptr m)	:_m(m)		{}

    S&		operator ()(T* p)		const	{return p->*_m;}

  private:
    const mem_ptr	_m;
};

//! S型のメンバ変数を持つT型オブジェクトへのポインタからそのメンバに直接アクセス(R/W)する関数オブジェクトを生成する
template <class S, class T> inline mem_var_t<S, T>
mem_var(S T::*m)
{
    return mem_var_t<S, T>(m);
}

/************************************************************************
*  class const_mem_var_t						*
************************************************************************/
//! S型のメンバ変数を持つT型オブジェクトへのポインタからそのメンバに直接アクセス(R)する関数オブジェクト
/*
  \param S	T型オブジェクトのメンバ変数の型
  \param T	S型メンバ変数を所有するオブジェクトの型
*/ 
template <class S, class T>
class const_mem_var_t : public std::unary_function<const T*, S>
{
  public:
    typedef S T::*	mem_ptr;
    
    explicit const_mem_var_t(mem_ptr m)		:_m(m)	{}

    const S&	operator ()(const T* p)		const	{return p->*_m;}

  private:
    const mem_ptr	_m;
};

//! S型のメンバ変数を持つT型オブジェクトへのポインタからそのメンバに直接アクセス(R)する関数オブジェクトを生成する
template <class S, class T> inline const_mem_var_t<S, T>
const_mem_var(S const T::* m)
{
    return const_mem_var_t<S, T>(m);
}

/************************************************************************
*  class mem_var_ref_t							*
************************************************************************/
//! S型のメンバ変数を持つT型オブジェクトへの参照からそのメンバに直接アクセス(R/W)する関数オブジェクト
/*
  \param S	T型オブジェクトのメンバ変数の型
  \param T	S型メンバ変数を所有するオブジェクトの型
*/ 
template <class S, class T>
class mem_var_ref_t : public std::unary_function<T&, S>
{
  public:
    typedef S T::*	mem_ptr;
    
    explicit mem_var_ref_t(mem_ptr m)		:_m(m)	{}

    S&		operator ()(T& p)		const	{return p.*_m;}

  private:
    const mem_ptr	_m;
};
    
//! S型のメンバ変数を持つT型オブジェクトへの参照からそのメンバに直接アクセス(R/W)する関数オブジェクトを生成する
template <class S, class T> inline mem_var_ref_t<S, T>
mem_var_ref(S T::*m)
{
    return mem_var_ref_t<S, T>(m);
}

/************************************************************************
*  class const_mem_var_ref_t						*
************************************************************************/
//! S型のメンバ変数を持つT型オブジェクトへの参照からそのメンバに直接アクセス(R)する関数オブジェクト
/*
  \param S	T型オブジェクトのメンバ変数の型
  \param T	S型メンバ変数を所有するオブジェクトの型
*/ 
template <class S, class T>
class const_mem_var_ref_t : public std::unary_function<const T&, S>
{
  public:
    typedef S const T::*	mem_ptr;
    
    explicit const_mem_var_ref_t(mem_ptr m)	:_m(m)	{}

    const S&	operator ()(const T& p)		const	{return p.*_m;}

  private:
    const mem_ptr	_m;
};
    
//! S型のメンバ変数を持つT型オブジェクトへの参照からそのメンバに直接アクセス(R)する関数オブジェクトを生成する
template <class S, class T> inline const_mem_var_ref_t<S, T>
const_mem_var_ref(S const T::* m)
{
    return const_mem_var_ref_t<S, T>(m);
}

}	// namespace TU
#endif	// !__TU_FUNCTIONAL_H
