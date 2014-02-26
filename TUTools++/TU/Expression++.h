/*
 *  平成14-19年（独）産業技術総合研究所 著作権所有
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
 *  Copyright 2002-2007.
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
  \file		Expression++.h
  \brief	遅延評価による効率的な配列実装のための式と演算子
*/
#ifndef __TUExpressionPP_h
#define __TUExpressionPP_h

#include <boost/type_traits.hpp>
#include <boost/iterator/iterator_adaptor.hpp>
#include <functional>
#include <iostream>

namespace TU
{
/************************************************************************
*  class OpNode								*
************************************************************************/
//! 演算子のノードを表すクラス
struct OpNode
{
};
    
/************************************************************************
*  class Expression<E>							*
************************************************************************/
//! 配列式を表すクラス
/*!
  \param E	式の実体の型
*/
template <class E>
struct Expression
{
  //! この式の実体への参照を返す.
    const E&	operator ()() const
		{
		    return static_cast<const E&>(*this);
		}
};

/************************************************************************
*  meta template predicates						*
************************************************************************/
namespace detail
{
  //! 与えられた型が配列式型であるかを判別するテンプレート述語
  /*!
    \param T	任意の型
  */
    template <class T>
    class IsExpression
    {
      private:
	typedef char	Small;
	struct Big	{ Small dummy[2]; };
	
	template <class _E>
	static Small	test(Expression<_E>)	;
	static Big	test(...)		;
	static T	makeT()			;
    
      public:
	enum		{ value = (sizeof(test(makeT())) == sizeof(Small)) };
    };

  //! 与えられた配列式型の評価結果の型を返すテンプレート述語
  /*!
    \param E	配列式を表す型
  */
    template <class E>
    struct ResultType
    {
	typedef typename E::result_type			type;
    };
    
  //! 配列式の要素型が演算子でない配列ならばその基底配列型, そうでなければ要素型そのもの
  /*!
    \param E	配列式を表す型
  */
    template <class E>
    class ValueType
    {
      private:
	typedef typename E::value_type			value_type;

      public:
	typedef typename boost::mpl::eval_if_c<
	    (IsExpression<value_type>::value &&
	     !boost::is_convertible<value_type, OpNode>::value),
	    ResultType<value_type>,
	    boost::mpl::identity<value_type> >::type	type;
    };

  //! 配列式が演算子ノードならばそれ自体もしくはその評価結果，そうでなければそれへの参照
  /*!
    \param E	配列式を表す型
    \param EVAL	演算子ノードを評価するならtrue, そうでなければfalse
  */
    template <class E, bool EVAL=false>
    struct ArgumentType
    {
	typedef typename boost::mpl::if_<
	    boost::is_convertible<E, OpNode>,
	    typename boost::mpl::if_c<
		EVAL, typename E::result_type, E>::type,
	    const E&>::type				type;
    };
}
    
/************************************************************************
*  class column_proxy<A>						*
************************************************************************/
//! 2次元配列の列を表す代理オブジェクト
/*!
  \param A	2次元配列の型
*/
template <class A>
class column_proxy : public Expression<column_proxy<A> >
{
  public:
  //! 定数反復子
    typedef vertical_iterator<typename A::const_iterator>
							const_iterator;
  //! 反復子
    typedef vertical_iterator<typename A::iterator>	iterator;
  //! 定数逆反復子
    typedef std::reverse_iterator<const_iterator>	const_reverse_iterator;
  //! 逆反復子
    typedef std::reverse_iterator<iterator>		reverse_iterator;
  //! 要素の型
    typedef typename iterator::value_type		value_type;
  //! 定数要素への参照
    typedef typename const_iterator::reference		const_reference;
  //! 要素への参照
    typedef typename iterator::reference		reference;
  //! 評価結果の型
    typedef column_proxy<typename A::result_type>	result_type;
    
  public:
  //! 2次元配列の列を表す代理オブジェクトを生成する.
  /*!
    \param a	2次元配列
    \param col	列を指定するindex
  */
    column_proxy(A& a, size_t col)	:_a(a), _col(col)		{}

  //! この列に他の配列を代入する.
  /*!
    \param expr	代入元の配列を表す式
    \return	この列
  */
    template <class E>
    column_proxy&		operator =(const Expression<E>& expr)
				{
				    if (expr.size() != size())
					throw std::logic_error("column_proxy<A>::operator =: mismatched size!");
				    std::copy(expr().cbegin(), expr().cend(),
					      begin());
				    return *this;
				}

  //! 列の要素数すなわち行数を返す.
    size_t			size() const
				{
				    return _a.size();
				}
  //! 列の先頭要素を指す定数反復子を返す.
    const_iterator		cbegin() const
				{
				    return const_iterator(_a.begin(), _col);
				}
  //! 列の先頭要素を指す定数反復子を返す.
    const_iterator		begin() const
				{
				    return cbegin();
				}
  //! 列の先頭要素を指す反復子を返す.
    iterator			begin()
				{
				    return iterator(_a.begin(), _col);
				}
  //! 列の末尾を指す定数反復子を返す.
    const_iterator		cend() const
				{
				    return const_iterator(_a.end(), _col);
				}
  //! 列の末尾を指す定数反復子を返す.
    const_iterator		end() const
				{
				    return cend();
				}
  //! 列の末尾を指す反復子を返す.
    iterator			end()
				{
				    return iterator(_a.end(), _col);
				}
  //! 列の末尾要素を指す定数逆反復子を返す.
    const_reverse_iterator	crbegin() const
				{
				    return const_reverse_iterator(end());
				}
  //! 列の末尾要素を指す定数逆反復子を返す.
    const_reverse_iterator	rbegin() const
				{
				    return crbegin();
				}
  //! 列の末尾要素を指す逆反復子を返す.
    reverse_iterator		rbegin()
				{
				    return reverse_iterator(end());
				}
  //! 列の先頭を指す定数逆反復子を返す.
    const_reverse_iterator	crend() const
				{
				    return const_reverse_iterator(begin());
				}
  //! 列の先頭を指す定数逆反復子を返す.
    const_reverse_iterator	rend() const
				{
				    return crend();
				}
  //! 列の先頭を指す逆反復子を返す.
    reverse_iterator		rend()
				{
				    return reverse_iterator(begin());
				}
  //! 列の定数要素にアクセスする.
  /*!
    \param i	要素を指定するindex
    \return	indexによって指定された定数要素
  */
    const_reference		operator [](size_t i) const
				{
				    return *(cbegin() + i);
				}
  //! 列の要素にアクセスする.
  /*!
    \param i	要素を指定するindex
    \return	indexによって指定された要素
  */
    reference			operator [](size_t i)
				{
				    return *(begin() + i);
				}

  private:
    A&			_a;	//!< 2次元配列への参照
    size_t const	_col;	//!< 列を指定するindex
};

/************************************************************************
*  class column_iterator<A>						*
************************************************************************/
//! 2次元配列の列を指す反復子
/*!
  \param A	2次元配列の型
*/
template <class A>
class column_iterator
    : public boost::iterator_facade<column_iterator<A>,
				    column_proxy<A>,
				    boost::random_access_traversal_tag,
				    column_proxy<A> >
{
  private:
    typedef boost::iterator_facade<column_iterator,
				   column_proxy<A>,
				   boost::random_access_traversal_tag,
				   column_proxy<A> >	super;

  public:
    typedef typename super::value_type			value_type;
    typedef typename super::reference			reference;
    typedef typename super::pointer			pointer;
    typedef typename super::difference_type		difference_type;
    typedef typename super::iterator_category		iterator_category;
    
    friend class	boost::iterator_core_access;
    
  public:
    column_iterator(A& a, size_t col)	:_a(a), _col(col)		{}

    reference		dereference() const
			{
			    return reference(_a, _col);
			}
    bool		equal(const column_iterator& iter) const
			{
			    return _col == iter._col;
			}
    void		increment()
			{
			    ++_col;
			}
    void		decrement()
			{
			    --_col;
			}
    void		advance(difference_type n)
			{
			    _col += n;
			}
    difference_type	distance_to(const column_iterator& iter) const
			{
			    return iter._col - _col;
			}
    
  private:
    A&			_a;
    difference_type	_col;
};

//! 2次元配列の先頭の列を指す定数反復子を返す.
/*!
  \param a	2次元配列
  \return	先頭の列を指す定数反復子
*/
template <class A> column_iterator<const A>
column_cbegin(const A& a)
{
    return column_iterator<const A>(a, 0);
}
    
//! 2次元配列の先頭の列を指す反復子を返す.
/*!
  \param a	2次元配列
  \return	先頭の列を指す反復子
*/
template <class A> column_iterator<A>
column_begin(A& a)
{
    return column_iterator<A>(a, 0);
}
    
//! 2次元配列の末尾の列を指す定数反復子を返す.
/*!
  \param a	2次元配列
  \return	末尾の列を指す定数反復子
*/
template <class A> column_iterator<const A>
column_cend(const A& a)
{
    return column_iterator<const A>(a, a.ncol());
}
    
//! 2次元配列の末尾の列を指す反復子を返す.
/*!
  \param a	2次元配列
  \return	末尾の列を指す反復子
*/
template <class A> column_iterator<A>
column_end(A& a)
{
    return column_iterator<A>(a, a.ncol());
}
    
//! 2次元配列の末尾の列を指す定数逆反復子を返す.
/*!
  \param a	2次元配列
  \return	末尾の列を指す定数逆反復子
*/
template <class A> std::reverse_iterator<column_iterator<const A> >
column_crbegin(const A& a)
{
    return std::reverse_iterator<column_iterator<const A> >(column_cend(a));
}
    
//! 2次元配列の末尾の列を指す逆反復子を返す.
/*!
  \param a	2次元配列
  \return	末尾の列を指す逆反復子
*/
template <class A> std::reverse_iterator<column_iterator<A> >
column_rbegin(A& a)
{
    return std::reverse_iterator<column_iterator<A> >(column_end(a));
}
    
//! 2次元配列の先頭の列を指す定数逆反復子を返す.
/*!
  \param a	2次元配列
  \return	先頭の列を指す定数逆反復子
*/
template <class A> std::reverse_iterator<column_iterator<const A> >
column_crend(const A& a)
{
    return std::reverse_iterator<column_iterator<const A> >(column_cbegin(a));
}
    
//! 2次元配列の先頭の列を指す逆反復子を返す.
/*!
  \param a	2次元配列
  \return	先頭の列を指す逆反復子
*/
template <class A> std::reverse_iterator<column_iterator<A> >
column_rend(A& a)
{
    return std::reverse_iterator<column_iterator<A> >(column_begin(a));
}

/************************************************************************
*  class UnaryOperator<OP, E>						*
************************************************************************/
//! 配列式に対する単項演算子を表すクラス
/*!
  \param OP	各成分に適用される単項演算子の型
  \param E	単項演算子の引数となる式の実体の型
*/
template <class OP, class E>
class UnaryOperator : public Expression<UnaryOperator<OP, E> >,
		      public OpNode
{
  private:
    typedef typename detail::ArgumentType<E>::type	argument_type;
    typedef typename detail::ValueType<E>::type		evalue_type;
    typedef boost::mpl::bool_<
	detail::IsExpression<evalue_type>::value>	evalue_is_expr;

  public:
  //! 評価結果の型
    typedef typename E::result_type			result_type;
  //! 成分の型
    typedef typename OP::result_type			element_type;
  //! 要素の型
    typedef typename boost::mpl::if_<
	evalue_is_expr, UnaryOperator<OP, evalue_type>,
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
    UnaryOperator(const Expression<E>& expr, const OP& op)
	:_e(expr()), _op(op)						{}

  //! 演算結果の先頭要素を指す定数反復子を返す.
    const_iterator	cbegin() const
			{
			    return const_iterator(_e.cbegin(), _op);
			}
  //! 演算結果の先頭要素を指す定数反復子を返す.
    const_iterator	begin() const
			{
			    return cbegin();
			}
  //! 演算結果の末尾を指す定数反復子を返す.
    const_iterator	cend() const
			{
			    return const_iterator(_e.cend(), _op);
			}
  //! 演算結果の末尾を指す定数反復子を返す.
    const_iterator	end() const
			{
			    return cend();
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
    const argument_type	_e;	//!< 引数となる式の実体
    const OP		_op;	//!< 単項演算子
};

//! 与えられた配列式の各要素の符号を反転する.
/*!
  \param expr	配列式
  \return	符号反転演算子ノード
*/
template <class E>
inline UnaryOperator<std::negate<typename E::element_type>, E>
operator -(const Expression<E>& expr)
{
    typedef std::negate<typename E::element_type>	op_type;
    
    return UnaryOperator<op_type, E>(expr, op_type());
}

//! 与えられた配列式の各要素に定数を掛ける.
/*!
  \param expr	配列式
  \param c	乗数
  \return	乗算演算子ノード
*/
template <class E> inline
UnaryOperator<std::binder2nd<std::multiplies<typename E::element_type> >, E>
operator *(const Expression<E>& expr, typename E::element_type c)
{
    typedef std::multiplies<typename E::element_type>	op_type;
    typedef std::binder2nd<op_type>			binder_type;
    
    return UnaryOperator<binder_type, E>(expr, binder_type(op_type(), c));
}

//! 与えられた配列式の各要素に定数を掛ける.
/*!
  \param c	乗数
  \param expr	配列式
  \return	乗算演算子ノード
*/
template <class E> inline
UnaryOperator<std::binder1st<std::multiplies<typename E::element_type> >, E>
operator *(typename E::element_type c, const Expression<E>& expr)
{
    typedef std::multiplies<typename E::element_type>	op_type;
    typedef std::binder1st<op_type>			binder_type;
    
    return UnaryOperator<binder_type, E>(expr, binder_type(op_type(), c));
}

//! 与えられた配列式の各要素を定数で割る.
/*!
  \param expr	配列式
  \param c	除数
  \return	除算演算子ノード
*/
template <class E> inline
UnaryOperator<std::binder2nd<std::divides<typename E::element_type> >, E>
operator /(const Expression<E>& expr, typename E::element_type c)
{
    typedef std::divides<typename E::element_type>	op_type;
    typedef std::binder2nd<op_type>			binder_type;
    
    return UnaryOperator<binder_type, E>(expr, binder_type(op_type(), c));
}

/************************************************************************
*  class BinaryOperator<OP, L, R>					*
************************************************************************/
//! 配列式に対する2項演算子を表すクラス
/*!
  \param OP	各成分に適用される2項演算子の型
  \param L	2項演算子の第1引数となる式の実体の型
  \param R	2項演算子の第2引数となる式の実体の型
*/
template <class OP, class L, class R>
class BinaryOperator : public Expression<BinaryOperator<OP, L, R> >,
		       public OpNode
{
  private:
    typedef typename detail::ArgumentType<L>::type	largument_type;
    typedef typename detail::ArgumentType<R>::type	rargument_type;
    typedef typename detail::ValueType<L>::type		lvalue_type;
    typedef typename detail::ValueType<R>::type		rvalue_type;
    typedef boost::mpl::bool_<
	detail::IsExpression<lvalue_type>::value>	lvalue_is_expr;

  public:
  //! 配列の型
    typedef typename L::result_type			result_type;
  //! 成分の型
    typedef typename OP::result_type			element_type;
  //! 要素の型
    typedef typename boost::mpl::if_<
	lvalue_is_expr,
	BinaryOperator<OP, lvalue_type, rvalue_type>,
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
    BinaryOperator(const Expression<L>& l,
		   const Expression<R>& r, const OP& op)
	:_l(l()), _r(r()), _op(op)
			{
    			    const size_t	d = _l.size();
			    if (d != _r.size())
				throw std::logic_error("BinaryOperator<OP, L, R>::BinaryOperator: mismatched size!");
			}

  //! 演算結果の先頭要素を指す定数反復子を返す.
    const_iterator	cbegin() const
			{
			    return const_iterator(_l.cbegin(),
						  _r.cbegin(), _op);
			}
  //! 演算結果の先頭要素を指す定数反復子を返す.
    const_iterator	begin() const
			{
			    return cbegin();
			}
  //! 演算結果の末尾を指す定数反復子を返す.
    const_iterator	cend() const
			{
			    return const_iterator(_l.cend(), _r.cend(), _op);
			}
  //! 演算結果の末尾を指す定数反復子を返す.
    const_iterator	end() const
			{
			    return cend();
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
    const largument_type	_l;	//!< 第1引数となる式の実体
    const rargument_type	_r;	//!< 第2引数となる式の実体
    const OP			_op;	//!< 2項演算子
};

//! 与えられた2つの配列式の各要素の和をとる.
/*!
  \param l	左辺の配列式
  \param r	右辺の配列式
  \return	加算演算子ノード
*/
template <class L, class R>
inline BinaryOperator<std::plus<typename L::element_type>, L, R>
operator +(const Expression<L>& l, const Expression<R>& r)
{
    typedef std::plus<typename L::element_type>		op_type;

    return BinaryOperator<op_type, L, R>(l, r, op_type());
}

//! 与えられた2つの配列式の各要素の差をとる.
/*!
  \param l	左辺の配列式
  \param r	右辺の配列式
  \return	減算演算子ノード
*/
template <class L, class R>
inline BinaryOperator<std::minus<typename L::element_type>, L, R>
operator -(const Expression<L>& l, const Expression<R>& r)
{
    typedef std::minus<typename L::element_type>	op_type;

    return BinaryOperator<op_type, L, R>(l, r, op_type());
}

/************************************************************************
*  I/O functions							*
************************************************************************/
template <class E> std::ostream&
operator <<(std::ostream& out, const Expression<E>& expr)
{
    typedef typename E::const_iterator	const_iterator;
    for (const_iterator iter = expr().cbegin(); iter != expr().cend(); ++iter)
	out << ' ' << *iter;
    return out << std::endl;
}

}
#endif	/* !__TUExpressionPP_h */
