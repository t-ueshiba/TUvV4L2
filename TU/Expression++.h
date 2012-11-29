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

#include <iterator>
#include <iostream>
#include <stdexcept>
#include <functional>
#include "TU/Array++.h"

namespace TU
{
/************************************************************************
*  class Expression<E>							*
************************************************************************/
template <class E>
class Expression
{
  public:
    const E&	entity()	const	{return static_cast<const E&>(*this);}
    u_int	size()		const	{return entity().size();}
    u_int	ncol()		const	{return entity().ncol();}
    
    template <class T, class B>
    void	assignTo(Array<T, B>& a) const
		{
		    for (u_int i = 0; i < a.size(); ++i)
			a[i] = entity()[i];
		}
    template <class T, class B>
    void	addTo(Array<T, B>& a) const
		{
		    a.check_size(size());
		    for (u_int i = 0; i < a.size(); ++i)
			a[i] += entity()[i];
		}
    template <class T, class B>
    void	subtractFrom(Array<T, B>& a) const
		{
		    a.check_size(size());
		    for (u_int i = 0; i < a.size(); ++i)
			a[i] -= entity()[i];
		}
    template <class T, class B, class R>
    void	assignTo(Array2<T, B, R>& a) const
		{
		    for (u_int i = 0; i < a.size(); ++i)
		    {
			typename Array2<T, B, R>::reference	row = a[i];

			for (u_int j = 0; j < row.size(); ++j)
			    row[j] = entity().eval(i, j);
		    }
		}
    template <class T, class B, class R>
    void	addTo(Array2<T, B, R>& a) const
		{
		    if ((a.size() != size()) || (a.ncol() != ncol()))
			throw std::logic_error("Expression<E>::operator +=: mismatched size!");
		    for (u_int i = 0; i < a.size(); ++i)
		    {
			typename Array2<T, B, R>::reference	row = a[i];
			
			for (u_int j = 0; j < row.size(); ++j)
			    row[j] += entity().eval(i, j);
		    }
		}
    template <class T, class B, class R>
    void	subtractFrom(Array2<T, B, R>& a) const
		{
		    if ((a.size() != size()) || (a.ncol() != ncol()))
			throw std::logic_error("Expression<E>::operator -=: mismatched size!");
		    for (u_int i = 0; i < a.size(); ++i)
		    {
			typename Array2<T, B, R>::reference	row = a[i];
			
			for (u_int j = 0; j < row.size(); ++j)
			    row[j] -= entity().eval(i, j);
		    }
		}
};

/************************************************************************
*  class UnaryOperator<OP, E>						*
************************************************************************/
template <class OP, class E>
class UnaryOperator : public Expression<UnaryOperator<OP, E> >
{
  public:
    typedef typename OP::result_type		element_type;
    typedef typename E::array_type		array_type;
    
  public:
    UnaryOperator(const OP& op, const Expression<E>& expr)
	:_op(op), _entity(expr.entity())	{}

    u_int		size()		const	{return _entity.size();}
    u_int		ncol()		const	{return _entity.ncol();}
    element_type	operator [](u_int i) const
			{
			    return _op(_entity[i]);
			}
    element_type	eval(u_int i, u_int j) const
			{
			    return _op(_entity.eval(i, j));
			}
			operator array_type() const
			{
			    return array_type(*this);
			}
	
  private:
    const OP	_op;
    const E&	_entity;
};

template <class E>
inline UnaryOperator<std::negate<typename E::element_type>, E>
operator -(const Expression<E>& expr)
{
    typedef std::negate<typename E::element_type>	op_type;
    
    return UnaryOperator<op_type, E>(op_type(), expr);
}

template <class E> inline
UnaryOperator<std::binder2nd<std::multiplies<typename E::element_type> >, E>
operator *(const Expression<E>& expr, typename E::element_type c)
{
    typedef std::multiplies<typename E::element_type>	op_type;
    typedef std::binder2nd<op_type>			binder_type;
    
    return UnaryOperator<binder_type, E>(binder_type(op_type(), c), expr);
}

template <class E> inline
UnaryOperator<std::binder1st<std::multiplies<typename E::element_type> >, E>
operator *(typename E::element_type c, const Expression<E>& expr)
{
    typedef std::multiplies<typename E::element_type>	op_type;
    typedef std::binder1st<op_type>			binder_type;
    
    return UnaryOperator<binder_type, E>(binder_type(op_type(), c), expr);
}

template <class E> inline
UnaryOperator<std::binder2nd<std::divides<typename E::element_type> >, E>
operator /(const Expression<E>& expr, typename E::element_type c)
{
    typedef std::divides<typename E::element_type>	op_type;
    typedef std::binder2nd<op_type>			binder_type;
    
    return UnaryOperator<binder_type, E>(binder_type(op_type(), c), expr);
}

/************************************************************************
*  class BinaryOperator<OP, L, R>					*
************************************************************************/
template <class OP, class L, class R>
class BinaryOperator : public Expression<BinaryOperator<OP, L, R> >
{
  public:
    typedef typename OP::result_type		element_type;
    typedef typename L::array_type		array_type;
    
  public:
    BinaryOperator(const OP& op,
		   const Expression<L>& l, const Expression<R>& r)
	:_op(op), _l(l.entity()), _r(r.entity())	{}

    u_int		size()			const	;
    u_int		ncol()			const	;
    element_type	operator [](u_int i) const
			{
			    return _op(_l[i], _r[i]);
			}
    element_type	eval(u_int i, u_int j) const
			{
			    return _op(_l.eval(i, j), _r.eval(i, j));
			}
			operator array_type() const
			{
			    return array_type(*this);
			}
    
  private:
    const OP	_op;
    const L&	_l;
    const R&	_r;
};

template <class OP, class L, class R> inline u_int
BinaryOperator<OP, L, R>::size() const
{
    const u_int	d = _l.size();
    if (d != _r.size())
	throw std::logic_error("BinaryOperator<OP, L, R>::size: mismatched size!");
    return d;
}

template <class OP, class L, class R> inline u_int
BinaryOperator<OP, L, R>::ncol() const
{
    const u_int	d = _l.ncol();
    if (d != _r.ncol())
	throw std::logic_error("BinaryOperator<OP, L, R>::ncol: mismatched size!");
    return d;
}

template <class L, class R>
inline BinaryOperator<std::plus<typename L::element_type>, L, R>
operator +(const Expression<L>& l, const Expression<R>& r)
{
    typedef std::plus<typename L::element_type>	op_type;

    return BinaryOperator<op_type, L, R>(op_type(), l, r);
}

template <class L, class R>
inline BinaryOperator<std::minus<typename L::element_type>, L, R>
operator -(const Expression<L>& l, const Expression<R>& r)
{
    typedef std::minus<typename L::element_type>	op_type;

    return BinaryOperator<op_type, L, R>(op_type(), l, r);
}


}
#endif	/* !__TUExpressionPP_h */
