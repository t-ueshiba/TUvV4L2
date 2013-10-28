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
#ifndef __TUfunctional_h
#define __TUfunctional_h

#include <functional>
#include <boost/tuple/tuple.hpp>

namespace TU
{
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
    
    result_type	operator ()(argument_type x)	const	{ return x; }
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
    typedef S		first_argument_type;
    typedef T		second_argument_type;
    typedef void	result_type;
    
    result_type	operator ()(first_argument_type  x,
			    second_argument_type y)	const	{ y = x; }
};

/************************************************************************
*  struct assign_plus<S, T>						*
************************************************************************/
//! 引数を加算
/*!
  \param S	引数の型
  \param T	加算元の型
*/
template <class S, class T>
struct assign_plus
{
    typedef S		first_argument_type;
    typedef T		second_argument_type;
    typedef void	result_type;
    
    result_type	operator ()(first_argument_type  x,
			    second_argument_type y)	const	{ y += x; }
};

/************************************************************************
*  struct assign_minus<S, T>						*
************************************************************************/
//! 引数を減算
/*!
  \param S	引数の型
  \param T	減算元の型
*/
template <class S, class T>
struct assign_minus
{
    typedef S		first_argument_type;
    typedef T		second_argument_type;
    typedef void	result_type;
    
    result_type	operator ()(first_argument_type  x,
			    second_argument_type y)	const	{ y -= x; }
};

/************************************************************************
*  struct assign_multiplies<S, T>					*
************************************************************************/
//! 引数を乗算
/*!
  \param S	引数の型
  \param T	乗算元の型
*/
template <class S, class T>
struct assign_multiplies
{
    typedef S		first_argument_type;
    typedef T		second_argument_type;
    typedef void	result_type;
    
    result_type	operator ()(first_argument_type  x,
			    second_argument_type y)	const	{ y *= x; }
};

/************************************************************************
*  struct assign_divides<S, T>						*
************************************************************************/
//! 引数を除算
/*!
  \param S	引数の型
  \param T	除算元の型
*/
template <class S, class T>
struct assign_divides
{
    typedef S		first_argument_type;
    typedef T		second_argument_type;
    typedef void	result_type;
    
    result_type	operator ()(first_argument_type  x,
			    second_argument_type y)	const	{ y /= x; }
};

/************************************************************************
*  struct assign_modulus<S, T>						*
************************************************************************/
//! 引数で割った時の剰余を代入
/*!
  \param S	引数の型
  \param T	剰余をとる元の型
*/
template <class S, class T>
struct assign_modulus
{
    typedef S		first_argument_type;
    typedef T		second_argument_type;
    typedef void	result_type;
    
    result_type	operator ()(first_argument_type  x,
			    second_argument_type y)	const	{ y %= x; }
};

/************************************************************************
*  struct assign_bit_and<S, T>						*
************************************************************************/
//! 引数とのAND
/*!
  \param S	引数の型
  \param T	ANDをとる元の型
*/
template <class S, class T>
struct assign_bit_and
{
    typedef S		first_argument_type;
    typedef T		second_argument_type;
    typedef void	result_type;
    
    result_type	operator ()(first_argument_type  x,
			    second_argument_type y)	const	{ x &= y; }
};

/************************************************************************
*  struct assign_bit_or<S, T>						*
************************************************************************/
//! 引数とのOR
/*!
  \param S	引数の型
  \param T	ORをとる元の型
*/
template <class S, class T>
struct assign_bit_or
{
    typedef S		first_argument_type;
    typedef T		second_argument_type;
    typedef void	result_type;
    
    result_type	operator ()(first_argument_type  x,
			    second_argument_type y)	const	{ y |= x; }
};

/************************************************************************
*  struct assign_bit_xor<S, T>						*
************************************************************************/
//! 引数とのXOR
/*!
  \param S	引数の型
  \param T	XORをとる元の型
*/
template <class S, class T>
struct assign_bit_xor
{
    typedef S		first_argument_type;
    typedef T		second_argument_type;
    typedef void	result_type;
    
    result_type	operator ()(first_argument_type  x,
			    second_argument_type y)	const	{ y ^= x; }
};

/************************************************************************
*  class unarize<FUNC>							*
************************************************************************/
//! 引数をtupleにまとめることによって2/3/4変数関数を1変数関数に変換
template <class FUNC>
class unarize
{
  public:
    typedef typename FUNC::result_type			result_type;

  public:
    unarize(const FUNC& func=FUNC())	:_func(func)	{}

    template <class S, class T> result_type
    operator ()(const boost::tuples::cons<
		     S, boost::tuples::cons<
		       T, boost::tuples::null_type> >& t) const
    {
	return _func(boost::get<0>(t), boost::get<1>(t));
    }

    template <class S, class T, class U> result_type
    operator ()(const boost::tuples::cons<
		     S, boost::tuples::cons<
		       T, boost::tuples::cons<
			 U, boost::tuples::null_type> > >& t) const
    {
	return _func(boost::get<0>(t), boost::get<1>(t), boost::get<2>(t));
    }

    template <class S, class T, class U, class V> result_type
    operator ()(const boost::tuples::cons<
		     S, boost::tuples::cons<
		       T, boost::tuples::cons<
			 U, boost::tuples::cons<
			   V, boost::tuples::null_type> > > >& t) const
    {
	return _func(boost::get<0>(t), boost::get<1>(t),
		     boost::get<2>(t), boost::get<3>(t));
    }

    FUNC const&	functor()			const	{return _func;}

  private:
    FUNC const	_func;
};

template <class FUNC> inline unarize<FUNC>
make_unary_function(const FUNC& func)
{
    return unarize<FUNC>(func);
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
#endif
