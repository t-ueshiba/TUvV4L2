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
#include <algorithm>
#include <boost/tuple/tuple.hpp>

namespace TU
{
/************************************************************************
*  class identity							*
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
*  class unarize							*
************************************************************************/
//! 2変数関数を2変数tupleを引数とする1変数関数に直す関数オブジェクト
/*!
  \param OP	2変数関数オブジェクトの型
*/
template <class OP>
class unarize
{
  public:
    typedef typename OP::result_type				result_type;

    unarize(const OP& op=OP())	:_op(op)			{}
    
    template <class TUPLE>
    result_type	operator ()(const TUPLE& t) const
		{
		    return _op(boost::get<0>(t), boost::get<1>(t));
		}

  private:
    const OP	_op;
};

/************************************************************************
*  class seq_transform							*
************************************************************************/
//! 1または2組のデータ列の各要素に対して1または2変数関数を適用して1組のデータ列を出力する関数オブジェクト
/*!
  \param RESULT	変換された要素を出力するデータ列の型
  \param OP	個々の要素に適用される1変数/2変数関数オブジェクトの型
*/
template <class RESULT, class OP>
class seq_transform
{
  public:
    typedef const RESULT&					result_type;

    seq_transform(const OP& op=OP())	:_op(op), _result()	{}

    template <class ARG>
    result_type	operator ()(const ARG& x) const
		{
		    _result.resize(x.size());
		    std::transform(x.begin(), x.end(), _result.begin(), _op);
		    return _result;
		}
    
    template <class ARG0, class ARG1>
    result_type	operator ()(const ARG0& x, const ARG1& y) const
		{
		    _result.resize(x.size());
		    std::transform(x.begin(), x.end(), y.begin(),
				   _result.begin(), _op);
		    return _result;
		}
    
  private:
    const OP		_op;
    mutable RESULT	_result;
};

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

}
#endif
