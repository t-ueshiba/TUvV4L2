/*
 *  $Id: functional.h,v 1.1 2012-08-16 01:30:37 ueshiba Exp $
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
mem_var_ref(S const T::* m)
{
    return const_mem_var_ref_t<S, T>(m);
}

}
#endif

