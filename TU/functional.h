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
#include <boost/type_traits.hpp>

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
    typedef second_argument_type			result_type;
    
    T	operator ()(const S& x, T y)		const	{ return y = x; }
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
    typedef second_argument_type			result_type;
    
    T	operator ()(const S& x, T y)		const	{ return y += x; }
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
    typedef second_argument_type			result_type;
    
    T	operator ()(const S& x, T y)		const	{ return y -= x; }
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
    typedef second_argument_type			result_type;
    
    T	operator ()(const S& x, T y)		const	{ return y *= x; }
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
    typedef second_argument_type			result_type;
    
    T	operator ()(const S& x, T y)		const	{ return y /= x; }
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
    typedef second_argument_type			result_type;
    
    T	operator ()(const S& x, T y)		const	{ return y %= x; }
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
    typedef second_argument_type			result_type;
    
    T	operator ()(const S& x, T y)		const	{ return y &= x; }
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
    typedef second_argument_type			result_type;
    
    T	operator ()(const S& x, T y)		const	{ return y |= x; }
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
    typedef second_argument_type			result_type;
    
    T	operator ()(const S& x, T y)		const	{ return y ^= x; }
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
#endif
