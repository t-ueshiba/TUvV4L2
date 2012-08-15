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
 *  $Id: utility.h,v 1.23 2012-08-15 07:17:55 ueshiba Exp $
 */
/*!
  \file		utility.h
  \brief	各種ユーティリティクラスおよび関数の定義と実装
*/
#ifndef __TUutility_h
#define __TUutility_h

#include <algorithm>
#include <functional>
#include <iterator>
#include <boost/iterator/transform_iterator.hpp>

/*!
  \namespace	std
  \brief	いくつかの基本的な関数を名前空間stdに追加
*/
namespace std
{
/************************************************************************
*  generic algorithms							*
************************************************************************/
//! 3つの引数のうち最小のものを返す．
template <class T> inline const T&
min(const T& a, const T& b, const T& c)
{
    return min(min(a, b), c);
}

//! 3つの引数のうち最大のものを返す．
template <class T> inline const T&
max(const T& a, const T& b, const T& c)
{
    return max(max(a, b), c);
}
    
//! 4つの引数のうち最小のものを返す．
template <class T> inline const T&
min(const T& a, const T& b, const T& c, const T& d)
{
    return min(min(a, b, c), d);
}

//! 4つの引数のうち最大のものを返す．
template <class T> inline const T&
max(const T& a, const T& b, const T& c, const T& d)
{
    return max(max(a, b, c), d);
}

}

/*!
  \namespace	TU
  \brief	本ライブラリで定義されたクラスおよび関数を納める名前空間
*/
namespace TU
{
/************************************************************************
*  generic algorithms							*
************************************************************************/
//! 条件を満たす要素が前半に，そうでないものが後半になるように並べ替える．
/*!
  \param begin	データ列の先頭を示す反復子
  \param end	データ列の末尾を示す反復子
  \param pred	条件を指定する単項演算子
  \return	条件を満たさない要素の先頭を示す反復子
*/
template <class Iter, class Pred> Iter
pull_if(Iter begin, Iter end, Pred pred)
{
    for (Iter iter = begin; iter != end; ++iter)
	if (pred(*iter))
	    std::iter_swap(begin++, iter);
    return begin;
}

//! 2つの引数の差の絶対値を返す．
template <class T> inline T
diff(const T& a, const T& b)
{
    return (a > b ? a - b : b - a);
}

//! 2次元データに対して3x3ウィンドウを走査して近傍処理を行う．
/*!
  \param begin	最初の行を示す反復子
  \param end	最後の行の次を示す反復子
  \param op	3x3ウィンドウを定義域とする演算子
*/
template <class Iterator, class OP> void
op3x3(Iterator begin, Iterator end, OP op)
{
    typedef typename std::iterator_traits<Iterator>::value_type	Row;
    typedef typename Row::iterator				RowIter;
    typedef typename std::iterator_traits<RowIter>::value_type	RowVal;
    
    Row	buf = *begin;			// 一つ前の行
    --end;
    for (Iterator iter = ++begin; iter != end; )
    {
	RowIter	p    = buf.begin();	// 左上画素
	RowIter	q    = iter->begin();	// 左画素	
	RowVal	val  = *q;		// 左画素における結果
	RowIter	rend = (++iter)->end();
	--rend;
	--rend;				// 左下画素の右端
	for (RowIter r = iter->begin(); r != rend; )  // 左下画素についてループ
	{
	    RowVal tmp = op(p, q, r++);	// 注目画素における結果
	    *p++ = *q;			// 次行の左上画素 = 左画素
	    *q++ = val;			// 左画素における結果を書き込む
	    val	 = tmp;			// 次ウィンドウの左画素における結果を保存
	}
	*p++ = *q;			// 次行の左上画素 = 左画素
	*q++ = val;			// 左画素における結果を書き込む
	*p   = *q;			// 次行の上画素 = 注目画素
    }
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
mem_var_ref(S const T::* m)
{
    return const_mem_var_ref_t<S, T>(m);
}

//! S型のメンバ変数を持つT型オブジェクトへの反復子からそのメンバに直接アクセス(R/W)する反復子を作る．
template <class Iterator, class S, class T>
inline boost::transform_iterator<mem_var_ref_t<S, T>, Iterator>
make_mbr_iterator(Iterator i, S T::* m)
{
    return boost::make_transform_iterator(i, mem_var_ref(m));
}
    
//! S型のメンバ変数を持つT型オブジェクトへの反復子からそのメンバに直接アクセス(R)する反復子を作る．
template <class Iterator, class S, class T>
inline boost::transform_iterator<const_mem_var_ref_t<S, T>, Iterator>
make_const_mbr_iterator(Iterator i, S const T::* m)
{
    return boost::make_transform_iterator(i, mem_var_ref(m));
}

//! std::pairへの反復子からその第1要素に直接アクセス(R/W)する反復子を作る．
/*!
  \param i	ベースとなる反復子
*/
template <class Iterator>
inline boost::transform_iterator<
    mem_var_ref_t<
	typename std::iterator_traits<Iterator>::value_type::first_type,
	typename std::iterator_traits<Iterator>::value_type>,
    Iterator>
make_first_iterator(Iterator i)
{
    return make_mbr_iterator(
		i, &std::iterator_traits<Iterator>::value_type::first);
}
    
//! std::pairへの反復子からその第1要素に直接アクセス(R)する反復子を作る．
/*!
  \param i	ベースとなる反復子
*/
template <class Iterator>
inline boost::transform_iterator<
    const_mem_var_ref_t<
	typename std::iterator_traits<Iterator>::value_type::first_type,
	typename std::iterator_traits<Iterator>::value_type>,
    Iterator>
make_const_first_iterator(Iterator i)
{
    return make_const_mbr_iterator(
		i, &std::iterator_traits<Iterator>::value_type::first);
}
    
//! std::pairへの反復子からその第2要素に直接アクセス(R/W)する反復子を作る．
/*!
  \param i	ベースとなる反復子
*/
template <class Iterator>
inline boost::transform_iterator<
    mem_var_ref_t<typename std::iterator_traits<Iterator>::value_type
							 ::second_type,
		  typename std::iterator_traits<Iterator>::value_type>,
    Iterator>
make_second_iterator(Iterator i)
{
    return make_mbr_iterator(
		i, &std::iterator_traits<Iterator>::value_type::second);
}
    
//! std::pairへの反復子からその第2要素に直接アクセス(R)する反復子を作る．
/*!
  \param i	ベースとなる反復子
*/
template <class Iterator>
inline boost::transform_iterator<
    const_mem_var_ref_t<
	typename std::iterator_traits<Iterator>::value_type::second_type,
	typename std::iterator_traits<Iterator>::value_type>,
    Iterator>
make_const_second_iterator(Iterator i)
{
    return make_const_mbr_iterator(
		i, &std::iterator_traits<Iterator>::value_type::second);
}
    
}

#endif	/* __TUutility_h */
