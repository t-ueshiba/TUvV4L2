/*
 *  平成9年 電子技術総合研究所 植芝俊夫 著作権所有
 *
 *  著作者による許可なしにこのプログラムの第三者への開示、複製、改変、
 *  使用等その他の著作人格権を侵害する行為を禁止します。
 *  このプログラムによって生じるいかなる損害に対しても、著作者は責任
 *  を負いません。 
 *
 *
 *  Copyright 1996
 *  Toshio UESHIBA, Electrotechnical Laboratory
 *
 *  All rights reserved.
 *  Any changing, copying or giving information about source programs of
 *  any part of this software and/or documentation without permission of the
 *  authors are prohibited.
 *
 *  No Warranty.
 *  Authors are not responsible for any damage in use of this program.
 */

/*
 *  $Id: Array++.h,v 1.2 2002-07-25 02:38:03 ueshiba Exp $
 */
#ifndef __TUArrayPP_h
#define __TUArrayPP_h

#include <iostream>
#include "TU/types.h"

namespace TU
{
/************************************************************************
*  class Array<T>							*
************************************************************************/
/*!
  T型オブジェクトの1次元配列クラス
*/
template <class T>
class Array
{
  public:
    typedef T	ET;				//!< 配列要素の型

  //! 要素数を指定して配列を生成する
    explicit Array(u_int d=0):_d(d), _p(new T[_d]), _shr(0), _siz(_d)	{}
  //! 外部の領域とそのサイズを指定して配列を生成する
    Array(T* p, u_int d) :_d(d), _p(p),	  _shr(1), _siz(_d)	{}
    Array(const Array&, u_int, u_int)	;
    Array(const Array&)			;
    ~Array()					{if (!_shr) delete [] _p;}
    Array&	operator =(const Array&)	;

  //! 配列の要素数を返す
    u_int	dim()				const	{return _d;}
  //! 配列の内部記憶領域へのポインタを返す
    		operator T*()			const	{return _p;}
  //! 配列の要素へアクセスする(indexのチェックあり)
    const ET&	at(int i)			const	;
  //! 配列の要素へアクセスする(indexのチェックあり)
    ET&		at(int i)				;
  //! 配列の要素へアクセスする(indexのチェックなし)
    const ET&	operator [](int i)		const	{return _p[i];}
  //! 配列の要素へアクセスする(indexのチェックなし)
    ET&		operator [](int i)			{return _p[i];}
    Array&	operator  =(double c)			;
    Array&	operator *=(double c)			;
    Array&	operator /=(double c)			;
    Array&	operator +=(const Array& a)		;
    Array&	operator -=(const Array& a)		;
    bool	operator ==(const Array& a)	const	;
  //! 2つの配列を要素毎に比較し，異なるものが存在するかを調べる
    bool	operator !=(const Array& a)	const	{return !(*this == a);}
  //! すべての要素の符号を反転した配列を返す
    Array	operator  -()			const	{Array r(*this);
							 r *= -1; return r;}
    std::istream&	restore(std::istream&)		;
    std::ostream&	save(std::ostream&)	const	;
    bool		resize(u_int)			;
    void		resize(T*, u_int)		;
    void		check_dim(u_int)	const	;
    std::istream&	get(std::istream&, int)		;
    
  private:
    u_int	_d;		// dimension of array
    ET*		_p;		// pointer to buffer area
    u_int	_shr : 1;	// buffer area is shared with other object
    u_int	_siz : 31;	// buffer size (unit: element, >= dim())
	
  //    friend std::istream&	operator >>(std::istream& in, Array& a)	;
};

template <class T> inline std::istream&
operator >>(std::istream& in, Array<T>& a)
{
    return a.get(in >> std::ws, 0);
}

template <class T> std::ostream&
operator <<(std::ostream&, const Array<T>&);

/************************************************************************
*  class Array2<T>							*
************************************************************************/
/*!
  T型配列の1次元配列として定義された2次元配列クラス
*/
template <class T>
class Array2 : public Array<T>
{
  public:
    typedef typename T::ET	ET;		//! T型配列の要素の型
    
  //! 行数と列数を指定して2次元配列を生成する
    explicit Array2(u_int r=0, u_int c=0)
	:Array<T>(r), _cols(c), _ary(nrow()*ncol())	{set_rows();}
  //! 外部の領域とその行数および列数を指定して2次元配列を生成する
    Array2(ET* p, u_int r, u_int c)
	:Array<T>(r), _cols(c), _ary(p, nrow()*ncol())	{set_rows();}
    Array2(const Array2&, u_int, u_int, u_int, u_int)	;
    Array2(const Array2& a)					;
    Array2&	operator =(const Array2& a)			;
    virtual ~Array2()						;

  //! 2次元配列の行数を返す    
    u_int	nrow()			const	{return dim();}
  //! 2次元配列の列数を返す    
    u_int	ncol()			const	{return _cols;}
  //! 2次元配列の内部記憶領域へのポインタを返す
		operator ET*()		const	{return (ET*)_ary;}

  //! 配列の全要素に数値を代入する
    Array2&	operator  =(double c)		{Array<T>::operator  =(c);
						 return *this;}
  //! 配列の全要素を定数倍する
    Array2&	operator *=(double c)		{Array<T>::operator *=(c);
						 return *this;}
  //! 配列の全要素を定数で割る
    Array2&	operator /=(double c)		{Array<T>::operator /=(c);
						 return *this;}
  //! 配列に別の配列を要素毎に足す
    Array2&	operator +=(const Array2& a)	{Array<T>::operator +=(a);
						 return *this;}
  //! 配列から別の配列を要素毎に引く
    Array2&	operator -=(const Array2& a)	{Array<T>::operator -=(a);
						 return *this;}
  //! すべての要素の符号を反転した配列を返す
    Array2	operator  -()		const	{Array2 r(*this);
						 r *= -1; return r;}
    std::istream&	restore(std::istream&)		;
    std::ostream&	save(std::ostream&)	const	;
    void		resize(u_int, u_int)		;
    void		resize(ET*, u_int, u_int)	;
    std::istream&	get(std::istream&, int, int, int)	;
    
  private:
    virtual void	set_rows()		;
    
    u_int		_cols;			// # of columns (width)
    Array<ET>		_ary;

  /*    friend std::istream&	operator >>(std::istream& in, Array2<T>& a)
	{return a.get(in >> std::ws, 0, 0, 0);}*/
};

template <class T> inline std::istream&
operator >>(std::istream& in, Array2<T>& a)
{
    return a.get(in >> std::ws, 0, 0, 0);
}

/************************************************************************
*  numerical operators							*
************************************************************************/
template <class T> inline Array<T>
operator +(const Array<T>& a, const Array<T>& b)
    {Array<T> r(a); r += b; return r;}

template <class T> inline Array<T>
operator -(const Array<T>& a, const Array<T>& b)
    {Array<T> r(a); r -= b; return r;}

template <class T> inline Array<T>
operator *(double c, const Array<T>& a)
    {Array<T> r(a); r *= c; return r;}

template <class T> inline Array<T>
operator *(const Array<T>& a, double c)
    {Array<T> r(a); r *= c; return r;}

template <class T> inline Array<T>
operator /(const Array<T>& a, double c)
    {Array<T> r(a); r /= c; return r;}

template <class T> inline Array2<T>
operator +(const Array2<T>& a, const Array2<T>& b)
    {Array2<T> r(a); r += b; return r;}

template <class T> inline Array2<T>
operator -(const Array2<T>& a, const Array2<T>& b)
    {Array2<T> r(a); r -= b; return r;}

template <class T> inline Array2<T>
operator *(double c, const Array2<T>& a)
    {Array2<T> r(a); r *= c; return r;}

template <class T> inline Array2<T>
operator *(const Array2<T>& a, double c)
    {Array2<T> r(a); r *= c; return r;}

template <class T> inline Array2<T>
operator /(const Array2<T>& a, double c)
    {Array2<T> r(a); r /= c; return r;}
 
}

#endif	/* !__TUArrayPP_h */
