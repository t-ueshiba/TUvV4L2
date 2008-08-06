/*
 *  平成9-19年（独）産業技術総合研究所 著作権所有
 *  
 *  創作者：植芝俊夫
 *
 *  本プログラムは（独）産業技術総合研究所の職員である植芝俊夫が創作し，
 *  （独）産業技術総合研究所が著作権を所有する秘密情報です．創作者によ
 *  る許可なしに本プログラムを使用，複製，改変，第三者へ開示する等の著
 *  作権を侵害する行為を禁止します．
 *  
 *  このプログラムによって生じるいかなる損害に対しても，著作権所有者お
 *  よび創作者は責任を負いません。
 *
 *  Copyright 1997-2007.
 *  National Institute of Advanced Industrial Science and Technology (AIST)
 *
 *  Creator: Toshio UESHIBA
 *
 *  [AIST Confidential and all rights reserved.]
 *  This program is confidential. Any using, copying, changing or
 *  giving any information concerning with this program to others
 *  without permission by the creator are strictly prohibited.
 *
 *  [No Warranty.]
 *  The copyright holders or the creator are not responsible for any
 *  damages in the use of this program.
 *  
 *  $Id: Array++.h,v 1.20 2008-08-06 07:51:44 ueshiba Exp $
 */
#ifndef __TUArrayPP_h
#define __TUArrayPP_h

#include <iostream>
#include <stdexcept>
#include "TU/types.h"
#ifdef __INTEL_COMPILER
#  include <mmintrin.h>
#endif

namespace TU
{
/************************************************************************
*  class Buf<T>								*
************************************************************************/
//! 可変長バッファクラス
/*!
  単独で使用することはなく，#TU::Array<T, B>または#TU::Array2<T, B>の
  第2テンプレート引数に指定することによって，それらの基底クラスとして使う．
  \param T	要素の型
*/
template <class T>
class Buf
{
  public:
    explicit Buf(u_int siz=0)					;
    Buf(T* p, u_int siz)					;
    Buf(const Buf& b)						;
    Buf&		operator =(const Buf& b)		;
    ~Buf()							;

			operator T*()				;
			operator const T*()		const	;
    size_t		size()				const	;
    u_int		dim()				const	;
    bool		resize(u_int siz)			;
    void		resize(T* p, u_int siz)			;
    static u_int	align(u_int siz)			;
    std::istream&	get(std::istream& in, int m=0)		;
    std::ostream&	put(std::ostream& out)		const	;
    
  private:
    u_int	_size;		// the number of elements in the buffer
    T*		_p;		// pointer to the buffer area
    u_int	_shared	  :  1;	// buffer area is shared with other object
    u_int	_capacity : 31;	// buffer capacity (unit: element, >= _size)
};

//! 指定した要素数のバッファを生成する．
/*!
  \param siz	要素数
*/
template <class T> inline
Buf<T>::Buf(u_int siz)
    :_size(siz), _p(new T[_size]), _shared(0), _capacity(_size)
{
}

//! 外部の領域と要素数を指定してバッファを生成する．
/*!
  \param p	外部領域へのポインタ
  \param siz	要素数
*/
template <class T> inline
Buf<T>::Buf(T* p, u_int siz)
    :_size(siz), _p(p), _shared(1), _capacity(_size)
{
}
    
//! コピーコンストラクタ
template <class T>
Buf<T>::Buf(const Buf<T>& b)
    :_size(b._size), _p(new T[_size]), _shared(0), _capacity(_size)
{
    for (int i = 0; i < _size; ++i)
	_p[i] = b._p[i];
}

//! 標準代入演算子
template <class T> Buf<T>&
Buf<T>::operator =(const Buf<T>& b)
{
    if (this != &b)
    {
	resize(b._size);
	for (int i = 0; i < _size; ++i)
	    _p[i] = b._p[i];
    }
    return *this;
}

//! デストラクタ
template <class T> inline
Buf<T>::~Buf()
{
    if (!_shared)
	delete [] _p;
}
    
//! バッファが使用する内部記憶領域へのポインタを返す．
template <class T> inline
Buf<T>::operator T*()
{
    return _p;
}

//! バッファが使用する内部記憶領域へのポインタを返す．
template <class T> inline
Buf<T>::operator const T*() const
{
    return _p;
}
    
//! バッファの要素数（次元 dim() に等しい）を返す．
template <class T> inline size_t
Buf<T>::size() const
{
    return _size;
}
    
//! バッファの次元（要素数 size() に等しい）を返す．
template <class T> inline u_int
Buf<T>::dim() const
{
    return _size;
}
    
//! バッファの要素数を変更する．
/*!
  ただし，他のオブジェクトと記憶領域を共有しているバッファの要素数を
  変更することはできない．
  \param siz			新しい要素数
  \return			sizが元の要素数よりも大きければtrue，そう
				でなければfalse
  \throw std::logic_error	記憶領域を他のオブジェクトと共有している場合
				に送出
*/
template <class T> bool
Buf<T>::resize(u_int siz)
{
    if (_size == siz)
	return false;
    
    if (_shared)
	throw std::logic_error("Buf<T>::resize: cannot change dimension of shared array!");

    const u_int	old_size = _size;
    if (_capacity < (_size = siz))
    {
	delete [] _p;
	_p = new T[_size];
	_capacity = _size;
    }
    return _size > old_size;
}

//! バッファが内部で使用する記憶領域を指定したものに変更する．
/*!
  \param p	新しい記憶領域へのポインタ
  \param siz	新しい要素数
*/
template <class T> inline void
Buf<T>::resize(T* p, u_int siz)
{
    _size = siz;
    if (!_shared)
	delete [] _p;
    _p = p;
    _shared = 1;
    _capacity = _size;
}

//! 記憶領域をalignするために必要な要素数を返す．
/*!
  必要な記憶容量がバッファによって決まる特定の値の倍数になるよう，与えられた
  要素数を繰り上げる．
  \param siz	要素数
  \return	alignされた要素数
*/
template <class T> inline u_int
Buf<T>::align(u_int siz)
{
    return siz;
}
    
//! 入力ストリームから指定した箇所に配列を読み込む(ASCII)．
/*!
  \param in	入力ストリーム
  \param m	読み込み先の先頭を指定するindex
  \return	inで指定した入力ストリーム
*/
template <class T> std::istream&
Buf<T>::get(std::istream& in, int m)
{
    const u_int	BufSiz = 2048;
    T		tmp[BufSiz];
    int		n = 0;
    
    while (n < BufSiz)
    {
	char	c;
	while (in.get(c))		// skip white spaces other than '\n'
	    if (!isspace(c) || c == '\n')
		break;

	if (!in || c == '\n')
	{
	    resize(m + n);
	    break;
	}

	in.putback(c);
	in >> tmp[n++];
    }
    if (n == BufSiz)
	get(in, m + n);

    for (int i = 0; i < n; ++i)
	_p[m + i] = tmp[i];

    return in;
}

//! 出力ストリームに配列を書き出す(ASCII)．
/*!
  \param out	出力ストリーム
  \return	outで指定した出力ストリーム
*/
template <class T> std::ostream&
Buf<T>::put(std::ostream& out) const
{
    for (int i = 0; i < _size; )
	out << ' ' << _p[i++];
    return out << std::endl;
}

#ifdef __INTEL_COMPILER
/************************************************************************
*  class AlignedBuf<T>							*
************************************************************************/
//! 記憶領域のアドレスが16byteの倍数になるようalignされた可変長バッファクラス
/*!
  単独で使用することはなく，#TU::Array<T, B>または#TU::Array2<T, B>の
  第2テンプレート引数に指定することによって，それらの基底クラスとして使う．
  \param T	要素の型
*/
template <class T>
class AlignedBuf : public Buf<T>
{
  public:
    explicit AlignedBuf(u_int siz=0)				;
    ~AlignedBuf()						;

    using		Buf<T>::operator T*;
    using		Buf<T>::operator const T*;
    using		Buf<T>::size;
    using		Buf<T>::dim;
    
    bool		resize(u_int siz)			;
    static u_int	align(u_int siz)			;
    
  private:
    static T*		memalign(u_int siz)			;

    enum		{ALIGN = 16};
    class LCM		//! sizeof(T)とALIGNの最小公倍数
    {
      public:
	LCM()							;
			operator u_int()		const	{return _val;}
      private:
	u_int		_val;
    };
    static const LCM	_lcm;
};

//! 指定した要素数のバッファを生成する．
/*!
  \param siz	要素数
*/
template <class T> inline
AlignedBuf<T>::AlignedBuf(u_int siz)
    :Buf<T>(memalign(siz), siz)
{
}

//! デストラクタ
template <class T> inline
AlignedBuf<T>::~AlignedBuf()
{
    _mm_free((T*)*this);
}
    
//! バッファの要素数を変更する．
/*!
  \param siz	新しい要素数
  \return	sizが元の要素数と等しければtrue，そうでなければfalse
*/
template <class T> inline bool
AlignedBuf<T>::resize(u_int siz)
{
    if (siz == size())
	return false;

    _mm_free((T*)*this);
    Buf<T>::resize(memalign(siz), siz);
    return true;
}

//! 記憶領域をalignするために必要な要素数を返す．
/*!
  必要な記憶容量が16byteの倍数になるよう，与えられた要素数を繰り上げる．
  \param siz	要素数
  \return	alignされた要素数
*/
template <class T> inline u_int
AlignedBuf<T>::align(u_int siz)
{
  // _lcm * m >= sizeof(T) * siz なる最小の m を求める．
    const int	m = (sizeof(T)*siz + _lcm - 1) / _lcm;
    return (_lcm * m) / sizeof(T);
}

template <class T> inline T*
AlignedBuf<T>::memalign(u_int siz)
{
    void*	p = _mm_malloc(sizeof(T)*siz, ALIGN);
    if (p == 0)
	throw std::runtime_error("AlignedBuf<T>::memalign(): failed to allocate memory!!");
    return (T*)p;
}

template <class T>
AlignedBuf<T>::LCM::LCM()
    :_val(ALIGN * sizeof(T))
{
  // sizeof(T)とALIGNの最大公約数(GCD)を求める．
    u_int	gcd = ALIGN;
    for (u_int m = sizeof(T); m > 0; m -= gcd)
	if (m < gcd)
	    std::swap(m, gcd);

  // sizeof(T)とALIGNの最小公倍数(LCM)
    _val /= gcd;
}

template <class T> const AlignedBuf<T>::LCM	AlignedBuf<T>::_lcm;

#endif	// __INTEL_COMPILER
/************************************************************************
*  class FixedSizedBuf<T, D>						*
************************************************************************/
//! 定数サイズのバッファクラス
/*!
  単独で使用することはなく，#TU::Array<T, B>の第2テンプレート引数に指定する
  ことによって#TU::Array<T, B>の基底クラスとして使う．
  \param T	要素の型
  \param D	バッファ中の要素数
*/
template <class T, size_t D>
class FixedSizedBuf
{
  public:
    explicit FixedSizedBuf(u_int siz=D)			;
    FixedSizedBuf(T* p, u_int siz)				;
    FixedSizedBuf(const FixedSizedBuf& b)			;
    FixedSizedBuf&	operator =(const FixedSizedBuf& b)	;
    
			operator T*()				;
			operator const T*()		const	;
    static size_t	size()					;
    static u_int	dim()					;
    static bool		resize(u_int siz)			;
    void		resize(T* p, u_int siz)			;
    static u_int	align(u_int siz)			;
    std::istream&	get(std::istream& in)			;
    std::ostream&	put(std::ostream& out)		const	;

  private:
    T			_p[D];				// D-sized buffer
};

//! バッファを生成する．
/*!
  \param siz			要素数
  \throw std::logic_error	sizがテンプレートパラメータDに一致しない場合に
				送出
*/
template <class T, size_t D> inline
FixedSizedBuf<T, D>::FixedSizedBuf(u_int siz)
{
    resize(siz);
}

//! 外部の領域と要素数を指定してバッファを生成する（ダミー関数）．
/*!
  実際はバッファが使用する記憶領域は固定されていて変更できないので，
  この関数は常に例外を送出する．
  \throw std::logic_error	この関数が呼ばれたら必ず送出
*/
template <class T, size_t D> inline
FixedSizedBuf<T, D>::FixedSizedBuf(T* p, u_int siz)
{
    throw std::logic_error("FixedSizedBuf<T, D>::FixedSizedBuf(T*, u_int): cannot specify a pointer to external storage!!");
}

//! コピーコンストラクタ
template <class T, size_t D>
FixedSizedBuf<T, D>::FixedSizedBuf(const FixedSizedBuf<T, D>& b)
{
    for (int i = 0; i < D; ++i)
	_p[i] = b._p[i];
}

//! 標準代入演算子
template <class T, size_t D> FixedSizedBuf<T, D>&
FixedSizedBuf<T, D>::operator =(const FixedSizedBuf<T, D>& b)
{
    if (this != &b)
	for (int i = 0; i < D; ++i)
	    _p[i] = b._p[i];
    return *this;
}

//! バッファが使用する内部記憶領域へのポインタを返す．
template <class T, size_t D> inline
FixedSizedBuf<T, D>::operator T*()
{
    return _p;
}

//! バッファが使用する内部記憶領域へのポインタを返す．
template <class T, size_t D> inline
FixedSizedBuf<T, D>::operator const T*() const
{
    return _p;
}
    
//! バッファの要素数（次元 dim() に等しい）を返す．
template <class T, size_t D> inline size_t
FixedSizedBuf<T, D>::size()
{
    return D;
}
    
//! バッファの次元（要素数 size() に等しい）を返す．
template <class T, size_t D> inline u_int
FixedSizedBuf<T, D>::dim()
{
    return D;
}
    
//! バッファの要素数を変更する．
/*!
  実際にはバッファの要素数を変更することはできないので，与えられた要素数が
  このバッファの要素数に等しい場合のみ，通常どおりにこの関数から制御が返る．
  \param siz			新しい要素数
  \return			常にfalse
  \throw std::logic_error	sizがテンプレートパラメータDに一致しない場合に
				送出
*/
template <class T, size_t D> inline bool
FixedSizedBuf<T, D>::resize(u_int siz)
{
    if (siz != D)
	throw std::logic_error("FixedSizedBuf<T, D>::resize(u_int): cannot change buffer size!!");
    return false;
}
    
//! バッファが内部で使用する記憶領域を指定したものに変更する．
/*!
  実際にはバッファの記憶領域を変更することはできないので，与えられたポインタ
  と要素数がこのバッファのそれらに等しい場合のみ，通常どおりにこの関数から制御
  が返る．
  \param p			新しい記憶領域へのポインタ
  \param siz			新しい要素数
  \throw std::logic_error	pがこのバッファの内部記憶領域に一致しないか，
				sizがテンプレートパラメータDに一致しない場合に
				送出
*/
template <class T, size_t D> inline void
FixedSizedBuf<T, D>::resize(T* p, u_int siz)
{
    if (p != _p || siz != D)
	throw std::logic_error("FixedSizedBuf<T, D>::resize(T*, u_int): cannot specify a potiner to external storage!!");
}
    
//! 記憶領域をalignするために必要な要素数を返す．
/*!
  必要な記憶容量がバッファによって決まる特定の値の倍数になるよう，与えられた
  要素数を繰り上げる．
  \param siz	要素数
  \return	alignされた要素数
*/
template <class T, size_t D> inline u_int
FixedSizedBuf<T, D>::align(u_int siz)
{
    return siz;
}
    
//! 入力ストリームから配列を読み込む(ASCII)．
/*!
  \param in	入力ストリーム
  \return	inで指定した入力ストリーム
*/
template <class T, size_t D> std::istream&
FixedSizedBuf<T, D>::get(std::istream& in)
{
    for (int i = 0; i < D; ++i)
	in >> _p[i];
    return in;
}
    
//! 出力ストリームに配列を書き出す(ASCII)．
/*!
  \param out	出力ストリーム
  \return	outで指定した出力ストリーム
*/
template <class T, size_t D> std::ostream&
FixedSizedBuf<T, D>::put(std::ostream& out) const
{
    for (int i = 0; i < D; )
	out << ' ' << _p[i++];
    return out;
}

/************************************************************************
*  class Array<T, B>							*
************************************************************************/
//! B型バッファによって実装されるT型オブジェクトの1次元配列クラス
/*!
  \param T	要素の型
  \param B	バッファ
*/
template <class T, class B=Buf<T> >
class Array : public B
{
  public:
    typedef T			ET;		  //!< 要素の型
    typedef ET			value_type;	  //!< 要素の型
    typedef ptrdiff_t		difference_type;  //!< ポインタ間の差
    typedef value_type&		reference;	  //!< 要素への参照
    typedef const value_type&	const_reference;  //!< 定数要素への参照
    typedef value_type*		pointer;	  //!< 要素へのポインタ
    typedef const value_type*	const_pointer;	  //!< 定数要素へのポインタ
    typedef pointer		iterator;	  //!< 反復子
    typedef const_pointer	const_iterator;	  //!< 定数反復子
    
  public:
    Array()								;
    explicit Array(u_int d)						;
    Array(T* p, u_int d)						;
    template <class B2>
    Array(const Array<T, B2>& a, int i, u_int d)			;
    template <class T2, class B2>
    Array(const Array<T2, B2>& a)					;
    template <class T2, class B2>
    Array&		operator =(const Array<T2, B2>& a)		;
    Array&		operator =(const T& c)				;

    iterator		begin()						;
    const_iterator	begin()					const	;
    iterator		end()						;
    const_iterator	end()					const	;

    using		B::size;
    using		B::dim;

			operator T*()					;
			operator const T*()			const	;
    T&			at(int i)					;
    const T&		at(int i)				const	;
    T&			operator [](int i)				;
    const T&		operator [](int i)			const	;
    Array&		operator *=(double c)				;
    Array&		operator /=(double c)				;
    template <class T2, class B2>
    Array&		operator +=(const Array<T2, B2>& a)		;
    template <class T2, class B2>
    Array&		operator -=(const Array<T2, B2>& a)		;
    template <class T2, class B2>
    bool		operator ==(const Array<T2, B2>& a)	const	;
    template <class T2, class B2>
    bool		operator !=(const Array<T2, B2>& a)	const	;
    std::istream&	restore(std::istream& in)			;
    std::ostream&	save(std::ostream& out)			const	;
    void		check_dim(u_int d)			const	;

  protected:
    static u_int	partial_dim(int i, u_int d, u_int a)		;
};

//! 配列を生成する．
template <class T, class B> inline
Array<T, B>::Array()
    :B()
{
}

//! 指定した要素数の配列を生成する．
/*!
  \param d	配列の要素数
*/
template <class T, class B> inline
Array<T, B>::Array(u_int d)
    :B(d)
{
}

//! 外部の領域と要素数を指定して配列を生成する．
/*!
  \param p	外部領域へのポインタ
  \param d	配列の要素数
*/
template <class T, class B> inline
Array<T, B>::Array(T* p, u_int d)
    :B(p, d)
{
}

//! 記憶領域を元の配列と共有した部分配列を作る．
/*!
  \param a	配列
  \param i	部分配列の第0要素を指定するindex
  \param d	部分配列の次元(要素数)
*/
template <class T, class B> template <class B2> inline
Array<T, B>::Array(const Array<T, B2>& a, int i, u_int d)
    :B((T*)&a[i], partial_dim(i, d, a.dim()))
{
}

//! 他の配列と同一要素を持つ配列を作る（コピーコンストラクタの拡張）．
/*!
  コピーコンストラクタは別個自動的に生成される．
  \param a	コピー元の配列
*/
template <class T, class B> template <class T2, class B2>
Array<T, B>::Array(const Array<T2, B2>& a)
    :B(a.dim())
{
    for (int i = 0; i < dim(); ++i)
	(*this)[i] = a[i];
}
	
//! 他の配列を自分に代入する（標準代入演算子の拡張）．
/*!
  標準代入演算子は別個自動的に生成される．
  \param a	コピー元の配列
  \return	この配列
*/
template <class T, class B> template <class T2, class B2> Array<T, B>&
Array<T, B>::operator =(const Array<T2, B2>& a)
{
    resize(a.dim());
    for (int i = 0; i < dim(); ++i)
	(*this)[i] = a[i];
    return *this;
}
    
//! 全ての要素に同一の値を代入する．
/*!
  \param c	代入する値
  \return	この配列
*/
template <class T, class B> Array<T, B>&
Array<T, B>::operator =(const T& c)
{
    for (int i = 0; i < dim(); )
	(*this)[i++] = c;
    return *this;
}

//! 全ての要素に同一の数値を掛ける．
/*!
  \param c	掛ける数値
  \return	この配列
*/
template <class T, class B> Array<T, B>&
Array<T, B>::operator *=(double c)
{
    for (int i = 0; i < dim(); )
	(*this)[i++] *= c;
    return *this;
}

//! 全ての要素を同一の数値で割る．
/*!
  \param c	割る数値
  \return	この配列
*/
template <class T, class B> Array<T, B>&
Array<T, B>::operator /=(double c)
{
    for (int i = 0; i < dim(); )
	(*this)[i++] /= c;
    return *this;
}

//! 配列の先頭要素を指す反復子を返す．
/*!
  \return	先頭要素を指す反復子
*/
template <class T, class B> inline typename Array<T, B>::iterator
Array<T, B>::begin()
{
    return operator pointer();
}

//! 配列の先頭要素を指す定数反復子を返す．
/*!
  \return	先頭要素を指す定数反復子
*/
template <class T, class B> inline typename Array<T, B>::const_iterator
Array<T, B>::begin() const
{
    return operator const_pointer();
}

//! 配列の末尾を指す反復子を返す．
/*!
  \return	末尾を指す反復子
*/
template <class T, class B> inline typename Array<T, B>::iterator
Array<T, B>::end()
{
    return begin() + size();
}

//! 配列の末尾を指す定数反復子を返す．
/*!
  \return	末尾を指す定数反復子
*/
template <class T, class B> inline typename Array<T, B>::const_iterator
Array<T, B>::end() const
{
    return begin() + size();
}

//! 配列の要素へアクセスする（indexのチェックあり）．
/*!
  \param i			要素を指定するindex
  \return			indexによって指定された要素
  \throw std::out_of_range	0 <= i < dim()でない場合に送出
*/
template <class T, class B> inline T&
Array<T, B>::at(int i)
{
    if (i < 0 || i >= dim())
	throw std::out_of_range("TU::Array<T, B>::at: invalid index!");
    return (*this)[i];
}

//! 配列の内部記憶領域へのポインタを返す．
/*!
  \return	内部記憶領域へのポインタ
*/
template <class T, class B> inline
Array<T, B>::operator T*()
{
    return B::operator T*();
}

//! 配列の内部記憶領域へのポインタを返す．
/*!
  \return	内部記憶領域へのポインタ
*/
template <class T, class B> inline
Array<T, B>::operator const T*() const
{
    return B::operator const T*();
}

//! 配列の要素へアクセスする（indexのチェックあり）．
/*!
  \param i			要素を指定するindex
  \return			indexによって指定された要素
  \throw std::out_of_range	0 <= i < dim()でない場合に送出
*/
template <class T, class B> inline const T&
Array<T, B>::at(int i) const
{
    if (i < 0 || i >= dim())
	throw std::out_of_range("TU::Array<T, B>::at: invalid index!");
    return (*this)[i];
}

//! 配列の要素へアクセスする（indexのチェックなし）．
/*!
  \param i	要素を指定するindex
  \return	indexによって指定された要素
*/
template <class T, class B> inline T&
Array<T, B>::operator [](int i)
{
    return Array::operator pointer()[i];
}

//! 配列の要素へアクセスする（indexのチェックなし）．
/*!
  \param i	要素を指定するindex
  \return	indexによって指定された要素
*/
template <class T, class B> inline const T&
Array<T, B>::operator [](int i) const
{
    return Array::operator const_pointer()[i];
}

//! この配列に他の配列を足す．
/*!
  \param a	足す配列
  \return	この配列
*/
template <class T, class B> template <class T2, class B2> Array<T, B>&
Array<T, B>::operator +=(const Array<T2, B2>& a)
{
    check_dim(a.dim());
    for (int i = 0; i < dim(); ++i)
	(*this)[i] += a[i];
    return *this;
}

//! この配列から他の配列を引く．
/*!
  \param a	引く配列
  \return	この配列
*/
template <class T, class B> template <class T2, class B2> Array<T, B>&
Array<T, B>::operator -=(const Array<T2, B2>& a)
{
    check_dim(a.dim());
    for (int i = 0; i < dim(); ++i)
	(*this)[i] -= a[i];
    return *this;
}

//! 2つの配列を要素毎に比較し，同じであるか調べる．
/*!
  \param a	比較対象となる配列
  \return	全ての要素が同じならばtrueを，そうでなければfalse
*/
template <class T, class B> template <class T2, class B2> bool
Array<T, B>::operator ==(const Array<T2, B2>& a) const
{
    if (dim() != a.dim())
	return false;
    for (int i = 0; i < dim(); ++i)
	if ((*this)[i] != a[i])
	    return false;
    return true;
}

//! 2つの配列を要素毎に比較し，異なるものが存在するか調べる．
/*!
  \param a	比較対象となる配列
  \return	異なる要素が存在すればtrueを，そうでなければfalse
*/
template <class T, class B> template <class T2, class B2> inline bool
Array<T, B>::operator !=(const Array<T2, B2>& a) const
{
    return !(*this == a);
}

//! 入力ストリームから配列を読み込む(binary)．
/*!
  \param in	入力ストリーム
  \return	inで指定した入力ストリーム
*/
template <class T, class B> inline std::istream&
Array<T, B>::restore(std::istream& in)
{
    in.read((char*)(T*)*this, sizeof(T) * dim());
    return in;
}

//! 出力ストリームに配列を書き出す(binary)．
/*!
  \param out	出力ストリーム
  \return	outで指定した出力ストリーム
*/
template <class T, class B> inline std::ostream&
Array<T, B>::save(std::ostream& out) const
{
    out.write((const char*)(const T*)*this, sizeof(T) * dim());
    return out;
}

//! 指定した値がこの配列の要素数に一致するか調べる．
/*!
  \param d			調べたい値
  \throw std::invalid_argument	d != dim()の場合に送出
*/
template <class T, class B> inline void
Array<T, B>::check_dim(u_int d) const
{
    if (d != dim())
	throw std::invalid_argument("Array<T, B>::check_dim: dimension mismatch!");
}

template <class T, class B> inline u_int
Array<T, B>::partial_dim(int i, u_int d, u_int a)
{
    return (i+d <= a ? d : i < a ? a-i : 0);
}

//! 入力ストリームから配列を読み込む(ASCII)．
/*!
  \param in	入力ストリーム
  \param a	配列の読み込み先
  \return	inで指定した入力ストリーム
*/
template <class T, class B> inline std::istream&
operator >>(std::istream& in, Array<T, B>& a)
{
    return a.get(in >> std::ws);
}

//! 出力ストリームへ配列を書き出す(ASCII)．
/*!
  \param out	出力ストリーム
  \param a	書き出す配列
  \return	outで指定した出力ストリーム
*/
template <class T, class B> inline std::ostream&
operator <<(std::ostream& out, const Array<T, B>& a)
{
    return a.put(out);
}

/************************************************************************
*  class Array2<T, B>							*
************************************************************************/
//! 1次元配列Tの1次元配列として定義された2次元配列クラス
/*!
  \param T	1次元配列の型
  \param B	バッファ
*/
template <class T, class B=Buf<typename T::ET> >
class Array2 : public Array<T>
{
  public:
    typedef T			RT;		  //!< 行の型
    typedef RT			row_type;	  //!< 行の型
    typedef typename T::ET	ET;		  //!< 要素の型
    typedef ET			value_type;	  //!< 要素の型
    typedef ptrdiff_t		difference_type;  //!< ポインタ間の差
    typedef value_type&		reference;	  //!< 要素への参照
    typedef const value_type&	const_reference;  //!< 定数要素への参照
    typedef value_type*		pointer;	  //!< 要素へのポインタ
    typedef const value_type*	const_pointer;	  //!< 定数要素へのポインタ

  public:
    explicit Array2(u_int r=0, u_int c=0)				;
    Array2(ET* p, u_int r, u_int c)					;
    template <class B2>
    Array2(const Array2<T, B2>& a, int i, int j, u_int r, u_int c)	;
    Array2(const Array2& a)						;
    template <class T2, class B2>
    Array2(const Array2<T2, B2>& a)					;
    Array2&		operator =(const Array2& a)			;
    template <class T2, class B2>
    Array2&		operator =(const Array2<T2, B2>& a)		;
    Array2&		operator =(const ET& c)				;

    using		Array<T>::begin;
    using		Array<T>::end;
    using		Array<T>::size;
    using		Array<T>::dim;
    
			operator ET*()					;
			operator const ET*()			const	;
    u_int		nrow()					const	;
    u_int		ncol()					const	;
    bool		resize(u_int r, u_int c)			;
    void		resize(ET* p, u_int r, u_int c)			;
    std::istream&	restore(std::istream& in)			;
    std::ostream&	save(std::ostream& out)			const	;
    std::istream&	get(std::istream& in,
			    int i=0, int j=0, int jmax=0)		;

  private:
    void		set_rows()					;
    
    u_int		_ncol;
    B			_buf;
};

//! 行数と列数を指定して2次元配列を生成する．
/*!
  \param r	行数
  \param c	列数
*/
template <class T, class B> inline
Array2<T, B>::Array2(u_int r, u_int c)
    :Array<T>(r), _ncol(c), _buf(nrow()*B::align(ncol()))
{
    set_rows();
}

//! 外部の領域と行数および列数を指定して2次元配列を生成する．
/*!
  \param p	外部領域へのポインタ
  \param r	行数
  \param c	列数
*/
template <class T, class B> inline
Array2<T, B>::Array2(ET* p, u_int r, u_int c)
    :Array<T>(r), _ncol(c), _buf(p, nrow()*B::align(ncol()))
{
    set_rows();
}

//! 記憶領域を元の配列と共有した部分配列を作る
/*!
  \param a	配列
  \param i	部分配列の左上隅要素の行を指定するindex
  \param j	部分配列の左上隅要素の列を指定するindex
  \param r	部分配列の行数
  \param c	部分配列の列数
*/
template <class T, class B> template <class B2>
Array2<T, B>::Array2(const Array2<T, B2>& a, int i, int j, u_int r, u_int c)
    :Array<T>(Array<T>::partial_dim(i, r, a.nrow())),
     _ncol(Array<T>::partial_dim(j, c, a.ncol())),
     _buf((nrow() > 0 && ncol() > 0 ? (ET*)&a[i][j] : 0),
	  nrow()*B::align(ncol()))
{
    for (int ii = 0; ii < nrow(); ++ii)
	(*this)[ii].resize((ET*)&a[i+ii][j], ncol());
}    

//! コピーコンストラクタ
/*!
  \param a	コピー元の配列
*/
template <class T, class B> inline
Array2<T, B>::Array2(const Array2& a)
    :Array<T>(a.nrow()), _ncol(a.ncol()), _buf(nrow()*B::align(ncol()))
{
    set_rows();
    Array<T>::operator =(a);
}    

//! 他の配列と同一要素を持つ配列を作る（コピーコンストラクタの拡張）．
/*!
  コピーコンストラクタを定義しないと自動的に作られてしまうので，
  このコンストラクタがあってもコピーコンストラクタを別個に定義
  しなければならない．
  \param a	コピー元の配列
*/
template <class T, class B> template <class T2, class B2> inline
Array2<T, B>::Array2(const Array2<T2, B2>& a)
    :Array<T>(a.nrow()), _ncol(a.ncol()), _buf(nrow()*B::align(ncol()))
{
    set_rows();
    Array<T>::operator =(a);
}    

//! 標準代入演算子
/*!
  \param a	コピー元の配列
  \return	この配列
*/
template <class T, class B> inline Array2<T, B>&
Array2<T, B>::operator =(const Array2& a)
{
    resize(a.nrow(), a.ncol());
    Array<T>::operator =(a);
    return *this;
}

//! 他の配列を自分に代入する（標準代入演算子の拡張）．
/*!
  標準代入演算子を定義しないと自動的に作られてしまうので，この代入演算子が
  あっても標準代入演算子を別個に定義しなければならない．
  \param a	コピー元の配列
  \return	この配列
*/
template <class T, class B> template <class T2, class B2> inline Array2<T, B>&
Array2<T, B>::operator =(const Array2<T2, B2>& a)
{
    resize(a.nrow(), a.ncol());
    Array<T>::operator =(a);
    return *this;
}

//! 全ての要素に同一の値を代入する．
/*!
  \param c	代入する値
  \return	この配列
*/
template <class T, class B> Array2<T, B>&
Array2<T, B>::operator =(const ET& c)
{
    for (int i = 0; i < nrow(); )
	(*this)[i++] = c;
    return *this;
}

//! 2次元配列の内部記憶領域へのポインタを返す．
/*!
  \return	内部記憶領域へのポインタ
*/
template <class T, class B> inline
Array2<T, B>::operator typename T::ET*()
{
    return _buf.operator ET*();
}

//! 2次元配列の内部記憶領域へのポインタを返す．
/*!
  \return	内部記憶領域へのポインタ
*/
template <class T, class B> inline
Array2<T, B>::operator const typename T::ET*() const
{
    return _buf.operator const ET*();
}

//! 2次元配列の行数を返す．
/*!
  \return	行数
*/
template <class T, class B> inline u_int
Array2<T, B>::nrow() const
{
    return size();
}

//! 2次元配列の列数を返す．
/*!
  \return	列数
*/
template <class T, class B> inline u_int
Array2<T, B>::ncol() const
{
    return _ncol;
}

//! 配列のサイズを変更する．
/*!
  \param r	新しい行数
  \param c	新しい列数
  \return	rが元の行数より大きい又はcが元の列数と異なればtrue，
		そうでなければfalse
*/
template <class T, class B> bool
Array2<T, B>::resize(u_int r, u_int c)
{
    if (!Array<T>::resize(r) && ncol() == c)
	return false;

    _ncol = c;
    _buf.resize(nrow()*B::align(ncol()));
    set_rows();
    return true;
}

//! 配列が内部で使用する記憶領域を指定したものに変更する．
/*!
  \param p	新しい記憶領域へのポインタ
  \param r	新しい行数
  \param c	新しい列数
*/
template <class T, class B> void
Array2<T, B>::resize(ET* p, u_int r, u_int c)
{
    Array<T>::resize(r);
    _ncol = c;
    _buf.resize(p, nrow()*B::align(ncol()));
    set_rows();
}

//! 入力ストリームから配列を読み込む(binary)．
/*!
  \param in	入力ストリーム
  \return	inで指定した入力ストリーム
*/
template <class T, class B> std::istream&
Array2<T, B>::restore(std::istream& in)
{
    for (int i = 0; i < nrow(); )
	(*this)[i++].restore(in);
    return in;
}

//! 出力ストリームに配列を書き出す(binary)．
/*!
  \param out	出力ストリーム
  \return	outで指定した出力ストリーム
*/
template <class T, class B> std::ostream&
Array2<T, B>::save(std::ostream& out) const
{
    for (int i = 0; i < nrow(); )
	(*this)[i++].save(out);
    return out;
}

//! 入力ストリームから指定した箇所に2次元配列を読み込む(ASCII)．
/*!
  \param in	入力ストリーム
  \param i	読み込み先の先頭行を指定するindex
  \param j	読み込み先の先頭列を指定するindex
  \param jmax	これまでに読んできた要素の列番号の最大値
  \return	inで指定した入力ストリーム
*/
template <class T, class B> std::istream&
Array2<T, B>::get(std::istream& in, int i, int j, int jmax)
{
    char	c;

    while (in.get(c))			// Skip white spaces other than '\n'.
	if (!isspace(c) || c == '\n')
	    break;

    if (!in || c == '\n')
    {
	++i;				// Proceed to the next row.
	if (j > jmax)
	    jmax = j;
	j = 0;				// Return to the first column.

	while (in.get(c))		// Skip white spaces other than '\n'.
	    if (!isspace(c) || c == '\n')
		break;

	if (!in || c == '\n')
	{
	    resize(i, jmax);
	    return in;
	}
    }
    in.putback(c);
    ET	val;
    in >> val;
    get(in, i, j + 1, jmax);
    (*this)[i][j] = val;
    return in;
}    

template <class T, class B> void
Array2<T, B>::set_rows()
{
    const u_int	stride = B::align(ncol());
    for (int i = 0; i < nrow(); ++i)
	(*this)[i].resize((ET*)*this + i*stride, ncol());
}
    
//! 入力ストリームから配列を読み込む(ASCII)．
/*!
  \param in	入力ストリーム
  \param a	配列の読み込み先
  \return	inで指定した入力ストリーム
*/
template <class T, class B> inline std::istream&
operator >>(std::istream& in, Array2<T, B>& a)
{
    return a.get(in >> std::ws);
}

}

#endif	/* !__TUArrayPP_h */
