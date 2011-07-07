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
 *  $Id: Array++.h,v 1.35 2011-07-07 07:51:04 ueshiba Exp $
 */
#ifndef __TUArrayPP_h
#define __TUArrayPP_h

#include <iterator>
#include <iostream>
#include <stdexcept>
#include "TU/types.h"
#ifdef __INTEL_COMPILER
  #include <mmintrin.h>
#endif

namespace TU
{
/************************************************************************
*  class Buf<T>								*
************************************************************************/
//! 可変長バッファクラス
/*!
  単独で使用することはなく，#TU::Array<T, B>または#TU::Array2<T, B, R>の
  第2テンプレート引数に指定することによって，それらの基底クラスとして使う．
  \param T	要素の型
*/
template <class T>
class Buf
{
  public:
    typedef T			value_type;	//!< 要素の型
    typedef value_type&		reference;	//!< 要素への参照
    typedef const value_type&	const_reference;//!< 定数要素への参照
    typedef value_type*		pointer;	//!< 要素へのポインタ
    typedef const value_type*	const_pointer;	//!< 定数要素へのポインタ
    
  public:
    explicit Buf(u_int siz=0)					;
    Buf(pointer p, u_int siz)					;
    Buf(const Buf& b)						;
    Buf&		operator =(const Buf& b)		;
    ~Buf()							;

    pointer		ptr()					;
    const_pointer	ptr()				const	;
    size_t		size()				const	;
    bool		resize(u_int siz)			;
    void		resize(pointer p, u_int siz)		;
    static u_int	stride(u_int siz)			;
    std::istream&	get(std::istream& in, u_int m=0)	;
    std::ostream&	put(std::ostream& out)		const	;
    
  private:
    u_int	_size;		// the number of elements in the buffer
    pointer	_p;		// pointer to the buffer area
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
Buf<T>::Buf(pointer p, u_int siz)
    :_size(siz), _p(p), _shared(1), _capacity(_size)
{
}
    
//! コピーコンストラクタ
template <class T>
Buf<T>::Buf(const Buf<T>& b)
    :_size(b._size), _p(new T[_size]), _shared(0), _capacity(_size)
{
    for (u_int i = 0; i < _size; ++i)
	_p[i] = b._p[i];
}

//! 標準代入演算子
template <class T> Buf<T>&
Buf<T>::operator =(const Buf<T>& b)
{
    if (this != &b)
    {
	resize(b._size);
	for (u_int i = 0; i < _size; ++i)
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
template <class T> inline typename Buf<T>::pointer
Buf<T>::ptr()
{
    return _p;
}

//! バッファが使用する内部記憶領域へのポインタを返す．
template <class T> inline typename Buf<T>::const_pointer
Buf<T>::ptr() const
{
    return _p;
}
    
//! バッファの要素数を返す．
template <class T> inline size_t
Buf<T>::size() const
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
	throw std::logic_error("Buf<T>::resize: cannot change size of shared buffer!");

    const u_int	old_size = _size;
    _size = siz;
    if (_capacity < _size)
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
Buf<T>::resize(pointer p, u_int siz)
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
Buf<T>::stride(u_int siz)
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
Buf<T>::get(std::istream& in, u_int m)
{
    const u_int	BufSiz = (sizeof(T) < 2048 ? 2048 / sizeof(T) : 1);
    T* const	tmp = new T[BufSiz];
    u_int	n = 0;
    
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

    for (u_int i = 0; i < n; ++i)
	_p[m + i] = tmp[i];

    delete [] tmp;
    
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
    for (u_int i = 0; i < _size; )
	out << ' ' << _p[i++];
    return out;
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
  private:
    typedef Buf<T>				super;

  public:
  //! 要素の型
    typedef typename super::value_type		value_type;
  //! 要素への参照
    typedef typename super::reference		reference;
  //! 定数要素への参照
    typedef typename super::const_reference	const_reference;
  //! 要素へのポインタ
    typedef typename super::pointer		pointer;
  //! 定数要素へのポインタ    
    typedef typename super::const_pointer	const_pointer;

  public:
    explicit AlignedBuf(u_int siz=0)				;
    AlignedBuf(const AlignedBuf& b)				;
    AlignedBuf&		operator =(const AlignedBuf& b)		;
    ~AlignedBuf()						;

    using		super::ptr;
    using		super::size;
    
    bool		resize(u_int siz)			;
    static u_int	stride(u_int siz)			;
    
  private:
    static pointer	memalloc(u_int siz)			;
    static void		memfree(pointer p)			;
    
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
    :super(memalloc(siz), siz)
{
}

//! コピーコンストラクタ
template <class T>
AlignedBuf<T>::AlignedBuf(const AlignedBuf<T>& b)
    :super(memalloc(b.size()), b.size())
{
    super::operator =(b);
}

//! 標準代入演算子
template <class T> AlignedBuf<T>&
AlignedBuf<T>::operator =(const AlignedBuf<T>& b)
{
    resize(b.size());		// Buf<T>::resize(u_int)は使えない．
    super::operator =(b);
    return *this;
}

//! デストラクタ
template <class T> inline
AlignedBuf<T>::~AlignedBuf()
{
    memfree(ptr());
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

    memfree(ptr());
    super::resize(memalloc(siz), siz);
    return true;
}

//! 記憶領域をalignするために必要な要素数を返す．
/*!
  必要な記憶容量が16byteの倍数になるよう，与えられた要素数を繰り上げる．
  \param siz	要素数
  \return	alignされた要素数
*/
template <class T> inline u_int
AlignedBuf<T>::stride(u_int siz)
{
  // _lcm * m >= sizeof(T) * siz なる最小の m を求める．
    const u_int	m = (sizeof(T)*siz + _lcm - 1) / _lcm;
    return (_lcm * m) / sizeof(T);
}

template <class T> inline typename AlignedBuf<T>::pointer
AlignedBuf<T>::memalloc(u_int siz)
{
    void*	p = _mm_malloc(sizeof(T)*siz, ALIGN);
    if (p == 0)
	throw std::runtime_error("AlignedBuf<T>::memalloc(): failed to allocate memory!!");
    return pointer(p);
}

template <class T> inline void
AlignedBuf<T>::memfree(pointer p)
{
    if (p != 0)
	_mm_free(p);
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
    typedef T			value_type;	//!< 要素の型
    typedef value_type&		reference;	//!< 要素への参照
    typedef const value_type&	const_reference;//!< 定数要素への参照
    typedef value_type*		pointer;	//!< 要素へのポインタ
    typedef const value_type*	const_pointer;	//!< 定数要素へのポインタ

  public:
    explicit FixedSizedBuf(u_int siz=D)				;
    FixedSizedBuf(pointer p, u_int siz)				;
    FixedSizedBuf(const FixedSizedBuf& b)			;
    FixedSizedBuf&	operator =(const FixedSizedBuf& b)	;
    
    pointer		ptr()					;
    const_pointer	ptr()				const	;
    static size_t	size()					;
    static bool		resize(u_int siz)			;
    void		resize(pointer p, u_int siz)		;
    static u_int	stride(u_int siz)			;
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
FixedSizedBuf<T, D>::FixedSizedBuf(pointer p, u_int siz)
{
    throw std::logic_error("FixedSizedBuf<T, D>::FixedSizedBuf(pointer, u_int): cannot specify a pointer to external storage!!");
}

//! コピーコンストラクタ
template <class T, size_t D>
FixedSizedBuf<T, D>::FixedSizedBuf(const FixedSizedBuf<T, D>& b)
{
    for (u_int i = 0; i < D; ++i)
	_p[i] = b._p[i];
}

//! 標準代入演算子
template <class T, size_t D> FixedSizedBuf<T, D>&
FixedSizedBuf<T, D>::operator =(const FixedSizedBuf<T, D>& b)
{
    if (this != &b)
	for (u_int i = 0; i < D; ++i)
	    _p[i] = b._p[i];
    return *this;
}

//! バッファが使用する内部記憶領域へのポインタを返す．
template <class T, size_t D> inline typename FixedSizedBuf<T, D>::pointer
FixedSizedBuf<T, D>::ptr()
{
    return _p;
}

//! バッファが使用する内部記憶領域へのポインタを返す．
template <class T, size_t D> inline typename FixedSizedBuf<T, D>::const_pointer
FixedSizedBuf<T, D>::ptr() const
{
    return _p;
}
    
//! バッファの要素数を返す．
template <class T, size_t D> inline size_t
FixedSizedBuf<T, D>::size()
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
FixedSizedBuf<T, D>::resize(pointer p, u_int siz)
{
    if (p != _p || siz != D)
	throw std::logic_error("FixedSizedBuf<T, D>::resize(pointer, u_int): cannot specify a potiner to external storage!!");
}
    
//! 記憶領域をalignするために必要な要素数を返す．
/*!
  必要な記憶容量がバッファによって決まる特定の値の倍数になるよう，与えられた
  要素数を繰り上げる．
  \param siz	要素数
  \return	alignされた要素数
*/
template <class T, size_t D> inline u_int
FixedSizedBuf<T, D>::stride(u_int siz)
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
    for (u_int i = 0; i < D; ++i)
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
    for (u_int i = 0; i < D; ++i)
	out << ' ' << _p[i];
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
  //! バッファの型
    typedef B						buf_type;
  //! 要素の型    
    typedef typename buf_type::value_type		value_type;
  //! 要素への参照
    typedef typename buf_type::reference		reference;
  //! 定数要素への参照
    typedef typename buf_type::const_reference		const_reference;
  //! 要素へのポインタ
    typedef typename buf_type::pointer			pointer;
  //! 定数要素へのポインタ
    typedef typename buf_type::const_pointer		const_pointer;
  //! 反復子
    typedef pointer					iterator;
  //! 定数反復子
    typedef const_pointer				const_iterator;
  //! 逆反復子    
    typedef std::reverse_iterator<iterator>		reverse_iterator;
  //! 定数逆反復子    
    typedef std::reverse_iterator<const_iterator>	const_reverse_iterator;
  //! ポインタ間の差
    typedef ptrdiff_t					difference_type;
    
  public:
    Array()								;
    explicit Array(u_int d)						;
    Array(pointer p, u_int d)						;
    template <class T2, class B2>
    Array(const Array<T2, B2>& a)					;
    template <class B2>
    Array(Array<T, B2>& a, u_int i, u_int d)				;
    template <class T2, class B2>
    Array&		operator =(const Array<T2, B2>& a)		;
    Array&		operator =(const value_type& c)			;

    iterator			begin()					;
    const_iterator		begin()				const	;
    iterator			end()					;
    const_iterator		end()				const	;
    reverse_iterator		rbegin()				;
    const_reverse_iterator	rbegin()			const	;
    reverse_iterator		rend()					;
    const_reverse_iterator	rend()				const	;

    using		buf_type::size;

			operator pointer()				;
  			operator const_pointer()		const	;
    u_int		dim()					const	;
    reference		operator [](int i)				;
    const_reference	operator [](int i)			const	;
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
    static u_int	partial_dim(u_int i, u_int d, u_int a)		;
};

//! 配列を生成する．
template <class T, class B> inline
Array<T, B>::Array()
    :buf_type()
{
}

//! 指定した要素数の配列を生成する．
/*!
  \param d	配列の要素数
*/
template <class T, class B> inline
Array<T, B>::Array(u_int d)
    :buf_type(d)
{
}

//! 外部の領域と要素数を指定して配列を生成する．
/*!
  \param p	外部領域へのポインタ
  \param d	配列の要素数
*/
template <class T, class B> inline
Array<T, B>::Array(pointer p, u_int d)
    :buf_type(p, d)
{
}

//! 他の配列と同一要素を持つ配列を作る（コピーコンストラクタの拡張）．
/*!
  コピーコンストラクタは別個自動的に生成される．
  \param a	コピー元の配列
*/
template <class T, class B> template <class T2, class B2>
Array<T, B>::Array(const Array<T2, B2>& a)
    :buf_type(a.dim())
{
    for (u_int i = 0; i < dim(); ++i)
	(*this)[i] = a[i];
}
	
//! 記憶領域を元の配列と共有した部分配列を作る．
/*!
  \param a	配列
  \param i	部分配列の第0要素を指定するindex
  \param d	部分配列の次元(要素数)
*/
template <class T, class B> template <class B2> inline
Array<T, B>::Array(Array<T, B2>& a, u_int i, u_int d)
    :buf_type(i < a.dim() ? pointer(&a[i]) : pointer((value_type*)0),
	      partial_dim(i, d, a.dim()))
{
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
    for (u_int i = 0; i < dim(); ++i)
	(*this)[i] = a[i];
    return *this;
}
    
//! 全ての要素に同一の値を代入する．
/*!
  \param c	代入する値
  \return	この配列
*/
template <class T, class B> Array<T, B>&
Array<T, B>::operator =(const value_type& c)
{
    for (u_int i = 0; i < dim(); )
	(*this)[i++] = c;
    return *this;
}

//! 配列の先頭要素を指す反復子を返す．
/*!
  \return	先頭要素を指す反復子
*/
template <class T, class B> inline typename Array<T, B>::iterator
Array<T, B>::begin()
{
    return buf_type::ptr();
}

//! 配列の先頭要素を指す定数反復子を返す．
/*!
  \return	先頭要素を指す定数反復子
*/
template <class T, class B> inline typename Array<T, B>::const_iterator
Array<T, B>::begin() const
{
    return buf_type::ptr();
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

//! 配列の末尾要素を指す逆反復子を返す．
/*!
  \return	末尾要素を指す逆反復子
*/
template <class T, class B> inline typename Array<T, B>::reverse_iterator
Array<T, B>::rbegin()
{
    return reverse_iterator(end());
}

//! 配列の末尾要素を指す逆反復子を返す．
/*!
  \return	末尾要素を指す逆反復子
*/
template <class T, class B> inline typename Array<T, B>::const_reverse_iterator
Array<T, B>::rbegin() const
{
    return const_reverse_iterator(end());
}

//! 配列の先頭を指す逆反復子を返す．
/*!
  \return	先頭を指す逆反復子
*/
template <class T, class B> inline typename Array<T, B>::reverse_iterator
Array<T, B>::rend()
{
    return reverse_iterator(begin());
}

//! 配列の先頭を指す逆反復子を返す．
/*!
  \return	先頭を指す逆反復子
*/
template <class T, class B> inline typename Array<T, B>::const_reverse_iterator
Array<T, B>::rend() const
{
    return const_reverse_iterator(begin());
}

//! 配列の内部記憶領域へのポインタを返す．
/*!
  \return	内部記憶領域へのポインタ
*/
template <class T, class B> inline
Array<T, B>::operator pointer()
{
    return buf_type::ptr();
}

//! 配列の内部記憶領域へのポインタを返す．
/*!
  \return	内部記憶領域へのポインタ
*/
template <class T, class B> inline
Array<T, B>::operator const_pointer() const
{
    return buf_type::ptr();
}

//! 配列の次元（要素数）を返す．
template <class T, class B> inline u_int
Array<T, B>::dim() const
{
    return size();
}
    
//! 配列の要素へアクセスする（LIBTUTOOLS_DEBUGを指定するとindexのチェックあり）．
/*!
  \param i			要素を指定するindex
  \return			indexによって指定された要素
  \throw std::out_of_range	0 <= i < dim()でない場合に送出
*/
template <class T, class B> inline typename Array<T, B>::reference
Array<T, B>::operator [](int i)
{
#ifdef LIBTUTOOLS_DEBUG
    if (i < 0 || u_int(i) >= dim())
	throw std::out_of_range("TU::Array<T, B>::operator []: invalid index!");
#endif
    return begin()[i];
}

//! 配列の要素へアクセスする（LIBTUTOOLS_DEBUGを指定するとindexのチェックあり）
/*!
  \param i			要素を指定するindex
  \return			indexによって指定された要素
  \throw std::out_of_range	0 <= i < dim()でない場合に送出
*/
    template <class T, class B> inline typename Array<T, B>::const_reference
Array<T, B>::operator [](int i) const
{
#ifdef LIBTUTOOLS_DEBUG
    if (i < 0 || u_int(i) >= dim())
	throw std::out_of_range("TU::Array<T, B>::operator []: invalid index!");
#endif
    return begin()[i];
}

//! 全ての要素に同一の数値を掛ける．
/*!
  \param c	掛ける数値
  \return	この配列
*/
template <class T, class B> Array<T, B>&
Array<T, B>::operator *=(double c)
{
    for (u_int i = 0; i < dim(); )
	(*this)[i++] *= c;
    return *this;
}

//! 全ての要素を同一の数値で割る．
/*!
  \param c	割る数値
  \return	この配列
*/
template <class T, class B> inline Array<T, B>&
Array<T, B>::operator /=(double c)
{
    return operator *=(1.0 / c);
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
    for (u_int i = 0; i < dim(); ++i)
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
    for (u_int i = 0; i < dim(); ++i)
	(*this)[i] -= a[i];
    return *this;
}

//! 2つの配列を要素毎に比較し，同じであるか調べる．
/*!
  \param a	比較対象となる配列
  \return	全ての要素が同じならばtrue，そうでなければfalse
*/
template <class T, class B> template <class T2, class B2> bool
Array<T, B>::operator ==(const Array<T2, B2>& a) const
{
    if (dim() != a.dim())
	return false;
    for (u_int i = 0; i < dim(); ++i)
	if ((*this)[i] != a[i])
	    return false;
    return true;
}

//! 2つの配列を要素毎に比較し，異なるものが存在するか調べる．
/*!
  \param a	比較対象となる配列
  \return	異なる要素が存在すればtrue，そうでなければfalse
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
    in.read((char*)pointer(*this), sizeof(value_type) * dim());
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
    out.write((const char*)const_pointer(*this), sizeof(T) * dim());
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
Array<T, B>::partial_dim(u_int i, u_int d, u_int a)
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

//! 出力ストリームへ配列を書き出し(ASCII)，さらに改行コードを出力する．
/*!
  \param out	出力ストリーム
  \param a	書き出す配列
  \return	outで指定した出力ストリーム
*/
template <class T, class B> inline std::ostream&
operator <<(std::ostream& out, const Array<T, B>& a)
{
    return a.put(out) << std::endl;
}

/************************************************************************
*  class Array2<T, B, R>						*
************************************************************************/
//! 1次元配列Tの1次元配列として定義された2次元配列クラス
/*!
  \param T	1次元配列の型
  \param B	バッファ
  \param R	行バッファ
*/
template <class T, class B=Buf<typename T::value_type>, class R=Buf<T> >
class Array2 : public Array<T, R>
{
  private:
    typedef Array<T, R>				super;
    
  public:
  //! 行バッファの型
    typedef R					row_buf_type;
  //! 行の型    
    typedef typename super::value_type		row_type;
  //! 行への参照    
    typedef typename super::reference		row_reference;
  //! 定数行への参照    
    typedef typename super::const_reference	row_const_reference;
  //! 行へのポインタ    
    typedef typename super::pointer		row_pointer;
  //! 定数行へのポインタ    
    typedef typename super::const_pointer	row_const_pointer;
  //! 行の反復子    
    typedef typename super::iterator		row_iterator;
  //! 行の定数反復子    
    typedef typename super::const_iterator	row_const_iterator;
  //! 行の逆反復子    
    typedef typename super::reverse_iterator	row_reverse_iterator;
  //! 行の定数逆反復子    
    typedef typename super::const_reverse_iterator
						row_const_reverse_iterator;
  //! バッファの型    
    typedef B					buf_type;
  //! 要素の型    
    typedef typename row_type::value_type	value_type;
  //! 要素への参照    
    typedef typename row_type::reference	reference;
  //! 定数要素への参照    
    typedef typename row_type::const_reference	const_reference;
  //! 要素へのポインタ    
    typedef typename row_type::pointer		pointer;
  //! 定数要素へのポインタ    
    typedef typename row_type::const_pointer	const_pointer;
  //! ポインタ間の差    
    typedef ptrdiff_t				difference_type;

  public:
    Array2()								;
    Array2(u_int r, u_int c)						;
    Array2(pointer p, u_int r, u_int c)					;
    Array2(const Array2& a)						;
    template <class T2, class B2, class R2>
    Array2(const Array2<T2, B2, R2>& a)					;
    template <class B2, class R2>
    Array2(Array2<T, B2, R2>& a, u_int i, u_int j, u_int r, u_int c)	;
    Array2&		operator =(const Array2& a)			;
    template <class T2, class B2, class R2>
    Array2&		operator =(const Array2<T2, B2, R2>& a)		;
    Array2&		operator =(const value_type& c)			;

    using		super::begin;
    using		super::end;
    using		super::rbegin;
    using		super::rend;
    using		super::size;
    using		super::dim;
    
			operator pointer()				;
			operator const_pointer()		const	;
    u_int		nrow()					const	;
    u_int		ncol()					const	;
    u_int		stride()				const	;
    bool		resize(u_int r, u_int c)			;
    void		resize(pointer p, u_int r, u_int c)		;
    std::istream&	restore(std::istream& in)			;
    std::ostream&	save(std::ostream& out)			const	;
    std::istream&	get(std::istream& in,
			    u_int i=0, u_int j=0, u_int jmax=0)		;

  private:
    void		set_rows()					;
    
    u_int		_ncol;
    buf_type		_buf;
};

//! 2次元配列を生成する．
template <class T, class B, class R> inline
Array2<T, B, R>::Array2()
    :super(), _ncol(0), _buf()
{
    if (nrow() > 0)
	_ncol = _buf.size() / nrow();
    set_rows();
}

//! 行数と列数を指定して2次元配列を生成する．
/*!
  \param r	行数
  \param c	列数
*/
template <class T, class B, class R> inline
Array2<T, B, R>::Array2(u_int r, u_int c)
    :super(r), _ncol(c), _buf(nrow()*buf_type::stride(ncol()))
{
    set_rows();
}

//! 外部の領域と行数および列数を指定して2次元配列を生成する．
/*!
  \param p	外部領域へのポインタ
  \param r	行数
  \param c	列数
*/
template <class T, class B, class R> inline
Array2<T, B, R>::Array2(pointer p, u_int r, u_int c)
    :super(r), _ncol(c), _buf(p, nrow()*buf_type::stride(ncol()))
{
    set_rows();
}

//! コピーコンストラクタ
/*!
  \param a	コピー元の配列
*/
template <class T, class B, class R> inline
Array2<T, B, R>::Array2(const Array2& a)
    :super(a.nrow()), _ncol(a.ncol()), _buf(nrow()*buf_type::stride(ncol()))
{
    set_rows();
    super::operator =(a);
}    

//! 他の配列と同一要素を持つ配列を作る（コピーコンストラクタの拡張）．
/*!
  コピーコンストラクタを定義しないと自動的に作られてしまうので，
  このコンストラクタがあってもコピーコンストラクタを別個に定義
  しなければならない．
  \param a	コピー元の配列
*/
template <class T, class B, class R> template <class T2, class B2, class R2>
inline
Array2<T, B, R>::Array2(const Array2<T2, B2, R2>& a)
    :super(a.nrow()), _ncol(a.ncol()), _buf(nrow()*buf_type::stride(ncol()))
{
    set_rows();
    super::operator =(a);
}    

//! 記憶領域を元の配列と共有した部分配列を作る
/*!
  \param a	配列
  \param i	部分配列の左上隅要素の行を指定するindex
  \param j	部分配列の左上隅要素の列を指定するindex
  \param r	部分配列の行数
  \param c	部分配列の列数
*/
template <class T, class B, class R> template <class B2, class R2>
Array2<T, B, R>::Array2(Array2<T, B2, R2>& a,
			u_int i, u_int j, u_int r, u_int c)
    :super(super::partial_dim(i, r, a.nrow())),
     _ncol(super::partial_dim(j, c, a.ncol())),
     _buf((nrow() > 0 && ncol() > 0 ? pointer(&a[i][j])
				    : pointer((value_type*)0)),
	  nrow()*buf_type::stride(ncol()))
{
    for (u_int ii = 0; ii < nrow(); ++ii)
	(*this)[ii].resize(pointer(&a[i+ii][j]), ncol());
}    

//! 標準代入演算子
/*!
  \param a	コピー元の配列
  \return	この配列
*/
template <class T, class B, class R> inline Array2<T, B, R>&
Array2<T, B, R>::operator =(const Array2& a)
{
    resize(a.nrow(), a.ncol());
    super::operator =(a);
    return *this;
}

//! 他の配列を自分に代入する（標準代入演算子の拡張）．
/*!
  標準代入演算子を定義しないと自動的に作られてしまうので，この代入演算子が
  あっても標準代入演算子を別個に定義しなければならない．
  \param a	コピー元の配列
  \return	この配列
*/
template <class T, class B, class R> template <class T2, class B2, class R2>
inline Array2<T, B, R>&
Array2<T, B, R>::operator =(const Array2<T2, B2, R2>& a)
{
    resize(a.nrow(), a.ncol());
    super::operator =(a);
    return *this;
}

//! 全ての要素に同一の値を代入する．
/*!
  \param c	代入する値
  \return	この配列
*/
template <class T, class B, class R> Array2<T, B, R>&
Array2<T, B, R>::operator =(const value_type& c)
{
    for (u_int i = 0; i < nrow(); )
	(*this)[i++] = c;
    return *this;
}

//! 2次元配列の内部記憶領域へのポインタを返す．
/*!
  \return	内部記憶領域へのポインタ
*/
template <class T, class B, class R> inline
Array2<T, B, R>::operator pointer()
{
    return _buf.ptr();
}

//! 2次元配列の内部記憶領域へのポインタを返す．
/*!
  \return	内部記憶領域へのポインタ
*/
template <class T, class B, class R> inline
Array2<T, B, R>::operator const_pointer() const
{
    return _buf.ptr();
}

//! 2次元配列の行数を返す．
/*!
  \return	行数
*/
template <class T, class B, class R> inline u_int
Array2<T, B, R>::nrow() const
{
    return size();
}

//! 2次元配列の列数を返す．
/*!
  \return	列数
*/
template <class T, class B, class R> inline u_int
Array2<T, B, R>::ncol() const
{
    return _ncol;
}

//! 2次元配列の隣接する行の間隔を要素数単位で返す．
/*!
  \return	間隔
*/
template <class T, class B, class R> inline u_int
Array2<T, B, R>::stride() const
{
    return (nrow() > 1 ? const_pointer((*this)[1]) - const_pointer((*this)[0])
		       : _ncol);
}

//! 配列のサイズを変更する．
/*!
  \param r	新しい行数
  \param c	新しい列数
  \return	rが元の行数より大きい又はcが元の列数と異なればtrue，
		そうでなければfalse
*/
template <class T, class B, class R> bool
Array2<T, B, R>::resize(u_int r, u_int c)
{
    if (!super::resize(r) && ncol() == c)
	return false;

    _ncol = c;
    _buf.resize(nrow()*buf_type::stride(ncol()));
    set_rows();
    return true;
}

//! 配列が内部で使用する記憶領域を指定したものに変更する．
/*!
  \param p	新しい記憶領域へのポインタ
  \param r	新しい行数
  \param c	新しい列数
*/
template <class T, class B, class R> void
Array2<T, B, R>::resize(pointer p, u_int r, u_int c)
{
    super::resize(r);
    _ncol = c;
    _buf.resize(p, nrow()*buf_type::stride(ncol()));
    set_rows();
}

//! 入力ストリームから配列を読み込む(binary)．
/*!
  \param in	入力ストリーム
  \return	inで指定した入力ストリーム
*/
template <class T, class B, class R> std::istream&
Array2<T, B, R>::restore(std::istream& in)
{
    for (u_int i = 0; i < nrow(); )
	(*this)[i++].restore(in);
    return in;
}

//! 出力ストリームに配列を書き出す(binary)．
/*!
  \param out	出力ストリーム
  \return	outで指定した出力ストリーム
*/
template <class T, class B, class R> std::ostream&
Array2<T, B, R>::save(std::ostream& out) const
{
    for (u_int i = 0; i < nrow(); )
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
template <class T, class B, class R> std::istream&
Array2<T, B, R>::get(std::istream& in, u_int i, u_int j, u_int jmax)
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
	    if (jmax > 0)
		resize(i, jmax);
	    return in;
	}
    }
    in.putback(c);
    value_type	val;
    in >> val;
    get(in, i, j + 1, jmax);
    (*this)[i][j] = val;
    return in;
}    

template <class T, class B, class R> void
Array2<T, B, R>::set_rows()
{
    const u_int	stride = buf_type::stride(ncol());
    for (u_int i = 0; i < nrow(); ++i)
	(*this)[i].resize(_buf.ptr() + i*stride, ncol());
}
    
//! 入力ストリームから配列を読み込む(ASCII)．
/*!
  \param in	入力ストリーム
  \param a	配列の読み込み先
  \return	inで指定した入力ストリーム
*/
template <class T, class B, class R> inline std::istream&
operator >>(std::istream& in, Array2<T, B, R>& a)
{
    return a.get(in >> std::ws);
}

}

#endif	/* !__TUArrayPP_h */
