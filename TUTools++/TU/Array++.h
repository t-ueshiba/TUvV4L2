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
  \file		Array++.h
  \brief	配列クラスの定義と実装
*/
#ifndef __TUArrayPP_h
#define __TUArrayPP_h

#include <algorithm>
#include <iostream>
#include <stdexcept>
#include "TU/types.h"
#include "TU/iterator.h"
#include "TU/mmInstructions.h"
//#if __cplusplus > 197711L
//#  define __CXX0X
//#  include <initializer_list>
//#endif

namespace TU
{
/************************************************************************
*  class BufTraits<T, ALIGNED>						*
************************************************************************/
template <class T, bool ALIGNED>
struct BufTraits
{
    typedef const T*					const_iterator;
    typedef T*						iterator;
};
#if defined(MMX)			// MMX が定義されているときは...
template <class T, bool ALIGNED>
struct BufTraits<mm::vec<T>, ALIGNED>	// 要素がvec<T>の配列の反復子を特別版に
{
    typedef mm::load_iterator<const T*, ALIGNED>	const_iterator;
    typedef mm::store_iterator<T*, ALIGNED>		iterator;
};
#endif

/************************************************************************
*  class Buf<T, ALIGNED>						*
************************************************************************/
//! 可変長バッファクラス
/*!
  単独で使用することはなく， TU::Array または TU::Array2 の
  第2テンプレート引数に指定することによって，それらの基底クラスとして使う．
  \param T		要素の型
  \param ALIGNED	バッファのアドレスがalignされていれば true,
			そうでなければ false
*/
template <class T, bool ALIGNED=false>
class Buf : public BufTraits<T, ALIGNED>
{
  private:
    enum	{ ALIGN = (ALIGNED ? 32 : 1) };
    
    typedef BufTraits<T, ALIGNED>			super;

  public:
    typedef T						value_type;
    typedef const value_type*				const_pointer;
    typedef value_type*					pointer;
    typedef typename super::const_iterator		const_iterator;
    typedef typename super::iterator			iterator;
    typedef typename std::iterator_traits<const_iterator>::reference
							const_reference;
    typedef typename std::iterator_traits<iterator>::reference
							reference;
    
  public:
    explicit		Buf(u_int siz=0)			;
			Buf(pointer p, u_int siz)		;
			Buf(const Buf& b)			;
    Buf&		operator =(const Buf& b)		;
			~Buf()					;

    const_pointer	data()				const	;
    pointer		data()					;
    const_iterator	cbegin()			const	;
    const_iterator	begin()				const	;
    iterator		begin()					;
    const_iterator	cend()				const	;
    const_iterator	end()				const	;
    iterator		end()					;
    u_int		size()				const	;
    bool		resize(u_int siz)			;
    void		resize(pointer p, u_int siz)		;
    static u_int	stride(u_int siz)			;
    std::istream&	get(std::istream& in, u_int m=0)	;

  private:
    template <bool _ALIGNED, class=void>
    struct Allocator
    {
	static pointer	alloc(u_int siz)	{ return new value_type[siz]; }
	static void	free(pointer p, u_int)	{ delete [] p; }
    };
#if defined(MMX)
    template <class DUMMY>		// MMX が定義されていれて，かつ
    struct Allocator<true, DUMMY>	// alignment するならば...
    {					// _mm_alloc(), _mm_free() を使用
	static pointer
	alloc(u_int siz)
	{
	    pointer	p = static_cast<pointer>(
				_mm_malloc(sizeof(value_type)*siz, ALIGN));
	    if (p == 0)
		throw std::runtime_error("Buf<T, ALIGNED>::Allocator<true>::alloc(): failed to allocate memory!!");
	    
	    for (pointer q = p; q != p + siz; ++q)
		new(q) value_type();	// 確保した各要素にコンストラクタを適用
	    return p;
	}

	static void
	free(pointer p, u_int siz)
	{
	    if (p != 0)
	    {
		for (pointer q = p; q != p + siz; ++q)
		    q->~value_type();	// 解放する各要素にデストラクタを適用
		_mm_free(p);
	    }
	}
    };
#endif
    typedef Allocator<ALIGNED>			allocator;
    
    template <u_int I, u_int J>
    struct GCD		// 最大公約数を求める template meta-function
    {
	enum	{value = GCD<(I > J ? I % J : I), (I > J ? J : J % I)>::value};
    };
    template <u_int I>
    struct GCD<I, 0>
    {
	enum	{value = I};
    };
    template <u_int J>
    struct GCD<0, J>
    {
	enum	{value = J};
    };

  private:
    u_int	_size;				//!< 要素数
    pointer	_p;				//!< 記憶領域の先頭ポインタ
    u_int	_shared	  : 1;			//!< 記憶領域の共有を示すフラグ
    u_int	_capacity : 8*sizeof(u_int)-1;	//!< 要素数単位の容量: >= _size
};

//! 指定した要素数のバッファを生成する．
/*!
  \param siz	要素数
*/
template <class T, bool ALIGNED> inline
Buf<T, ALIGNED>::Buf(u_int siz)
    :_size(siz), _p(allocator::alloc(_size)), _shared(0), _capacity(_size)
{
}

//! 外部の領域と要素数を指定してバッファを生成する．
/*!
  \param p	外部領域へのポインタ
  \param siz	要素数
*/
template <class T, bool ALIGNED> inline
Buf<T, ALIGNED>::Buf(pointer p, u_int siz)
    :_size(siz), _p(p), _shared(1), _capacity(_size)
{
}
    
//! コピーコンストラクタ
template <class T, bool ALIGNED>
Buf<T, ALIGNED>::Buf(const Buf& b)
    :_size(b._size), _p(allocator::alloc(_size)), _shared(0), _capacity(_size)
{
    std::copy(b.cbegin(), b.cend(), begin());
}

//! 標準代入演算子
template <class T, bool ALIGNED> Buf<T, ALIGNED>&
Buf<T, ALIGNED>::operator =(const Buf& b)
{
    if (this != &b)
    {
	resize(b._size);
	std::copy(b.cbegin(), b.cend(), begin());
    }
    return *this;
}

//! デストラクタ
template <class T, bool ALIGNED> inline
Buf<T, ALIGNED>::~Buf()
{
    if (!_shared)
	allocator::free(_p, _size);
}
    
//! バッファが使用する内部記憶領域への定数ポインタを返す．
template <class T, bool ALIGNED> inline typename Buf<T, ALIGNED>::const_pointer
Buf<T, ALIGNED>::data() const
{
    return _p;
}
    
//! バッファが使用する内部記憶領域へのポインタを返す．
template <class T, bool ALIGNED> inline typename Buf<T, ALIGNED>::pointer
Buf<T, ALIGNED>::data()
{
    return _p;
}

//! バッファの先頭要素を指す定数反復子を返す．
template <class T, bool ALIGNED> inline typename Buf<T, ALIGNED>::const_iterator
Buf<T, ALIGNED>::cbegin() const
{
    return data();
}

//! バッファの先頭要素を指す定数反復子を返す．
template <class T, bool ALIGNED> inline typename Buf<T, ALIGNED>::const_iterator
Buf<T, ALIGNED>::begin() const
{
    return cbegin();
}

//! バッファの先頭要素を指す反復子を返す．
template <class T, bool ALIGNED> inline typename Buf<T, ALIGNED>::iterator
Buf<T, ALIGNED>::begin()
{
    return data();
}

//! バッファの末尾を指す定数反復子を返す．
template <class T, bool ALIGNED> inline typename Buf<T, ALIGNED>::const_iterator
Buf<T, ALIGNED>::cend() const
{
    return cbegin() + size();
}

//! バッファの末尾を指す定数反復子を返す．
template <class T, bool ALIGNED> inline typename Buf<T, ALIGNED>::const_iterator
Buf<T, ALIGNED>::end() const
{
    return cend();
}

//! バッファの末尾を指す反復子を返す．
template <class T, bool ALIGNED> inline typename Buf<T, ALIGNED>::iterator
Buf<T, ALIGNED>::end()
{
    return begin() + size();
}

//! バッファの要素数を返す．
template <class T, bool ALIGNED> inline u_int
Buf<T, ALIGNED>::size() const
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
template <class T, bool ALIGNED> bool
Buf<T, ALIGNED>::resize(u_int siz)
{
    if (_size == siz)
	return false;
    
    if (_shared)
	throw std::logic_error("Buf<T, ALIGNED>::resize: cannot change size of shared buffer!");

    const u_int	old_size = _size;
    _size = siz;
    if (_capacity < _size)
    {
	allocator::free(_p, old_size);
	_p = allocator::alloc(_size);
	_capacity = _size;
    }
    return _size > old_size;
}

//! バッファが内部で使用する記憶領域を指定したものに変更する．
/*!
  \param p	新しい記憶領域へのポインタ
  \param siz	新しい要素数
*/
template <class T, bool ALIGNED> inline void
Buf<T, ALIGNED>::resize(pointer p, u_int siz)
{
    if (!_shared)
	allocator::free(_p, _size);
    _size     = siz;
    _p	      = p;
    _shared   = 1;
    _capacity = _size;
}

//! 記憶領域をalignするために必要な要素数を返す．
/*!
  必要な記憶容量がバッファによって決まる特定の値の倍数になるよう，与えられた
  要素数を繰り上げる．
  \param siz	要素数
  \return	alignされた要素数
*/
template <class T, bool ALIGNED> inline u_int
Buf<T, ALIGNED>::stride(u_int siz)
{
#if defined(MMX)
    if (ALIGNED)
    {
	const u_int	LCM = ALIGN * sizeof(T) / GCD<sizeof(T), ALIGN>::value;
      // LCM * m >= sizeof(T) * siz なる最小の m を求める．
	const u_int	m = (sizeof(T)*siz + LCM - 1) / LCM;
	return (LCM * m) / sizeof(T);
    }
    else
#endif
	return siz;
}
    
//! 入力ストリームから指定した箇所に配列を読み込む(ASCII)．
/*!
  \param in	入力ストリーム
  \param m	読み込み先の先頭を指定するindex
  \return	inで指定した入力ストリーム
*/
template <class T, bool ALIGNED> std::istream&
Buf<T, ALIGNED>::get(std::istream& in, u_int m)
{
    const u_int		BufSiz = (sizeof(value_type) < 2048 ?
				  2048 / sizeof(value_type) : 1);
    pointer const	tmp = new value_type[BufSiz];
    u_int		n = 0;
    
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
    
/************************************************************************
*  class FixedSizedBuf<T, D, ALIGNED>					*
************************************************************************/
//! 定数サイズのバッファクラス
/*!
  単独で使用することはなく， TU::Array の第2テンプレート引数に指定する
  ことによって TU::Array の基底クラスとして使う．
  \param T		要素の型
  \param D		バッファ中の要素数
  \param ALIGNED	バッファのアドレスがalignされていれば true,
			そうでなければ false
*/
template <class T, u_int D, bool ALIGNED=false>
class FixedSizedBuf : public BufTraits<T, ALIGNED>
{
  private:
    typedef BufTraits<T, ALIGNED>			super;

  public:
    typedef T						value_type;
    typedef const value_type*				const_pointer;
    typedef value_type*					pointer;
    typedef typename super::const_iterator		const_iterator;
    typedef typename super::iterator			iterator;
    typedef typename std::iterator_traits<const_iterator>::reference
							const_reference;
    typedef typename std::iterator_traits<iterator>::reference
							reference;
    
  public:
    explicit		FixedSizedBuf(u_int siz=D)		;
			FixedSizedBuf(pointer, u_int)		;
			FixedSizedBuf(const FixedSizedBuf& b)	;
    FixedSizedBuf&	operator =(const FixedSizedBuf& b)	;
    
    const_pointer	data()				const	;
    pointer		data()					;
    const_iterator	cbegin()			const	;
    const_iterator	begin()				const	;
    iterator		begin()					;
    const_iterator	cend()				const	;
    const_iterator	end()				const	;
    iterator		end()					;
    static u_int	size()					;
    static bool		resize(u_int siz)			;
    void		resize(pointer p, u_int siz)		;
    static u_int	stride(u_int siz)			;
    std::istream&	get(std::istream& in)			;

  private:
    template <u_int N, class=void>
    struct copy
    {
	void	exec(const_iterator src, iterator dst) const
		{
		    *dst = *src;
		    copy<N+1>().exec(++src, ++dst);
		}
    };
    template <class DUMMY>
    struct copy<D, DUMMY>
    {
	void	exec(const_iterator, iterator)		const	{}
    };
    
    template <bool _ALIGNED, class=void>
    struct Buffer		{ value_type p[D]; };
#if defined(MMX)
    template <class DUMMY>
    struct Buffer<true, DUMMY>	{ __declspec(align(32)) value_type p[D]; };
#endif

  private:
    Buffer<ALIGNED>	_buf;				// D-sized buffer
};

//! バッファを生成する．
/*!
  \param siz			要素数
  \throw std::logic_error	sizがテンプレートパラメータDに一致しない場合に
				送出
*/
template <class T, u_int D, bool ALIGNED> inline
FixedSizedBuf<T, D, ALIGNED>::FixedSizedBuf(u_int siz)
{
    resize(siz);
}

//! 外部の領域と要素数を指定してバッファを生成する（ダミー関数）．
/*!
  実際はバッファが使用する記憶領域は固定されていて変更できないので，
  この関数は常に例外を送出する．
  \throw std::logic_error	この関数が呼ばれたら必ず送出
*/
template <class T, u_int D, bool ALIGNED> inline
FixedSizedBuf<T, D, ALIGNED>::FixedSizedBuf(pointer, u_int)
{
    throw std::logic_error("FixedSizedBuf<T, D, ALIGNED>::FixedSizedBuf(pointer, u_int): cannot specify a pointer to external storage!!");
}

//! コピーコンストラクタ
template <class T, u_int D, bool ALIGNED>
FixedSizedBuf<T, D, ALIGNED>::FixedSizedBuf(const FixedSizedBuf& b)
{
    copy<0>().exec(b.cbegin(), begin());
}

//! 標準代入演算子
template <class T, u_int D, bool ALIGNED> FixedSizedBuf<T, D, ALIGNED>&
FixedSizedBuf<T, D, ALIGNED>::operator =(const FixedSizedBuf& b)
{
    if (this != &b)
	copy<0>().exec(b.cbegin(), begin());
    return *this;
}

//! バッファが使用する内部記憶領域への定数ポインタを返す．
template <class T, u_int D, bool ALIGNED>
inline typename FixedSizedBuf<T, D, ALIGNED>::const_pointer
FixedSizedBuf<T, D, ALIGNED>::data() const
{
    return _buf.p;
}
    
//! バッファが使用する内部記憶領域へのポインタを返す．
template <class T, u_int D, bool ALIGNED>
inline typename FixedSizedBuf<T, D, ALIGNED>::pointer
FixedSizedBuf<T, D, ALIGNED>::data()
{
    return _buf.p;
}

//! バッファの先頭要素を指す定数反復子を返す．
template <class T, u_int D, bool ALIGNED>
inline typename FixedSizedBuf<T, D, ALIGNED>::const_iterator
FixedSizedBuf<T, D, ALIGNED>::cbegin() const
{
    return data();
}

//! バッファの先頭要素を指す定数反復子を返す．
template <class T, u_int D, bool ALIGNED>
inline typename FixedSizedBuf<T, D, ALIGNED>::const_iterator
FixedSizedBuf<T, D, ALIGNED>::begin() const
{
    return cbegin();
}

//! バッファの先頭要素を指す反復子を返す．
template <class T, u_int D, bool ALIGNED>
inline typename FixedSizedBuf<T, D, ALIGNED>::iterator
FixedSizedBuf<T, D, ALIGNED>::begin()
{
    return data();
}

//! バッファの末尾を指す定数反復子を返す．
template <class T, u_int D, bool ALIGNED>
inline typename FixedSizedBuf<T, D, ALIGNED>::const_iterator
FixedSizedBuf<T, D, ALIGNED>::cend() const
{
    return cbegin() + size();
}

//! バッファの末尾を指す定数反復子を返す．
template <class T, u_int D, bool ALIGNED>
inline typename FixedSizedBuf<T, D, ALIGNED>::const_iterator
FixedSizedBuf<T, D, ALIGNED>::end() const
{
    return cend();
}

//! バッファの末尾を指す反復子を返す．
template <class T, u_int D, bool ALIGNED>
inline typename FixedSizedBuf<T, D, ALIGNED>::iterator
FixedSizedBuf<T, D, ALIGNED>::end()
{
    return begin() + size();
}

//! バッファの要素数を返す．
template <class T, u_int D, bool ALIGNED> inline u_int
FixedSizedBuf<T, D, ALIGNED>::size()
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
template <class T, u_int D, bool ALIGNED> inline bool
FixedSizedBuf<T, D, ALIGNED>::resize(u_int siz)
{
    if (siz != D)
	throw std::logic_error("FixedSizedBuf<T, D, ALIGNED>::resize(u_int): cannot change buffer size!!");
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
template <class T, u_int D, bool ALIGNED> inline void
FixedSizedBuf<T, D, ALIGNED>::resize(pointer p, u_int siz)
{
    if (p != _buf.p || siz != D)
	throw std::logic_error("FixedSizedBuf<T, D, ALIGNED>::resize(pointer, u_int): cannot specify a potiner to external storage!!");
}
    
//! 記憶領域をalignするために必要な要素数を返す．
/*!
  必要な記憶容量がバッファによって決まる特定の値の倍数になるよう，与えられた
  要素数を繰り上げる．
  \param siz	要素数
  \return	alignされた要素数
*/
template <class T, u_int D, bool ALIGNED> inline u_int
FixedSizedBuf<T, D, ALIGNED>::stride(u_int siz)
{
    return siz;
}
    
//! 入力ストリームから配列を読み込む(ASCII)．
/*!
  \param in	入力ストリーム
  \return	inで指定した入力ストリーム
*/
template <class T, u_int D, bool ALIGNED> std::istream&
FixedSizedBuf<T, D, ALIGNED>::get(std::istream& in)
{
    for (u_int i = 0; i < D; ++i)
	in >> _buf.p[i];
    return in;
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
  private:
    typedef B					super;

    template <class _A>
    class IsArray
    {
      private:
	typedef char				Small;
	struct Big				{ char dummy[2]; };

	template <class _T, class _B>
	static Small	test(Array<_T, _B>)	;
	static Big	test(...)		;
	static _A	makeA()			;
    
      public:
	enum		{ value = (sizeof(test(makeA())) == sizeof(Small)) };
    };
    
    template <class _T, bool>
    struct ElementType
    {
	typedef _T				type;
    };
    template <class _T>
    struct ElementType<_T, true>
    {
	typedef typename _T::element_type	type;
    };

  public:
  //! 成分の型
    typedef typename ElementType<T, IsArray<T>::value>::type
							element_type;
  //! 要素の型    
    typedef typename super::value_type			value_type;
  //! 定数要素へのポインタ
    typedef typename super::const_pointer		const_pointer;
  //! 要素へのポインタ
    typedef typename super::pointer			pointer;
  //! 定数反復子
    typedef typename super::const_iterator		const_iterator;
  //! 反復子
    typedef typename super::iterator			iterator;
  //! 定数逆反復子    
    typedef std::reverse_iterator<const_iterator>	const_reverse_iterator;
  //! 逆反復子    
    typedef std::reverse_iterator<iterator>		reverse_iterator;
  //! 定数要素への参照
    typedef typename super::const_reference		const_reference;
  //! 要素への参照
    typedef typename super::reference			reference;
  //! ポインタ間の差
    typedef std::ptrdiff_t				difference_type;
    
  public:
    Array()								;
    explicit Array(u_int d)						;
    Array(pointer p, u_int d)						;
    template <class B2>
    Array(Array<T, B2>& a, u_int i, u_int d)				;
    template <class T2, class B2>
    Array(const Array<T2, B2>& a)					;
    template <class T2, class B2>
    Array&	operator =(const Array<T2, B2>& a)			;
#ifdef __CXX0X
    Array(std::initializer_list<value_type> args)			;
    Array&	operator =(std::initializer_list<value_type> args)	;
#endif
    Array&	operator =(const element_type& c)			;

    using	super::data;
    using	super::cbegin;
    using	super::begin;
    using	super::cend;
    using	super::end;
    using	super::size;
    using	super::resize;

    u_int			dim()				const	;
    const_reverse_iterator	crbegin()			const	;
    const_reverse_iterator	rbegin()			const	;
    reverse_iterator		rbegin()				;
    const_reverse_iterator	crend()				const	;
    const_reverse_iterator	rend()				const	;
    reverse_iterator		rend()					;

    const_reference	operator [](u_int i)			const	;
    reference		operator [](u_int i)				;
    const element_type&	eval(u_int i, u_int j)			const	;
    Array&		operator *=(element_type c)			;
    template <class T2>
    Array&		operator /=(T2 c)				;
    template <class T2, class B2>
    Array&		operator +=(const Array<T2, B2>& a)		;
    template <class T2, class B2>
    Array&		operator -=(const Array<T2, B2>& a)		;
    template <class T2, class B2>
    bool		operator ==(const Array<T2, B2>& a)	const	;
    template <class T2, class B2>
    bool		operator !=(const Array<T2, B2>& a)	const	;
    std::istream&	get(std::istream& in)				;
    std::ostream&	put(std::ostream& out)			const	;
    std::istream&	restore(std::istream& in)			;
    std::ostream&	save(std::ostream& out)			const	;
    void		check_size(u_int d)			const	;

  protected:
    static u_int	partial_size(u_int i, u_int d, u_int a)	;
};

//! 配列を生成する．
template <class T, class B> inline
Array<T, B>::Array()
    :super()
{
}

//! 指定した要素数の配列を生成する．
/*!
  \param d	配列の要素数
*/
template <class T, class B> inline
Array<T, B>::Array(u_int d)
    :super(d)
{
}

//! 外部の領域と要素数を指定して配列を生成する．
/*!
  \param p	外部領域へのポインタ
  \param d	配列の要素数
*/
template <class T, class B> inline
Array<T, B>::Array(pointer p, u_int d)
    :super(p, d)
{
}

//! 記憶領域を元の配列と共有した部分配列を作る．
/*!
  \param a	配列
  \param i	部分配列の第0要素を指定するindex
  \param d	部分配列の次元(要素数)
*/
template <class T, class B> template <class B2> inline
Array<T, B>::Array(Array<T, B2>& a, u_int i, u_int d)
    :super(i < a.size() ? pointer(&a[i]) : pointer((element_type*)0),
	   partial_size(i, d, a.size()))
{
}

//! 他の配列と同一要素を持つ配列を作る（コピーコンストラクタの拡張）．
/*!
  コピーコンストラクタは別個自動的に生成される．
  \param a	コピー元の配列
*/
template <class T, class B> template <class T2, class B2>
Array<T, B>::Array(const Array<T2, B2>& a)
    :super(a.size())
{
    std::copy(a.cbegin(), a.cend(), begin());
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
    resize(a.size());
    std::copy(a.cbegin(), a.cend(), begin());
    return *this;
}

#ifdef __CXX0X
template <class T, class B>
Array<T, B>::Array(std::initializer_list<value_type> args)
    :super(args.size())
{
    u_int	i = 0;
    for (auto val : args)
	(*this)[i++] = val;
}

template <class T, class B> Array<T, B>&
Array<T, B>::operator =(std::initializer_list<value_type> args)
{
    resize(args.size());
    u_int	i = 0;
    for (auto val : args)
	(*this)[i++] = val;
    return *this;
}
#endif
    
//! 全ての要素に同一の値を代入する．
/*!
  \param c	代入する値
  \return	この配列
*/
template <class T, class B> Array<T, B>&
Array<T, B>::operator =(const element_type& c)
{
    std::fill(begin(), end(), c);
    return *this;
}

//! 要素数を返す．
template <class T, class B> inline u_int
Array<T, B>::dim() const
{
    return size();
}
    
//! 配列の末尾要素を指す定数逆反復子を返す．
/*!
  \return	末尾要素を指す定数逆反復子
*/
template <class T, class B> inline typename Array<T, B>::const_reverse_iterator
Array<T, B>::crbegin() const
{
    return const_reverse_iterator(cend());
}

//! 配列の末尾要素を指す定数逆反復子を返す．
/*!
  \return	末尾要素を指す定数逆反復子
*/
template <class T, class B> inline typename Array<T, B>::const_reverse_iterator
Array<T, B>::rbegin() const
{
    return crbegin();
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

//! 配列の先頭を指す定数逆反復子を返す．
/*!
  \return	先頭を指す定数逆反復子
*/
template <class T, class B> inline typename Array<T, B>::const_reverse_iterator
Array<T, B>::crend() const
{
    return const_reverse_iterator(cbegin());
}

//! 配列の先頭を指す定数逆反復子を返す．
/*!
  \return	先頭を指す定数逆反復子
*/
template <class T, class B> inline typename Array<T, B>::const_reverse_iterator
Array<T, B>::rend() const
{
    return crend();
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

//! 配列の要素へアクセスする（LIBTUTOOLS_DEBUGを指定するとindexのチェックあり）
/*!
  \param i			要素を指定するindex
  \return			indexによって指定された要素
  \throw std::out_of_range	0 <= i < size()でない場合に送出
*/
template <class T, class B> inline typename Array<T, B>::const_reference
Array<T, B>::operator [](u_int i) const
{
#ifdef LIBTUTOOLS_DEBUG
    if (i < 0 || u_int(i) >= size())
	throw std::out_of_range("TU::Array<T, B>::operator []: invalid index!");
#endif
    return *(cbegin() + i);
}

//! 配列の要素へアクセスする（LIBTUTOOLS_DEBUGを指定するとindexのチェックあり）
/*!
  \param i			要素を指定するindex
  \return			indexによって指定された要素
  \throw std::out_of_range	0 <= i < size()でない場合に送出
*/
template <class T, class B> inline typename Array<T, B>::reference
Array<T, B>::operator [](u_int i)
{
#ifdef LIBTUTOOLS_DEBUG
    if (i < 0 || u_int(i) >= size())
	throw std::out_of_range("TU::Array<T, B>::operator []: invalid index!");
#endif
    return *(begin() + i);
}
#if 0
template <class T, class B> inline const typename Array<T, B>::element_type&
Array<T, B>::eval(u_int i, u_int j) const
{
    return (*this)[i][j];
}
#endif
//! 全ての要素に同一の数値を掛ける．
/*!
  \param c	掛ける数値
  \return	この配列
*/
template <class T, class B> Array<T, B>&
Array<T, B>::operator *=(element_type c)
{
    for (iterator q = begin(); q != end(); ++q)
	*q *= c;
    return *this;
}

//! 全ての要素を同一の数値で割る．
/*!
  \param c	割る数値
  \return	この配列
*/
template <class T, class B> template <class T2> inline Array<T, B>&
Array<T, B>::operator /=(T2 c)
{
    for (iterator q = begin(); q != end(); ++q)
	*q /= c;
    return *this;
}

//! この配列に他の配列を足す．
/*!
  \param a	足す配列
  \return	この配列
*/
template <class T, class B> template <class T2, class B2> Array<T, B>&
Array<T, B>::operator +=(const Array<T2, B2>& a)
{
  //check_size(a.size());
    typename Array<T2, B2>::const_iterator	p = a.cbegin();
    for (iterator q = begin(); q != end(); ++q, ++p)
	*q += *p;
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
  //check_size(a.size());
    typename Array<T2, B2>::const_iterator	p = a.cbegin();
    for (iterator q = begin(); q != end(); ++q, ++p)
	*q -= *p;
    return *this;
}

//! 2つの配列を要素毎に比較し，同じであるか調べる．
/*!
  \param expr	比較対象となる配列
  \return	全ての要素が同じならばtrue，そうでなければfalse
*/
template <class T, class B> template <class T2, class B2> bool
Array<T, B>::operator ==(const Array<T2, B2>& a) const
{
    if (size() != a.size())
	return false;
    typename Array<T2, B2>::const_iterator	p = a.cbegin();
    for (const_iterator q = cbegin(); q != cend(); ++q, ++p)
	if (*q != *p)
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

//! 入力ストリームから指定した箇所に配列を読み込む(ASCII)．
/*!
  \param in	入力ストリーム
  \param m	読み込み先の先頭を指定するindex
  \return	inで指定した入力ストリーム
*/
template <class T, class B> inline std::istream&
Array<T, B>::get(std::istream& in)
{
    return super::get(in);
}

//! 出力ストリームに配列を書き出す(ASCII)．
/*!
  \param out	出力ストリーム
  \return	outで指定した出力ストリーム
*/
template <class T, class B> std::ostream&
Array<T, B>::put(std::ostream& out) const
{
    for (const_iterator q = cbegin(); q != cend(); ++q)
	out << ' ' << *q;
    return out;
}

//! 入力ストリームから配列を読み込む(binary)．
/*!
  \param in	入力ストリーム
  \return	inで指定した入力ストリーム
*/
template <class T, class B> inline std::istream&
Array<T, B>::restore(std::istream& in)
{
    in.read((char*)data(), sizeof(value_type) * size());
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
    out.write((const char*)data(), sizeof(value_type) * size());
    return out;
}

//! 指定した値がこの配列の要素数に一致するか調べる．
/*!
  \param d			調べたい値
  \throw std::logic_error	d != size()の場合に送出
*/
template <class T, class B> inline void
Array<T, B>::check_size(u_int d) const
{
    if (d != size())
	throw std::logic_error("Array<T, B>::check_size: mismatched size!");
}

template <class T, class B> inline u_int
Array<T, B>::partial_size(u_int i, u_int d, u_int a)
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
    typedef Array<T, R>					super;
    typedef B						buf_type;
    
  public:
  //! 成分の型    
    typedef typename super::element_type		element_type;
  //! 行の型    
    typedef typename super::value_type			value_type;
  //! 行の定数反復子    
    typedef typename super::const_iterator		const_iterator;
  //! 行の反復子    
    typedef typename super::iterator			iterator;
  //! 行の定数逆反復子    
    typedef typename super::const_reverse_iterator	const_reverse_iterator;
  //! 行の逆反復子    
    typedef typename super::reverse_iterator		reverse_iterator;
  //! 定数行への参照    
    typedef typename super::const_reference		const_reference;
  //! 行への参照    
    typedef typename super::reference			reference;
  //! 定数要素へのポインタ    
    typedef typename buf_type::const_pointer		const_pointer;
  //! 要素へのポインタ    
    typedef typename buf_type::pointer			pointer;
  //! ポインタ間の差    
    typedef std::ptrdiff_t				difference_type;

  public:
    Array2()								;
    Array2(u_int r, u_int c)						;
    Array2(pointer p, u_int r, u_int c)					;
    Array2(buf_type buf, u_int r, u_int c)				;
    template <class B2, class R2>
    Array2(Array2<T, B2, R2>& a, u_int i, u_int j, u_int r, u_int c)	;
    Array2(const Array2& a)						;
    Array2&	operator =(const Array2& a)				;
    template <class T2, class B2, class R2>
    Array2(const Array2<T2, B2, R2>& a)					;
    template <class T2, class B2, class R2>
    Array2&	operator =(const Array2<T2, B2, R2>& a)			;
    Array2&	operator =(const element_type& c)			;
#ifdef __CXX0X
    Array2(std::initializer_list<value_type> args)			;
    Array2&	operator =(std::initializer_list<value_type> args)	;
#endif
    using		super::cbegin;
    using		super::begin;
    using		super::cend;
    using		super::end;
    using		super::crbegin;
    using		super::rbegin;
    using		super::crend;
    using		super::rend;
    using		super::size;
    using		super::dim;
    
    const_pointer	data()					const	;
    pointer		data()						;
    u_int		nrow()					const	;
    u_int		ncol()					const	;
    u_int		stride()				const	;
    bool		resize(u_int r, u_int c)			;
    void		resize(pointer p, u_int r, u_int c)		;
    std::istream&	restore(std::istream& in)			;
    std::ostream&	save(std::ostream& out)			const	;
    std::istream&	get(std::istream& in,
			    u_int i=0, u_int j=0, u_int jmax=0)	;

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
    :super(r), _ncol(c), _buf(nrow()*_buf.stride(ncol()))
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
    :super(r), _ncol(c), _buf(p, nrow()*_buf.stride(ncol()))
{
    set_rows();
}

//! 記憶領域を元の配列と共有した部分配列を作る.
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
    :super(super::partial_size(i, r, a.nrow())),
     _ncol(super::partial_size(j, c, a.ncol())),
     _buf((nrow() > 0 && ncol() > 0 ?
	   pointer(&a[i][j]) : pointer((typename buf_type::value_type*)0)),
	  nrow()*_buf.stride(ncol()))
{
    for (u_int ii = 0; ii < nrow(); ++ii)
	(*this)[ii].resize(pointer(&a[i+ii][j]), ncol());
}    

//! コピーコンストラクタ
/*!
  \param a	コピー元の配列
*/
template <class T, class B, class R> inline
Array2<T, B, R>::Array2(const Array2& a)
    :super(a.nrow()), _ncol(a.ncol()),
     _buf(nrow()*_buf.stride(ncol()))
{
    set_rows();
    super::operator =((const super&)a);
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
    super::operator =((const super&)a);
    return *this;
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
    :super(a.nrow()), _ncol(a.ncol()),
     _buf(nrow()*_buf.stride(ncol()))
{
    set_rows();
    super::operator =(a);
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

#ifdef __CXX0X
template <class T, class B, class R>
Array2<T, B, R>::Array2(std::initializer_list<value_type> args)
    :super(args.size()), _ncol(args.size() ? args.begin()->size() : 0),
     _buf(nrow()*_buf.stride(ncol()))
{
    set_rows();
    u_int	i = 0;
    for (auto val : args)
	(*this)[i++] = val;
}

template <class T, class B, class R> Array2<T, B, R>&
Array2<T, B, R>::operator =(std::initializer_list<value_type> args)
{
    resize(args.size(), (args.size() ? args.begin()->size() : 0));
    u_int	i = 0;
    for (auto val : args)
	(*this)[i++] = val;
    return *this;
}
#endif

//! 全ての要素に同一の値を代入する．
/*!
  \param c	代入する値
  \return	この配列
*/
template <class T, class B, class R> Array2<T, B, R>&
Array2<T, B, R>::operator =(const element_type& c)
{
    std::fill(begin(), end(), c);
    return *this;
}

//! 2次元配列の内部記憶領域への定数ポインタを返す．
/*!
  \return	内部記憶領域への定数ポインタ
*/
template <class T, class B, class R>
inline typename Array2<T, B, R>::const_pointer
Array2<T, B, R>::data() const
{
    return _buf.data();
}

//! 2次元配列の内部記憶領域へのポインタを返す．
/*!
  \return	内部記憶領域へのポインタ
*/
template <class T, class B, class R> inline typename Array2<T, B, R>::pointer
Array2<T, B, R>::data()
{
    return _buf.data();
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
    return (nrow() > 1 ? (*this)[1].data() - (*this)[0].data() : _ncol);
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
    _buf.resize(nrow()*_buf.stride(ncol()));
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
    _buf.resize(p, nrow()*_buf.stride(ncol()));
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
    for (iterator q = begin(); q != end(); ++q)
	q->restore(in);
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
    for (const_iterator q = cbegin(); q != cend(); ++q)
	q->save(out);
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
    element_type	val;
    in >> val;
    get(in, i, j + 1, jmax);
    (*this)[i][j] = val;
    return in;
}    

template <class T, class B, class R> void
Array2<T, B, R>::set_rows()
{
    const u_int	stride = _buf.stride(ncol());
    for (u_int i = 0; i < nrow(); ++i)
	(*this)[i].resize(_buf.data() + i*stride, ncol());
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
