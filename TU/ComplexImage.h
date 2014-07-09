/*
 *  平成21-22年（独）産業技術総合研究所 著作権所有
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
 *  Copyright 2009-2010.
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
 *  $Id: ComplexImage.h,v 1.2 2010-11-22 06:16:19 ueshiba Exp $
 */
#ifndef __TU_COMPLEXIMAGE_H
#define __TU_COMPLEXIMAGE_H

#include "TU/CCSImage.h"

namespace TU
{
/************************************************************************
*  class ComplexImageLine<T>						*
************************************************************************/
//! 複素数値1次元配列
template <class T>
class ComplexImageLine : public ImageLine<std::complex<T> >
{
  public:
    typedef ImageLine<std::complex<T> >			line_type;
    typedef typename line_type::iterator		iterator;
    typedef typename line_type::const_iterator		const_iterator;
    typedef typename line_type::reverse_iterator	reverse_iterator;
    typedef typename line_type::const_reverse_iterator	const_reverse_iterator;

  private:
    typedef CCSImageLine<T>				ccs_type;
    typedef typename ccs_type::iterator			ccs_iterator;
    typedef typename ccs_type::const_iterator		ccs_const_iterator;
    typedef typename ccs_type::reverse_iterator		ccs_reverse_iterator;
    typedef typename ccs_type::const_reverse_iterator	ccs_const_reverse_iterator;
    
  public:
  //! 複素数値1次元配列を生成する．
  /*!
    \param d	配列の要素数
  */
    explicit ComplexImageLine(u_int d=0) :line_type(d)			{}

    using		line_type::begin;
    using		line_type::end;
    using		line_type::rbegin;
    using		line_type::rend;
    using		line_type::size;
    using		line_type::check_size;

    CCSImageLine<T>	real()					const	;
    CCSImageLine<T>	imag()					const	;
    CCSImageLine<T>	mag()					const	;
    CCSImageLine<T>	specmag()				const	;
    ComplexImageLine<T>&
			operator *=(const ComplexImageLine<T>& spectrum);
    ComplexImageLine<T>&
			operator /=(const CCSImageLine<T>& magnitude)	;
    T			maximum(T& uMax)			const	;
};

//! この複素数値1次元配列の実数部分から成る1次元配列を返す．
/*!
  \return	この配列の各要素の実数部分を要素とした実数値1次元配列
*/
template <class T> CCSImageLine<T>
ComplexImageLine<T>::real() const
{
    CCSImageLine<T>	ccs(size());
    const_iterator	p = begin();
    for (ccs_iterator q = ccs.begin(); q != ccs.end(); )
	*q++ = (p++)->real();

    return ccs;
}
    
//! この複素数値1次元配列の虚数部分から成る1次元配列を返す．
/*!
  \return	この配列の各要素の虚数部分を要素とした実数値1次元配列
*/
template <class T> CCSImageLine<T>
ComplexImageLine<T>::imag() const
{
    CCSImageLine<T>	ccs(size());
    const_iterator	p = begin();
    for (ccs_iterator q = ccs.begin(); q != ccs.end(); )
	*q++ = (p++)->imag();

    return imag;
}
    
//! 空間領域にある複素数値1次元配列の各要素の振幅を要素とする実数値1次元配列を返す．
/*!
  返される1次元配列は空間領域に属する．また，原点は配列の左端となる．
  \return	振幅を要素とする1次元配列
*/
template <class T> CCSImageLine<T>
ComplexImageLine<T>::mag() const
{
    using namespace	std;
    
    CCSImageLine<T>	ccs(size());
    const_iterator	p = begin();
    for (ccs_iterator q = ccs.begin(); q != ccs.end(); )
	*q++ = abs(*p++);

    return ccs;
}

//! 周波数領域にある複素数値1次元配列の各要素の振幅を要素とする実数値1次元配列を返す．
/*!
  返される1次元配列は空間領域に属する．また，原点が配列の左端から中央に移される．
  \return	振幅を要素とする1次元配列
*/
template <class T> CCSImageLine<T>
ComplexImageLine<T>::specmag() const
{
    using namespace	std;
    
    CCSImageLine<T>	ccs(size());
    ccs_iterator	mid = ccs.begin() + ccs.size() / 2;
    const_iterator	p = begin();
    for (ccs_iterator q = mid; q != ccs.end(); )
	*q++ = abs(*p++);
    for (ccs_iterator q = begin(); q != mid; )
	*q++ = abs(*p++);

    return ccs;
}
    
//! この複素数値1次元配列に別の1次元配列の複素共役を掛ける．
/*!
  \param spectrum	複素数値1次元配列	
  \return		specturmの複素共役を掛けた後のこの配列
*/
template <class T> ComplexImageLine<T>&
ComplexImageLine<T>::operator *=(const ComplexImageLine<T>& spectrum)
{
    using namespace	std;
    
    check_size(spectrum.size());
    
    const_iterator	p = spectrum.begin();
    for (iterator q = begin(); q != end(); )
	*q++ *= conj(*p++);

    return *this;
}

//! この複素数値1次元配列を実数値1次元配列で割る．
/*!
  \param magnitude	実数値1次元配列	
  \return		商をとった後のこの複素数値1次元配列
*/
template <class T> ComplexImageLine<T>&
ComplexImageLine<T>::operator /=(const CCSImageLine<T>& magnitude)
{
    check_size(magnitude.size());

    T			thresh = magnitude.maximum() * 0.1;
    ccs_const_iterator	p = magnitude.begin();
    for (iterator q = begin(); q != end(); ++q)
    {
	const T	val = *p++;
	if (val > thresh)
	    *q /= val;
	else
	    *q = 0;
    }
    
    return *this;
}

//! この複素数値1次元配列の配列要素中の最大値とその位置を返す．
/*!
  \param uMax	最大値を与える要素の位置
  \return	最大値
*/
template <class T> T
ComplexImageLine<T>::maximum(T& uMax) const
{
    using namespace	std;
    
#if _DEBUG >= 2
    Image<T>	tmp(50, size());
#endif
    T	valMax = numeric_limits<T>::min();
    for (u_int u = 0; u < size(); ++u)
    {
	const T	val = abs((*this)[u]);
	if (val > valMax)
	{
	    valMax = val;
	    uMax   = u;
	}
#if _DEBUG >= 2
	for (u_int r = 0; r < tmp.width(); ++r)
	    tmp[u][r] = val;
#endif
    }
#if _DEBUG >= 2
    tmp *= T(255) / valMax;
    tmp.save(cout, ImageBase::FLOAT);
    cerr << "ComplexImageLine<T>::maximum()..." << endl;
#endif
#ifdef _DEBUG
    cerr << "ComplexImageLine<T>::maximum(): " << valMax << "@(" << uMax
	 << ") in [0, " << size() << ')' << endl;
#endif
    return valMax;
}

/************************************************************************
*  class ComplexImage<T>						*
************************************************************************/
//! 複素数値2次元配列
template <class T>
class ComplexImage : public Image<std::complex<T> >
{
  public:
    typedef Image<std::complex<T> >			image_type;
    typedef typename image_type::iterator		iterator;
    typedef typename image_type::const_iterator		const_iterator;
    typedef typename image_type::reverse_iterator	reverse_iterator;
    typedef typename image_type::const_reverse_iterator	const_reverse_iterator;
    typedef ImageLine<std::complex<T> >			line_type;
    typedef typename line_type::iterator		pixel_iterator;
    typedef typename line_type::const_iterator		const_pixel_iterator;
    typedef typename line_type::reverse_iterator	reverse_pixel_iterator;
    typedef typename line_type::const_reverse_iterator
						const_reverse_pixel_iterator;

  private:
    typedef CCSImage<T>					ccs_image_type;
    typedef typename ccs_image_type::iterator		ccs_iterator;
    typedef typename ccs_image_type::const_iterator	ccs_cons_iterator;
    typedef typename ccs_image_type::reverse_iterator	ccs_reverse_iterator;
    typedef typename ccs_image_type::const_reverse_iterator
						ccs_const_reverse_iterator;
    typedef ImageLine<T>				ccs_line_type;
    typedef typename ccs_line_type::iterator		ccs_pixel_iterator;
    typedef typename ccs_line_type::const_iterator	ccs_const_pixel_iterator;
    typedef typename ccs_line_type::reverse_iterator	ccs_reverse_pixel_iterator;
    typedef typename ccs_line_type::const_reverse_iterator
						ccs_const_reverse_pixel_iterator;
    
  public:
  //! 複素数値2次元配列を生成する．
  /*!
    \param w	配列の幅
    \param h	配列の高さ
  */
    explicit ComplexImage(u_int w=0, u_int h=0)	:image_type(w, h)	{}

    using		image_type::begin;
    using		image_type::end;
    using		image_type::rbegin;
    using		image_type::rend;
    using		image_type::size;
    using		image_type::width;
    using		image_type::height;
    using		image_type::check_size;

    CCSImage<T>		real()					const	;
    CCSImage<T>		imag()					const	;
    CCSImage<T>		mag()					const	;
    CCSImage<T>		specmag()				const	;
    ComplexImage<T>&	operator *=(const ComplexImage<T>& spectrum)	;
    ComplexImage<T>&	operator /=(const CCSImage<T>& magnitude)	;
    T			maximum(Point2<T>& pMax)		const	;
};

//! この複素数値2次元配列の実数部分から成る2次元配列を返す．
/*!
  \return	この配列の各要素の実数部分を要素とした実数値2次元配列
*/
template <class T> CCSImage<T>
ComplexImage<T>::real() const
{
    CCSImage<T>	ccs(width(), height());
    for (u_int v = 0; v < ccs.height(); ++v)
    {
	ccs_line_type&		line = ccs[v];
	const_pixel_iterator	p = (*this)[v].begin();
	for (ccs_pixel_iterator q = line.begin(); q != line.end(); )
	    *q++ = (p++)->real();
    }

    return ccs;
}

//! この複素数値2次元配列の虚数部分から成る2次元配列を返す．
/*!
  \return	この配列の各要素の虚数部分を要素とした実数値2次元配列
*/
template <class T> CCSImage<T>
ComplexImage<T>::imag() const
{
    CCSImage<T>	ccs(width(), height());
    for (u_int v = 0; v < ccs.height(); ++v)
    {
	ccs_line_type&		line = ccs[v];
	const_pixel_iterator	p = (*this)[v].begin();
	for (ccs_pixel_iterator q = line.begin(); q != line.end(); )
	    *q++ = (p++)->imag();
    }

    return ccs;
}

//! 空間領域にある複素数値2次元配列の各要素の振幅を要素とする2次元配列を返す．
/*!
  返される2次元配列の原点は配列の左上隅から移動されない．
  \return	振幅を要素とする2次元配列
*/
template <class T> CCSImage<T>
ComplexImage<T>::mag() const
{
    using namespace	std;
    
    CCSImage<T>	ccs(width(), height());
    for (u_int v = 0; v < ccs.height(); ++v)
    {
	ccs_line_type&		line = ccs[v];
	const_pixel_iterator	p = (*this)[v].begin();
	for (ccs_pixel_iterator q = line.begin(); q != line.end(); )
	    *q++ = abs(*p++);
    }
#if _DEBUG >= 3
    Image<T>	tmp(ccs);
    tmp *= T(255) / ccs.maximum();
    tmp.save(cout, ImageBase::FLOAT);
    cerr << "ComplexImage<T>::mag()..." << endl;
#endif

    return ccs;
}
    
//! 周波数領域にある複素数値2次元配列の各要素の振幅を要素とする2次元配列を返す．
/*!
  返される2次元配列の原点は配列の左上隅ではなく中央に位置する．
  \return	振幅を要素とする2次元配列
*/
template <class T> CCSImage<T>
ComplexImage<T>::specmag() const
{
    using namespace	std;
    
    CCSImage<T>	ccs(width(), height());
    const u_int	v2 = ccs.height() / 2;
    for (u_int v = 0; v < v2; ++v)
    {
	ccs_line_type&		line = ccs[v + v2];
	ccs_pixel_iterator	mid = line.begin() + line.size() / 2;
	const_pixel_iterator	p = (*this)[v].begin();
	for (ccs_pixel_iterator q = mid; q != line.end(); )
	    *q++ = abs(*p++);
	for (ccs_pixel_iterator q = line.begin(); q != mid; )
	    *q++ = abs(*p++);
    }
    for (u_int v = v2; v < height(); ++v)
    {
	ccs_line_type&		line = ccs[v - v2];
	ccs_pixel_iterator	mid = line.begin() + line.size() / 2;
	const_pixel_iterator	p = (*this)[v].begin();
	for (ccs_pixel_iterator q = mid; q != line.end(); )
	    *q++ = abs(*p++);
	for (ccs_pixel_iterator q = line.begin(); q != mid; )
	    *q++ = abs(*p++);
    }
#if _DEBUG >= 3
    Image<T>	tmp(ccs);
    tmp *= T(255) / ccs.maximum();
    tmp.save(cout, ImageBase::FLOAT);
    cerr << "ComplexImage<T>::specmag()..." << endl;
#endif

    return ccs;
}

//! この複素数値2次元配列が周波数領域にあるとき，これに別の2次元配列の複素共役を掛ける．
/*!
  \param spectrum	周波数領域にある複素数値2次元配列	
  \return		specturmの複素共役を掛けた後のこの配列
*/
template <class T> ComplexImage<T>&
ComplexImage<T>::operator *=(const ComplexImage<T>& spectrum)
{
    using namespace	std;
    
    check_size(spectrum.size());
    
    for (u_int v = 0; v < height(); ++v)
    {
	line_type&		line = (*this)[v];
	const_pixel_iterator	p = spectrum[v].begin();
	for (pixel_iterator q = line.begin(); q != line.end(); )
	    *q++ *= conj(*p++);
    }

    return *this;
}
    
//! この複素数値2次元配列が空間領域にあるとき，これを実数値2次元配列で割る．
/*!
  \param magnitude	空間領域にある実数値2次元配列	
  \return		商をとった後のこの複素数値2次元配列
*/
template <class T> ComplexImage<T>&
ComplexImage<T>::operator /=(const CCSImage<T>& magnitude)
{
    check_size(magnitude.size());

    T	thresh = magnitude.maximum() * 0.1;
    for (u_int v = 0; v < height(); ++v)
    {
	line_type&			line = (*this)[v];
	ccs_const_pixel_iterator	p = magnitude[v].begin();
	for (pixel_iterator q  = line.begin(); q != line.end(); ++q)
	{
	    const T	val = *p++;
	    if (val > thresh)
		*q /= val;
	    else
		*q = 0;
	}
    }
    
    return *this;
}

//! この複素数値2次元配列が空間領域にあるとき，配列要素中の最大値とその位置を返す．
/*!
  \param pMax	最大値を与える要素の位置
  \return	最大値
*/
template <class T> T
ComplexImage<T>::maximum(Point2<T>& pMax) const
{
    using namespace	std;
    
#if _DEBUG >= 3
    Image<T>	tmp(width(), height());
#endif
    T	valMax = numeric_limits<T>::min();
    for (u_int v = 0; v < height(); ++v)
    {
	for (u_int u = 0; u < width(); ++u)
	{
	    const T	val = abs((*this)[v][u]);
	    if (val > valMax)
	    {
		valMax	= val;
		pMax[0] = u;
		pMax[1] = v;
	    }
#if _DEBUG >= 3
	    tmp[v][u] = val;
#endif	    
	}
    }
#if _DEBUG >= 3
    tmp *= T(255) / valMax;
    tmp.save(cout, ImageBase::FLOAT);
    cerr << "ComplexImage<T>::maximum()..." << endl;
#endif
#ifdef _DEBUG
    cerr << "ComplexImage<T>::maximum(): " << valMax
	 << "@(" << pMax[0] << ", " << pMax[1] << ") in ["
	 << width() << 'x' << height() << ']' << endl;
#endif
    return valMax;
}

}
#endif	// !__TU_COMPLEXIMAGE_H
