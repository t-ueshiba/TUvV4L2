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
 *  $Id: CCSImage.h,v 1.2 2010-11-22 06:16:19 ueshiba Exp $
 */
#ifndef __TU_CCSIMAGE_H
#define __TU_CCSIMAGE_H

#include "TU/Image++.h"
#include <complex>
#include <limits>

namespace TU
{
/************************************************************************
*  static functions							*
************************************************************************/
template <class T> static inline void
setSign(T& val)
{
    val = (val > T(0) ? T(1) : val < T(0) ? T(-1) : T(0));
}
    
/************************************************************************
*  class CCSImageLine<T>						*
************************************************************************/
//! その離散フーリエ変換が complex conjugate symmetrical 形式で表される実数値1次元配列
template <class T>
class CCSImageLine : public ImageLine<T>
{
  public:
    typedef ImageLine<T>				line_type;
    typedef typename line_type::iterator		iterator;
    typedef typename line_type::const_iterator		const_iterator;
    typedef typename line_type::reverse_iterator	reverse_iterator;
    typedef typename line_type::const_reverse_iterator	const_reverse_iterator;
    
  public:
  //! CCS形式の1次元配列を生成する．
  /*!
    \param d	配列の要素数
  */
    explicit CCSImageLine(u_int d=0) :line_type(d)			{}

    using		line_type::begin;
    using		line_type::end;
    using		line_type::rbegin;
    using		line_type::rend;
    using		line_type::size;
    using		line_type::check_size;
    
    std::complex<T>	operator ()(u_int u)			const	;
    std::complex<T>	conj(u_int u)				const	;
    CCSImageLine<T>	mag()					const	;
    CCSImageLine<T>	specmag()				const	;
    CCSImageLine<T>&	pdiff(const CCSImageLine<T>& spectrum)		;
    CCSImageLine<T>&	operator *=(const CCSImageLine<T>& spectrum)	;
    CCSImageLine<T>&	operator /=(const CCSImageLine<T>& magnitude)	;
    T			maximum(T& uMax)			const	;
    T			maximum()				const	;
};

//! このCCS配列が周波数領域にあるとき，指定されたindexに対する値（複素数）を返す．
/*!
  \param u	index
  \return	u によって指定された要素の値
*/
template <class T> inline std::complex<T>
CCSImageLine<T>::operator ()(u_int u) const
{
    using namespace	std;

    const u_int	u2 = size() / 2;
    if (u == 0)
	return (*this)[0];
    else if (u < u2)
    {
	const u_int	uu = 2*u;
	return complex<T>((*this)[uu-1], (*this)[uu]);
    }
    else if (u == u2)
	return (*this)[size() - 1];
    else
    {
	const u_int	uu = 2*(size() - u);
	return complex<T>((*this)[uu-1], (*this)[uu]);
    }
}

//! このCCS配列が周波数領域にあるとき，指定されたindexに対する値の共役複素数値を返す．
/*!
  \param u	index
  \return	u によって指定された要素の値の共役複素数値
*/
template <class T> inline std::complex<T>
CCSImageLine<T>::conj(u_int u) const
{
    return std::complex<T>((*this)[u]);
}

//! このCCS配列が空間領域にあるとき，各要素の振幅を要素とする1次元配列を返す．
/*!
  返される1次元配列は空間領域に属する．また，原点は配列の左端となる．
  \return	振幅を要素とする1次元配列
*/
template <class T> CCSImageLine<T>
CCSImageLine<T>::mag() const
{
    using namespace	std;
    
    CCSImageLine<T>	ccs(size());
    const_iterator	p = begin();
    for (iterator q = ccs.begin(); q != ccs.end(); )
	*q++ = abs(*p++);
    
    return ccs;
}

//! このCCS配列が周波数領域にあるとき，各要素の振幅を要素とする1次元配列を返す．
/*!
  返される1次元配列は空間領域に属する．また，原点が配列の左端から中央に移される．
  \return	振幅を要素とする1次元配列
*/
template <class T> CCSImageLine<T>
CCSImageLine<T>::specmag() const
{
    using namespace	std;
    
    CCSImageLine<T>	ccs(size());
    const u_int		u2 = ccs.size() / 2;
    iterator		pF = ccs.begin() + u2;
    iterator		pB = pF;
    for (const_iterator p = begin() + 1, pe = end() - 1; p != pe; p += 2)
    {
	*++pF = *--pB = abs(complex<T>(*p, *(p+1)));
    }
    ccs[u2] = abs((*this)[0]);
    ccs[0]  = abs((*this)[size() - 1]);
    
    return ccs;
}

//! このCCS配列が周波数領域にあるとき，与えられたもうひとつのCCS配列との位相差に変換する．
/*!
  \param spectrum	周波数領域にあるCCS配列	
  \return		位相差を表す1次元配列
*/
template <class T> CCSImageLine<T>&
CCSImageLine<T>::pdiff(const CCSImageLine<T>& spectrum)
{
    using namespace	std;
    
    check_size(spectrum.size());

    iterator	p = begin() + 1;
    for (const_iterator q  = spectrum.begin() + 1,
			qe = spectrum.end()   - 1; q != qe; q += 2)
    {
	complex<T>	val = complex<T>(*p, *(p+1)) * complex<T>(*q, -*(q+1));
	if (val != T(0))
	    val /= abs(val);
	*p++ = val.real();
	*p++ = val.imag();
    }
    const u_int	u1 = size() - 1;
    setSign((*this)[0]  *= spectrum[0]);
    setSign((*this)[u1] *= spectrum[u1]);

    return *this;
}

//! このCCS配列が周波数領域にあるとき，これに別のCCS配列の複素共役を掛ける．
/*!
  \param spectrum	周波数領域にあるCCS配列	
  \return		specturmの複素共役を掛けた後のこの配列
*/
template <class T> CCSImageLine<T>&
CCSImageLine<T>::operator *=(const CCSImageLine<T>& spectrum)
{
    using namespace	std;
    
    check_size(spectrum.size());

    iterator	p = begin() + 1;
    for (const_iterator q  = spectrum.begin() + 1,
			qe = spectrum.end()   - 1; q != qe; q += 2)
    {
	complex<T>	val = complex<T>(*p, *(p+1)) * complex<T>(*q, -*(q+1));
	*p++ = val.real();
	*p++ = val.imag();
    }
    const u_int	u1 = size() - 1;
    (*this)[0]  *= spectrum[0];
    (*this)[u1] *= spectrum[u1];

    return *this;
}

//! このCCS配列が空間領域にあるとき，もうひとつのCCS配列で割る．
/*!
  \param spectrum	空間領域にあるCCS配列	
  \return		商をとった後のこのCCS配列
*/
template <class T> CCSImageLine<T>&
CCSImageLine<T>::operator /=(const CCSImageLine<T>& magnitude)
{
    using namespace	std;
    
    check_size(magnitude.size());

    T			thresh = magnitude.maximum() * 0.1;
    const_iterator	p = magnitude.begin();
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

//! このCCS配列が空間領域にあるとき，配列要素中の最大値とその位置を返す．
/*!
  \param uMax	最大値を与える要素の位置
  \return	最大値
*/
template <class T> T
CCSImageLine<T>::maximum(T& uMax) const
{
    using namespace	std;
    
    T	valMax = numeric_limits<T>::min();
    for (u_int u = 0; u < size(); ++u)
    {
	const T	val = (*this)[u];
	if (val > valMax)
	{
	    valMax = val;
	    uMax   = u;
	}
    }
#if _DEBUG >= 2
    Image<T>	tmp(50, size());
    for (u_int t = 0; t < tmp.height(); ++t)
	tmp[t] = (*this)[t] * T(255) / valMax;
    tmp.save(cout, ImageBase::FLOAT);
    cerr << "CCSImageLine<T>::maximum()..." << endl;
#endif
#ifdef _DEBUG
    cerr << "CCSImageLine<T>::maximum(): " << valMax << "@(" << uMax
	 << ") in [0, " << size() << ')' << endl;
#endif
    return valMax;
}
    
template <class T> T
CCSImageLine<T>::maximum() const
{
    using namespace	std;
    
    T	valMax = numeric_limits<T>::min();
    for (u_int u = 0; u < size(); ++u)
    {
	const T	val = (*this)[u];
	if (val > valMax)
	    valMax = val;
    }

    return valMax;
}

/************************************************************************
*  class CCSImage<T>							*
************************************************************************/
//! その離散フーリエ変換が complex conjugate symmetrical 形式で表される実数値2次元配列
template <class T>
class CCSImage : public Image<T>
{
  public:
    typedef Image<T>					image_type;
    typedef typename image_type::iterator		iterator;
    typedef typename image_type::const_iterator		const_iterator;
    typedef typename image_type::reverse_iterator	reverse_iterator;
    typedef typename image_type::const_reverse_iterator	const_reverse_iterator;
    typedef ImageLine<T>				line_type;
    typedef typename line_type::iterator		pixel_iterator;
    typedef typename line_type::const_iterator		const_pixel_iterator;
    typedef typename line_type::reverse_iterator	reverse_pixel_iterator;
    typedef typename line_type::const_reverse_iterator	const_reverse_pixel_iterator;
    
  public:
  //! CCS形式の2次元配列を生成する．
  /*!
    \param w	配列の幅
    \param h	配列の高さ
  */
    explicit CCSImage(u_int w=0, u_int h=0) :image_type(w, h)		{}

    using		image_type::begin;
    using		image_type::end;
    using		image_type::rbegin;
    using		image_type::rend;
    using		image_type::size;
    using		image_type::width;
    using		image_type::height;
    using		image_type::check_size;

    std::complex<T>	operator ()(u_int u, u_int v)		const	;
    std::complex<T>	conj(u_int u, u_int v)			const	;
    CCSImage<T>		specmag()				const	;
    CCSImage<T>		logpolar()				const	;
    CCSImageLine<T>	intpolar()				const	;
    CCSImage<T>&	pdiff(const CCSImage<T>& spectrum)		;
    CCSImage<T>&	operator *=(const CCSImage<T>& spectrum)	;
    T			maximum(Point2<T>& pMax)		const	;
    T			maximum()				const	;
};

//! このCCS配列が周波数領域にあるとき，指定されたindexに対する値（複素数）を返す．
/*!
  \param u	横index
  \param v	縦index
  \return	u, v によって指定された要素の値
*/
template <class T> std::complex<T>
CCSImage<T>::operator ()(u_int u, u_int v) const
{
    using namespace	std;
    
    const u_int	u2 = width() / 2;
    u_int	uu;
    if (u == 0)
	uu = 0;
    else if (u < u2)
    {
	uu = 2*u;
	return complex<T>((*this)[v][uu-1], (*this)[v][uu]);
    }
    else if (u == u2)
	uu = width() - 1;
    else
    {
	uu = 2*(width() - u);
	return complex<T>((*this)[v][uu-1], -(*this)[v][uu]);
    }

    const u_int	v2 = height() / 2;
    u_int	vv;
    if (v == 0)
	return (*this)[0][uu];
    else if (v < v2)
    {
	vv = 2*v;
	return complex<T>((*this)[vv-1][uu], (*this)[vv][uu]);
    }
    else if (v == v2)
	return (*this)[height() - 1][uu];
    else
    {
	vv = 2*(height() - v);
	return complex<T>((*this)[vv-1][uu], -(*this)[vv][uu]);
    }
}

//! このCCS配列が周波数領域にあるとき，指定されたindexに対する値の共役複素数値を返す．
/*!
  \param u	横index
  \param v	縦index
  \return	u, v によって指定された要素の値
*/
template <class T> inline std::complex<T>
CCSImage<T>::conj(u_int u, u_int v) const
{
    return std::conj((*this)(u, v));
}

//! このCCS配列が周波数領域にあるとき，各要素の振幅を要素とする2次元配列を返す．
/*!
  返される2次元配列は空間領域に属し，原点に対して対称である．この原点は配列の
  左上隅ではなく中央に位置する．
  \return	振幅を要素とする2次元配列
*/
template <class T> CCSImage<T>
CCSImage<T>::specmag() const
{
    using namespace	std;
    
  // CCS行列から振幅情報を取り出す．
  // 1. CCS行列の最初と最後の列以外の振幅を計算する．
    CCSImage<T>	ccs(width(), height());
    const u_int	u2 = ccs.width()  / 2;
    const u_int	v2 = ccs.height() / 2;
    iterator	lF = ccs.begin() + v2;
    iterator	lB = lF;
    for (const_iterator l = begin(); l != end(); ++l)
    {
	if (lF == ccs.end())
	{
	    lF = ccs.begin();
	    lB = ccs.end();
	}
	pixel_iterator	pF = (lF++)->begin() + u2;
	pixel_iterator	pB = (--lB)->begin() + u2;
	for (const_pixel_iterator p  = l->begin() + 1,
				  pe = l->end()	  - 1; p != pe; p += 2)
	{
	    *++pF = *--pB = abs(complex<T>(*p, *(p+1)));
	}
    }

  // 2. CCS行列の最初と最後の列（最初と最後の行以外）の振幅を計算する．
    const u_int	u1 = width() - 1;
    lF = ccs.begin() + v2;
    lB = lF;
    for (const_iterator l = begin() + 1; l != end() - 1; l += 2)
    {
	(*++lF)[u2] = (*--lB)[u2] = abs(complex<T>((*l)[0],  (*(l+1))[0]));
	(*lF)[0]    = (*lB)[0]    = abs(complex<T>((*l)[u1], (*(l+1))[u1]));
    }

  // 3. CCS行列の四隅の振幅を計算する．
    const u_int	v1 = height() - 1;
    ccs[v2][u2] = abs((*this)[0][0]);
    ccs[v2][0]  = abs((*this)[0][u1]);
    ccs[0][u2]  = abs((*this)[v1][0]);
    ccs[0][0]   = abs((*this)[v1][u1]);

#if _DEBUG >= 3
    Image<T>	tmp(ccs);
    tmp *= T(255) / ccs.maximum();
    tmp.save(cout, ImageBase::FLOAT);
    cerr << "CCSImage<T>::mag()..." << endl;
#endif
    return ccs;
}

//! このCCS配列が空間領域にあるとき，そのlog-polar表現を返す．
/*!
  \return		log-polar座標系で表現された2次元配列
*/
template <class T> CCSImage<T>
CCSImage<T>::logpolar() const
{
    using namespace	std;

    const u_int		size = max(width(), height()) / 2;

    static Array<T>	windowR;
    initializeHanningWindow(windowR, size);
    
    CCSImage<T>	lp(size, size);
    const T	u2 = width() / 2, v2 = height() / 2,
		u1 = width() - 1, v1 = height() - 1;
    const T	base = pow(T(lp.width() - 1), T(1)/T(lp.width() - 1));
    const T	step = T(M_PI) / T(lp.height());
    for (u_int t = 0; t < lp.height(); ++t)
    {
	line_type&	line = lp[t];
	const T		ang = t * step;
	Point2<T>	p(cos(ang), sin(ang));

	for (u_int r = 0; r < lp.width(); ++r)
	{
	    Point2<T>	q(p[0] + u2, p[1] + v2);

	    if (0 <= q[0] && q[0] <= u1 && 0 <= q[1] && q[1] <= v1)
	      //line[r] = at(q);
	      //line[r] = r * at(q);
		line[r] = windowR[r] * at(q);
	      //line[r] = log(T(1) + at(q));
	    p *= base;
	}
    }
#if _DEBUG >= 1
    Image<T>	tmp(lp);
    tmp *= T(255) / lp.maximum();
    tmp.save(cout, ImageBase::FLOAT);
    cerr << "CCSImage<T>::logpolar()..." << endl;
#endif
    return lp;
}

//! このCCS配列が空間領域にあって原点が配列中央にあるとき，半径方向に積分したpolar表現を返す．
/*!
  \return		半径方向に積分されpolar座標系で表現された1次元配列
*/
template <class T> CCSImageLine<T>
CCSImage<T>::intpolar() const
{
    using namespace	std;

    const u_int		size = max(width(), height());

    static Array<T>	windowR;
    initializeHanningWindow(windowR, size);

    CCSImageLine<T>	ip(size);
    const T		u2 = width() / 2, v2 = height() / 2,
			u1 = width() - 1, v1 = height() - 1;
    const T		base = pow(T(ip.size() - 1), T(1)/T(ip.size() - 1));
    const T		step = T(M_PI) / T(ip.size());
    for (u_int t = 0; t < ip.size(); ++t)
    {
	T&		pix = ip[t];
	const T		ang = t * step;
	const Point2<T>	p(cos(ang), sin(ang));
	for (u_int r = 1; r < ip.size(); ++r)
	{
	    Point2<T>	q(r*p[0] + u2, r*p[1] + v2);
	    if (0 <= q[0] && q[0] <= u1 && 0 <= q[1] && q[1] <= v1)
	      //pix += at(q);
	      //pix += r * at(q);
		pix += windowR[r] * at(q);
	      //pix += log(T(1) + at(q));
	}
    }
#if _DEBUG >= 1
    T		maxVal = ip.maximum();
    Image<T>	tmp(50, ip.size());
    for (u_int t = 0; t < tmp.height(); ++t)
	tmp[t] = ip[t] * T(255) / maxVal;
    tmp.save(cout, ImageBase::FLOAT);
    cerr << "CCSImage<T>::intpolar()..." << endl;
#endif
    return ip;
}
    
//! このCCS配列が周波数領域にあるとき，与えられたもうひとつのCCS配列との位相差に変換する．
/*!
  \param spectrum	周波数領域にあるCCS配列	
  \return		specturmとの位相差に変換した後のこの配列
*/
template <class T> CCSImage<T>&
CCSImage<T>::pdiff(const CCSImage<T>& spectrum)
{
    using namespace	std;

    check_size(spectrum.size());
    
  // 1. CCS行列の最初と最後の列以外の積の位相を求める．
    iterator	l = begin();
    for (const_iterator m = spectrum.begin(); m != spectrum.end(); ++m)
    {
	pixel_iterator	p = (l++)->begin() + 1;
	for (const_pixel_iterator q  = m->begin() + 1,
				  qe = m->end()	  - 1; q != qe; q += 2)
	{
	    complex<T>	val = complex<T>(*p, *(p+1)) * complex<T>(*q, -*(q+1));
	    if (val != T(0))
		val /= abs(val);
	    *p++ = val.real();
	    *p++ = val.imag();
	}
    }

  // 2. CCS行列の最初と最後の列（最初と最後の行以外）の積の位相を求める．
    const u_int	u1 = width() - 1;
    l = begin() + 1;
    for (const_iterator m  = spectrum.begin() + 1;
			m != spectrum.end()   - 1; m += 2)
    {
      // 最初の列の要素の実部と虚部を含む 2x1 部分行列をとって積を求める．
	complex<T>	val = complex<T>((*l)[0],  (*(l+1))[0])
			    * complex<T>((*m)[0], -(*(m+1))[0]);
	if (val != T(0))
	    val /= abs(val);
	(*l)[0]     = val.real();
	(*(l+1))[0] = val.imag();
	
      // 最後の列の要素の実部と虚部を含む 2x1 部分行列をとって積の位相を求める．
	val = complex<T>((*l)[u1],  (*(l+1))[u1])
	    * complex<T>((*m)[u1], -(*(m+1))[u1]);
	if (val != T(0))
	    val /= abs(val);
	(*l++)[u1] = val.real();
	(*l++)[u1] = val.imag();
    }

  // 3. CCS行列の四隅の積の位相を求める．
    const u_int	v1 = height() - 1;
    setSign((*this)[0][0]   *= spectrum[0][0]);
    setSign((*this)[0][u1]  *= spectrum[0][u1]);
    setSign((*this)[v1][0]  *= spectrum[v1][0]);
    setSign((*this)[v1][u1] *= spectrum[v1][u1]);

    return *this;
}
    
//! このCCS配列が周波数領域にあるとき，これに別のCCS配列の複素共役を掛ける．
/*!
  \param spectrum	周波数領域にあるCCS配列	
  \return		specturmの複素共役を掛けた後のこの配列
*/
template <class T> CCSImage<T>&
CCSImage<T>::operator *=(const CCSImage<T>& spectrum)
{
    using namespace	std;

    check_size(spectrum.size());
    
  // 1. CCS行列の最初と最後の列以外の積を求める．
    iterator	l = begin();
    for (const_iterator m = spectrum.begin(); m != spectrum.end(); ++m)
    {
	pixel_iterator	p = (l++)->begin() + 1;
	for (const_pixel_iterator q  = m->begin() + 1,
				  qe = m->end()	  - 1; q != qe; q += 2)
	{
	    complex<T>	val = complex<T>(*p, *(p+1)) * complex<T>(*q, -*(q+1));
	    *p++ = val.real();
	    *p++ = val.imag();
	}
    }

  // 2. CCS行列の最初と最後の列（最初と最後の行以外）の積を求める．
    const u_int	u1 = width() - 1;
    l = begin() + 1;
    for (const_iterator m  = spectrum.begin() + 1;
			m != spectrum.end()   - 1; m += 2)
    {
      // 最初の列の要素の実部と虚部を含む 2x1 部分行列をとって積を求める．
	complex<T>	val = complex<T>((*l)[0],  (*(l+1))[0])
			    * complex<T>((*m)[0], -(*(m+1))[0]);
	(*l)[0]     = val.real();
	(*(l+1))[0] = val.imag();
	
      // 最後の列の要素の実部と虚部を含む 2x1 部分行列をとって積を求める．
	val = complex<T>((*l)[u1],  (*(l+1))[u1])
	    * complex<T>((*m)[u1], -(*(m+1))[u1]);
	(*l++)[u1] = val.real();
	(*l++)[u1] = val.imag();
    }

  // 3. CCS行列の四隅の積を求める．
    const u_int	v1 = height() - 1;
    (*this)[0][0]   *= spectrum[0][0];
    (*this)[0][u1]  *= spectrum[0][u1];
    (*this)[v1][0]  *= spectrum[v1][0];
    (*this)[v1][u1] *= spectrum[v1][u1];

    return *this;
}

//! このCCS配列が空間領域にあるとき，配列要素中の最大値とその位置を返す．
/*!
  \param pMax	最大値を与える要素の位置
  \return	最大値
*/
template <class T> T
CCSImage<T>::maximum(Point2<T>& pMax) const
{
    using namespace	std;
    
    T	valMax = numeric_limits<T>::min();

    for (u_int v = 0; v < height(); ++v)
    {
	const ImageLine<T>&	line = (*this)[v];
	for (u_int u = 0; u < width(); ++u)
	{
	    const T	val = line[u];
	    if (val > valMax)
	    {
		valMax	= val;
		pMax[0] = u;
		pMax[1] = v;
	    }
	}
    }
#if _DEBUG >= 3
    Image<T>	tmp(*this);
    tmp *= T(255) / maximum();
    tmp.save(cout, ImageBase::FLOAT);
    cerr << "CCSImage<T>::maximum()..." << endl;
#endif
#ifdef _DEBUG
    cerr << "CCSImage<T>::maximum(): " << valMax
	 << "@(" << pMax[0] << ", " << pMax[1] << ") in ["
	 << width() << 'x' << height() << ']' << endl;
#endif
    return valMax;
}
    
template <class T> T
CCSImage<T>::maximum() const
{
    using namespace	std;
    
    T	valMax = numeric_limits<T>::min();

    for (u_int v = 0; v < height(); ++v)
    {
	const ImageLine<T>&	line = (*this)[v];
	for (u_int u = 0; u < width(); ++u)
	{
	    const T	val = line[u];
	    if (val > valMax)
		valMax	= val;
	}
    }

    return valMax;
}
    
}
#endif	// !__TU_CCSIMAGE_H
