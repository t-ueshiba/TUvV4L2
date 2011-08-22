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
 *  $Id: IntegralImage.h,v 1.7 2011-08-22 00:06:25 ueshiba Exp $
 */
/*!
  \file		IntegralImage.h
  \brief	クラス TU::IntegralImage, TU::DiagonalIntegralImage の定義と実装
*/
#ifndef	__TUIntegralImage_h
#define	__TUIntegralImage_h

#include "TU/Image++.h"

namespace TU
{
/************************************************************************
*  class IntegralImage<T>						*
************************************************************************/
//! 積分画像(integral image)を表すクラス
template <class T>
class IntegralImage : public Image<T>
{
  public:
    IntegralImage()							;
    template <class S, class B>
    IntegralImage(const Image<S, B>& image)				;

    template <class S, class B> IntegralImage&
		initialize(const Image<S, B>& image)			;
    T		crop(int u, int v, int w, int h)		const	;
    T		crop2(int umin, int umax, int vmin, int vmax)	const	;
    T		crossVal(int u, int v, int cropSize)		const	;
    template <class S, class B> const IntegralImage&
		crossVal(Image<S, B>& out, int cropSize)	const	;

    u_int	originalWidth()					const	;
    u_int	originalHeight()				const	;
    
    using	Image<T>::width;
    using	Image<T>::height;
};

//! 空の積分画像を作る
template <class T> inline
IntegralImage<T>::IntegralImage()
{
}
    
//! 与えられた画像から積分画像を作る
/*!
  \param image		入力画像
*/
template <class T> template <class S, class B> inline
IntegralImage<T>::IntegralImage(const Image<S, B>& image)
{
    initialize(image);
}
    
//! 与えられた画像から積分画像を作る
/*!
  \param image		入力画像
  \return		この積分画像
*/
template <class T> template <class S, class B> IntegralImage<T>&
IntegralImage<T>::initialize(const Image<S, B>& image)
{
  // 原画像よりも上と左に１ピクセルずつ大きいサイズを用意
    resize(image.height() + 1, image.width() + 1);

  // 上と左に余白を入れる
    (*this)[0] = 0;				// 0行目はすべて0
    for (u_int v = 1; v < height(); ++v)
    {
	T*		dst = (*this)[v];
	*dst = 0;				// 0列目は0

	const S*	src = image[v-1];
	T		val = 0;		// この行のこの画素までの和
	const T*	prv = (*this)[v-1];	// 1行上
	const T* const	end = dst + width();
	while (++dst < end)
	    *dst = (val += *src++) + *(++prv);
    }

    return *this;
}

//! 原画像に設定した長方形ウィンドウ内の画素値の総和を返す
/*!
  \param u		ウィンドウの左上隅の横座標
  \param v		ウィンドウの左上隅の縦座標
  \param w		ウィンドウの幅
  \param h		ウィンドウの高さ
  \return		ウィンドウ内の画素値の総和
*/
template <class T> inline T
IntegralImage<T>::crop(int u, int v, int w, int h) const
{
    const int	u1 = std::min(u + w, int(width())  - 1),
		v1 = std::min(v + h, int(height()) - 1);
    if (u >= int(width()) || v >= int(height()) || u1 < 0 || v1 < 0)
	return 0;
    if (u < 0)
	u = 0;
    if (v < 0)
	v = 0;
    
    return (*this)[v1][u1] + (*this)[v ][u]
	 - (*this)[v ][u1] - (*this)[v1][u];
}

//! 原画像に設定した長方形ウィンドウ内の画素値の総和を返す
/*!
  \param umin		ウィンドウの左上隅の横座標
  \param umax		ウィンドウの右下隅の横座標
  \param vmin		ウィンドウの左上隅の縦座標
  \param vmax		ウィンドウの右下隅の縦座標
  \return		ウィンドウ内の画素値の総和
*/
template <class T> inline T
IntegralImage<T>::crop2(int umin, int umax, int vmin, int vmax) const
{
    ++umax;
    ++vmax;

    return (*this)[vmax][umax] + (*this)[vmin][umin]
	 - (*this)[vmin][umax] - (*this)[vmax][umin];
}

//! 原画像に正方形の二値十字テンプレートを適用した値を返す
/*!
  \param u		テンプレート中心の横座標
  \param v		テンプレート中心の縦座標
  \param cropSize	テンプレートは一辺 2*cropSize + 1 の正方形
  \return		テンプレートを適用した値
*/
template <class T> inline T
IntegralImage<T>::crossVal(int u, int v, int cropSize) const
{
    return crop(u+1,	    v+1,	cropSize, cropSize)
	 - crop(u-cropSize, v+1,	cropSize, cropSize)
	 + crop(u-cropSize, v-cropSize, cropSize, cropSize)
	 - crop(u+1,	    v-cropSize, cropSize, cropSize);
}
    
//! 原画像の全ての点に正方形の二値十字テンプレートを適用した画像を求める
/*!
  \param out		原画像にテンプレートを適用した出力画像
  \param cropSize	テンプレートサイズを指定するパラメータ
			テンプレートは一辺 2*cropSize+1 の正方形
  \return		この積分画像
*/
template <class T> template <class S, class B> const IntegralImage<T>&
IntegralImage<T>::crossVal(Image<S, B>& out, int cropSize) const
{
    out.resize(originalHeight(), originalWidth());
    for (u_int v = 0; v < out.height(); ++v)
	for (u_int u = 0; u < out.width(); ++u)
	    out[v][u] = crossVal(u, v, cropSize);

    return *this;
}

//! 原画像の幅を返す
/*!
  積分画像自体の幅と高さはそれぞれ原画像よりも1だけ大きいのでこの分を減じた値を返す．
  \return	原画像の幅
*/
template <class T> u_int
IntegralImage<T>::originalWidth() const
{
    return width() - 1;
}

//! 原画像の高さを返す
/*!
  積分画像自体の幅と高さはそれぞれ原画像よりも1だけ大きいのでこの分を減じた値を返す．
  \return	原画像の高さ
*/
template <class T> u_int
IntegralImage<T>::originalHeight() const
{
    return height() - 1;
}

/************************************************************************
*  class DiagonalIntegralImage<T>					*
************************************************************************/
//! 対角積分画像(diagonal integral image)を表すクラス
template <class T>
class DiagonalIntegralImage : public Image<T>
{
  public:
    DiagonalIntegralImage()						;
    template <class S, class B>
    DiagonalIntegralImage(const Image<S, B>& image)			;

    template <class S, class B> DiagonalIntegralImage&
		initialize(const Image<S, B>& image)			;
    T		crop(int u, int v, int w, int h)		const	;
    T		crossVal(int u, int v, int cropSize)		const	;
    template <class S, class B> const DiagonalIntegralImage&
		crossVal(Image<S, B>& out, int cropSize)	const	;

    u_int	originalWidth()					const	;
    u_int	originalHeight()				const	;
    
    using	Image<T>::width;
    using	Image<T>::height;

  private:
    void	correct(int& u, int& v)				const	;
};

//! 空の対角積分画像を作る
template <class T> inline
DiagonalIntegralImage<T>::DiagonalIntegralImage()
{
}
    
//! 与えられた画像から対角積分画像を作る
/*!
  \param image		入力画像
*/
template <class T> template <class S, class B> inline
DiagonalIntegralImage<T>::DiagonalIntegralImage(const Image<S, B>& image)
{
    initialize(image);
}
    
//! 与えられた画像から対角積分画像を作る
/*!
  \param image		入力画像
  \return		この対角積分画像
*/
template <class T> template <class S, class B> DiagonalIntegralImage<T>&
DiagonalIntegralImage<T>::initialize(const Image<S, B>& image)
{
    resize(image.height(), image.width());
    
    Array<T>	K(width() + height() - 1), L(width() + height() - 1);
    for (u_int i = 0; i < K.dim(); ++i)
	K[i] = L[i] = 0;
    
    for (u_int v = 0; v < height(); ++v)
    {
	const S*	src = image[v];
	T		*dst = (*this)[v],
			*kp = &K[height() - 1 - v], *lp = &L[v];
	if (v == 0)
	    for (const T* const end = dst + width(); dst < end; )
		*dst++ = *kp++ = *lp++ = *src++;
	else
	{
	    const T*	prv = (*this)[v-1];
	    for (const T* const end = dst + width(); dst < end; )
	    {
		*dst++ = *src + *kp + *lp + *prv++;
		*kp++ += *src;
		*lp++ += *src++;
	    }
	}
    }

    return *this;
}

//! 原画像に45度傾けて設定した長方形ウィンドウ内の画素値の総和を返す
/*!
  \param u		ウィンドウの上隅の横座標
  \param v		ウィンドウの上隅の縦座標
  \param w		ウィンドウの幅
  \param h		ウィンドウの高さ
  \return		ウィンドウ内の画素値の総和
*/
template <class T> T
DiagonalIntegralImage<T>::crop(int u, int v, int w, int h) const
{
    --v;
    int		ul = u - h, vl = v + h, ur = u + w, vr = v + w,
		ut = u + w - h, vt = v + w + h;
    correct(u,  v);
    correct(ul, vl);
    correct(ur, vr);
    correct(ut, vt);
    if (vt >= int(height()))
	return 0;
    return (v  >= 0 ? (*this)[v][u]   : 0)
	 + (vt >= 0 ? (*this)[vt][ut] : 0)
	 - (vl >= 0 ? (*this)[vl][ul] : 0)
	 - (vr >= 0 ? (*this)[vr][ur] : 0);
}

//! 原画像に正方形の二値クロステンプレートを適用した値を返す
/*!
  \param u		テンプレート中心の横座標
  \param v		テンプレート中心の縦座標
  \param cropSize	テンプレートは一辺 2*cropSize + 1 の正方形
  \return		テンプレートを適用した値
*/
template <class T> inline T
DiagonalIntegralImage<T>::crossVal(int u, int v, int cropSize) const
{
    return crop(u+cropSize+1, v-cropSize+1, cropSize, cropSize)
	 - crop(u,	      v+2,	    cropSize, cropSize)
	 + crop(u-cropSize-1, v-cropSize+1, cropSize, cropSize)
	 - crop(u,	      v-2*cropSize, cropSize, cropSize);
}

//! 原画像の全ての点に正方形の二値クロステンプレートを適用した画像を求める
/*!
  \param out		原画像にテンプレートを適用した出力画像
  \param cropSize	テンプレートサイズを指定するパラメータ
			テンプレートは一辺 2*cropSize+1 の正方形
  \return		この対角積分画像
*/
template <class T> template <class S, class B> const DiagonalIntegralImage<T>&
DiagonalIntegralImage<T>::crossVal(Image<S, B>& out, int cropSize) const
{
    out.resize(height(), width());
    for (u_int v = 0; v < out.height() - 2*cropSize - 1; ++v)
	for (u_int u = 0; u < out.width(); ++u)
	    out[v][u] = crossVal(u, v, cropSize);

    return *this;
}

//! 原画像の幅を返す
/*!
  対角積分画像自体の幅と高さは原画像と同じである．
  \return	原画像の幅
*/
template <class T> u_int
DiagonalIntegralImage<T>::originalWidth() const
{
    return width();
}

//! 原画像の高さを返す
/*!
  対角積分画像自体の幅と高さは原画像と同じである．
  \return	原画像の高さ
*/
template <class T> u_int
DiagonalIntegralImage<T>::originalHeight() const
{
    return height();
}

template <class T> inline void
DiagonalIntegralImage<T>::correct(int& u, int& v) const
{
    if (u < 0)
    {
	v += u;
	u  = 0;
    }
    else if (u >= int(width()))
    {
	v += (int(width()) - 1 - u);
	u  = width() - 1;
    }
}

}

#endif	/* !__TUIntegralImage_h */
