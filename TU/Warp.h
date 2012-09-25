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
  \file		Warp.h
  \brief	クラス TU::Warp の定義と実装
*/
#ifndef	__TUWarp_h
#define	__TUWarp_h

#include "TU/Image++.h"
#include "TU/Camera++.h"
#include "TU/mmInstructions.h"
#if defined(USE_TBB)
#  include <tbb/parallel_for.h>
#  include <tbb/blocked_range.h>
#endif

namespace TU
{
/************************************************************************
*  class Warp								*
************************************************************************/
//! 画像を変形するためのクラス
class Warp
{
  private:
    struct FracArray
    {
	FracArray(u_int d=0)
	    :us(d), vs(d), du(d), dv(d), lmost(0)	{}

	u_int		width()			const	{return us.dim();}
	void		resize(u_int d)			;

#if defined(__INTEL_COMPILER)
	Array<short,  AlignedBuf<short> >	us, vs;
	Array<u_char, AlignedBuf<u_char> >	du, dv;
#else
	Array<short>				us, vs;
	Array<u_char>				du, dv;
#endif
	int					lmost;
    };

#if defined(USE_TBB)
    template <class T>
    class WarpLine
    {
      public:
	WarpLine(const Warp& warp, const Image<T>& in, Image<T>& out)
	    :_warp(warp), _in(in), _out(out)				{}
	
	void	operator ()(const tbb::blocked_range<u_int>& r) const
		{
		    for (u_int v = r.begin(); v != r.end(); ++v)
			_warp.warpLine(_in, _out, v);
		}

      private:
	const Warp&	_warp;
	const Image<T>&	_in;
	Image<T>&	_out;
    };
#endif
    
  public:
  //! 画像変形オブジェクトを生成する．
    Warp()	:_fracs(), _width(0)			{}

  //! 出力画像の幅を返す．
  /*!
    return	出力画像の幅
  */
    u_int	width()				const	{return _width;}

  //! 出力画像の高さを返す．
  /*!
    return	出力画像の高さ
  */
    u_int	height()			const	{return _fracs.dim();}
    
    int		lmost(int v)			const	;
    int		rmost(int v)			const	;

    template <class T>
    void	initialize(const Matrix<T, FixedSizedBuf<T, 9>,
			   FixedSizedBuf<Vector<T>, 3> >& Htinv,
			   u_int inWidth,  u_int inHeight,
			   u_int outWidth, u_int outHeight)		;
    template <class I>
    void	initialize(const typename I::matrix33_type& Htinv,
			   const I& intrinsic,
			   u_int inWidth,  u_int inHeight,
			   u_int outWidth, u_int outHeight)		;
    template <class T>
    void	operator ()(const Image<T>& in, Image<T>& out)	const	;
    Vector2f	operator ()(int u, int v)			const	;
#if defined(SSE2)
    mm::F32vec	src(int u, int v)				const	;
#endif

  private:
    template <class T>
    void	warpLine(const Image<T>& in,
			 Image<T>& out, u_int v)		const	;
    
  private:
    Array<FracArray>	_fracs;
    u_int		_width;
};

inline void
Warp::FracArray::resize(u_int d)
{
    us.resize(d);
    vs.resize(d);
    du.resize(d);
    dv.resize(d);
}

//! 出力画像における指定された行の有効左端位置を返す．
/*!
  入力画像が矩形でも出力画像も矩形とは限らないので，出力画像の一部しか
  入力画像の値域(有効領域)とならない．本関数は，出力画像の指定された行
  について，その有効領域の左端となる画素位置を返す．
  \param v	行を指定するintex
  \return	左端位置
*/
inline int
Warp::lmost(int v) const
{
    return _fracs[v].lmost;
}

//! 出力画像における指定された行の有効右端位置の次を返す．
/*!
  入力画像が矩形でも出力画像も矩形とは限らないので，出力画像の一部しか
  入力画像の値域(有効領域)とならない．本関数は，出力画像の指定された行
  について，その有効領域の右端の右隣となる画素位置を返す．
  \param v	行を指定するintex
  \return	右端位置の次
*/
inline int
Warp::rmost(int v) const
{
    return _fracs[v].lmost + _fracs[v].width();
}

//! 画像を射影変換するための行列を設定する．
/*!
  入力画像点uは射影変換
  \f[
    \TUbeginarray{c} \TUvec{v}{} \\ 1 \TUendarray \simeq
    \TUvec{H}{} \TUbeginarray{c} \TUvec{u}{} \\ 1 \TUendarray
  \f]
  によって出力画像点vに写される．
  \param Htinv		変形を指定する3x3射影変換行列の逆行列の転置，すなわち
			\f$\TUtinv{H}{}\f$
  \param inWidth	入力画像の幅
  \param inHeight	入力画像の高さ
  \param outWidth	出力画像の幅
  \param outHeight	出力画像の高さ
*/
template <class T> inline void
Warp::initialize(const Matrix<T, FixedSizedBuf<T, 9>,
			      FixedSizedBuf<Vector<T>, 3> >& Htinv,
		 u_int inWidth,  u_int inHeight,
		 u_int outWidth, u_int outHeight)
{
    initialize(Htinv, IntrinsicBase<T>(),
	       inWidth, inHeight, outWidth, outHeight);
}

//! 画像の非線形歪みを除去した後に射影変換を行うための行列とカメラ内部パラメータを設定する．
/*!

  canonical座標xから画像座標uへの変換が\f$\TUvec{u}{} = {\cal
  K}(\TUvec{x}{})\f$ と表されるカメラ内部パラメータについて，その線形変
  換部分を表す3x3上半三角行列をKとすると，
  \f[
    \TUbeginarray{c} \TUbar{u}{} \\ 1 \TUendarray =
    \TUvec{K}{}
    \TUbeginarray{c} {\cal K}^{-1}(\TUvec{u}{}) \\ 1 \TUendarray
  \f]
  によって画像の非線形歪みだけを除去できる．本関数は，この歪みを除去した画像点を
  射影変換Hによって出力画像点vに写すように変形パラメータを設定する．すなわち，
  全体の変形は
  \f[
    \TUbeginarray{c} \TUvec{v}{} \\ 1 \TUendarray \simeq
    \TUvec{H}{}\TUvec{K}{}
    \TUbeginarray{c} {\cal K}^{-1}(\TUvec{u}{}) \\ 1 \TUendarray
  \f]
  となる．
  \param Htinv		変形を指定する3x3射影変換行列の逆行列の転置
  \param intrinsic	入力画像に加えれられている放射歪曲を表すカメラ内部パラメータ
  \param inWidth	入力画像の幅
  \param inHeight	入力画像の高さ
  \param outWidth	出力画像の幅
  \param outHeight	出力画像の高さ
*/
template <class I> void
Warp::initialize(const typename I::matrix33_type& Htinv, const I& intrinsic,
		 u_int inWidth,  u_int inHeight,
		 u_int outWidth, u_int outHeight)
{
    typedef I						intrinsic_type;
    typedef typename intrinsic_type::point2_type	point2_type;
    typedef typename intrinsic_type::vector_type	vector_type;
    typedef typename intrinsic_type::matrix_type	matrix_type;
    
    _fracs.resize(outHeight);
    _width = outWidth;
    
  /* Compute frac for each pixel. */
    const matrix_type&	HKtinv = Htinv * intrinsic.Ktinv();
    vector_type		leftmost = HKtinv[2];
    for (u_int v = 0; v < height(); ++v)
    {
	vector_type	x = leftmost;
	FracArray	frac(width());
	u_int		n = 0;
	for (u_int u = 0; u < width(); ++u)
	{
	    const point2_type&	m = intrinsic.u(point2_type(x[0]/x[2],
							    x[1]/x[2]));
	    if (0.0 <= m[0] && m[0] <= inWidth - 2 &&
		0.0 <= m[1] && m[1] <= inHeight - 2)
	    {
		if (n == 0)
		    frac.lmost = u;
		frac.us[n] = (short)floor(m[0]);
		frac.vs[n] = (short)floor(m[1]);
		frac.du[n] = (u_char)floor((m[0] - floor(m[0])) * 128.0);
		frac.dv[n] = (u_char)floor((m[1] - floor(m[1])) * 128.0);
		++n;
	    }
	    x += HKtinv[0];
	}

	_fracs[v].resize(n);
	_fracs[v].lmost = frac.lmost;
	for (u_int u = 0; u < n; ++u)
	{
	    _fracs[v].us[u] = frac.us[u];
	    _fracs[v].vs[u] = frac.vs[u];
	    _fracs[v].du[u] = frac.du[u];
	    _fracs[v].dv[u] = frac.dv[u];
	}

	leftmost += HKtinv[1];
    }
}

//! 出力画像の範囲を指定して画像を変形する．
/*!
  \param in	入力画像
  \param out	出力画像
*/
template <class T> void
Warp::operator ()(const Image<T>& in, Image<T>& out) const
{
    out.resize(height(), width());

#if defined(USE_TBB)
    using namespace	tbb;

    parallel_for(blocked_range<u_int>(0, out.height(), 1),
		 WarpLine<T>(*this, in, out));
#else
    for (u_int v = 0; v < out.height(); ++v)
	warpLine(in, out, v);
#endif
}
    
//! 出力画像点を指定してそれにマップされる入力画像点の2次元座標を返す．
/*!
  \param u	出力画像点の横座標
  \param v	出力画像点の縦座標
  \return	出力画像点(u, v)にマップされる入力画像点の2次元座標
*/
inline Vector2f
Warp::operator ()(int u, int v) const
{
    Vector2f		val;
    const FracArray&	fracs = _fracs[v];
    val[0] = float(fracs.us[u]) + float(fracs.du[u]) / 128.0f;
    val[1] = float(fracs.vs[u]) + float(fracs.dv[u]) / 128.0f;
    return val;
}

#if defined(SSE2)
//! 2つの出力画像点を指定してそれぞれにマップされる2つの入力画像点の2次元座標を返す．
/*!
  指定された2次元座標(u, v)に対し，2点(u, v-1), (u, v)にマップされる入力画像点の
  2次元座標が返される．
  \param u	出力画像点の横座標
  \param v	出力画像点の縦座標
  \return	出力画像点(u, v-1), (u, v)にマップされる入力画像点の2次元座標
*/
inline mm::F32vec
Warp::src(int u, int v) const
{
    using namespace	mm;
    
    const FracArray	&fp = _fracs[v-1], &fc = _fracs[v];
    const Is16vec	tmp(fc.dv[u], fc.du[u], fp.dv[u], fp.du[u],
			    fc.vs[u], fc.us[u], fp.vs[u], fp.us[u]);
    return cvt<float>(tmp) + cvt<float>(shift_r<4>(tmp)) / F32vec(128.0);
}
#endif
}

#endif	/* !__TUWarp_h */
