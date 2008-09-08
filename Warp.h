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
 *  $Id: Warp.h,v 1.2 2008-09-08 08:06:23 ueshiba Exp $
 */
#ifndef	__TUWarp_h
#define	__TUWarp_h

#include "TU/Image++.h"
#include "TU/Camera.h"

namespace TU
{
/************************************************************************
*  class Warp								*
************************************************************************/
//! 画像を変形するためのクラス．
class Warp
{
  private:
    struct FracArray
    {
	FracArray(u_int d=0)
	    :us(d), vs(d), du(d), dv(d), lmost(0)	{}

	u_int		width()			const	{return us.dim();}
	void		resize(u_int d)			;

	Array<short>				us, vs;
#ifdef __INTEL_COMPILER	
	Array<u_char, AlignedBuf<u_char> >	du, dv;
#else
	Array<u_char>				du, dv;
#endif
	int					lmost;
    };
    
  public:
    Warp()	:_fracs(), _width(0)			{}

    u_int	width()				const	{return _width;}
    u_int	height()			const	{return _fracs.dim();}
    int		lmost(int v)			const	;
    int		rmost(int v)			const	;

    void	initialize(const Matrix<double>& Htinv,
			   u_int inWidth,  u_int inHeight,
			   u_int outWidth, u_int outHeight)		;
    void	initialize(const Matrix<double>& Htinv,
			   const CameraBase::Intrinsic& intrinsic,
			   u_int inWidth,  u_int inHeight,
			   u_int outWidth, u_int outHeight)		;
    template <class T>
    void	operator ()(const Image<T>& in, Image<T>& out,
			    int vs=0, int ve=0)			const	;
    Vector2f	operator ()(int u, int v)			const	;

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

inline int
Warp::lmost(int v) const
{
    return _fracs[v].lmost;
}

inline int
Warp::rmost(int v) const
{
    return _fracs[v].lmost + _fracs[v].width();
}

inline Vector2f
Warp::operator ()(int u, int v) const
{
    Vector2f		val;
    const FracArray&	fracs = _fracs[v];
    val[0] = float(fracs.us[u]) + float(fracs.du[u]) / 128.0;
    val[1] = float(fracs.vs[u]) + float(fracs.dv[u]) / 128.0;
    return val;
}

}

#endif	/* !__TUWarp_h */
