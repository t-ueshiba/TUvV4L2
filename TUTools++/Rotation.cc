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
 *  $Id: Rotation.cc,v 1.6 2006-12-19 07:09:24 ueshiba Exp $
 */
#include "TU/Vector++.h"

namespace TU
{
static inline double	sqr(double x)	{return x * x;}
    
//! 2次元超平面内での回転を生成する
/*!
  \param p	p軸を指定するindex.
  \param q	q軸を指定するindex.
  \param x	回転角を生成する際のx値.
  \param y	回転角を生成する際のy値.
		\f[
		  \cos\theta = \frac{x}{\sqrt{x^2+y^2}},{\hskip 1em}
		  \sin\theta = \frac{y}{\sqrt{x^2+y^2}}
		\f]
*/
Rotation::Rotation(int p, int q, double x, double y)
    :_p(p), _q(q), _l(1.0), _c(1.0), _s(0.0)
{
    const double	absx = fabs(x), absy = fabs(y);
    _l = (absx > absy ? absx * sqrt(1.0 + sqr(absy/absx))
		      : absy * sqrt(1.0 + sqr(absx/absy)));
    if (_l != 0.0)
    {
	_c = x / _l;
	_s = y / _l;
    }
}

//! 2次元超平面内での回転を生成する
/*!
  \param p	p軸を指定するindex.
  \param q	q軸を指定するindex.
  \param theta	回転角.
*/
Rotation::Rotation(int p, int q, double theta)
    :_p(p), _q(q), _l(1.0), _c(::cos(theta)), _s(::sin(theta))
{
}
 
}
