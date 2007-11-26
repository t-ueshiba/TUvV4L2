/*
 *  平成19年（独）産業技術総合研究所 著作権所有
 *  
 *  創作者：植芝俊夫
 *
 *  本プログラムは（独）産業技術総合研究所の職員である植芝俊夫が創作し，
 *  同所が著作権を所有する秘密情報です．著作者による許可なしにこのプロ
 *  グラムを第三者へ開示，複製，改変，使用する等の著作権を侵害する行為
 *  を禁止します．
 *  
 *  このプログラムによって生じるいかなる損害に対しても、著作者は責任
 *  を負いません。 
 *
 *  Copyright 2007
 *  National Institute of Advanced Industrial Science and Technology (AIST)
 *
 *  Author: Toshio UESHIBA
 *
 *  Confidentail and all rights reserved.
 *  This program is confidential. Any changing, copying or giving
 *  information about the source code of any part of this software
 *  and/or documents without permission by the authors are prohibited.
 *
 *  No Warranty.
 *  Authors are not responsible for any damages in the use of this program.
 *  
 *  $Id: Rotation.cc,v 1.8 2007-11-26 07:28:09 ueshiba Exp $
 */
#include "TU/Vector++.h"

namespace TU
{
static inline double	sqr(double x)	{return x * x;}
    
//! 2次元超平面内での回転を生成する
/*!
  \param p	p軸を指定するindex
  \param q	q軸を指定するindex
  \param x	回転角を生成する際のx値
  \param y	回転角を生成する際のy値
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
  \param p	p軸を指定するindex
  \param q	q軸を指定するindex
  \param theta	回転角
*/
Rotation::Rotation(int p, int q, double theta)
    :_p(p), _q(q), _l(1.0), _c(::cos(theta)), _s(::sin(theta))
{
}
 
}
