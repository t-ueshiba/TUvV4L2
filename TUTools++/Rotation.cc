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
 *  $Id: Rotation.cc,v 1.3 2003-03-14 02:26:07 ueshiba Exp $
 */
#include "TU/Vector++.h"

namespace TU
{
Rotation::Rotation(int p, int q, double x, double y)
    :_p(p), _q(q), _c(1.0), _s(0.0)
{
    const double	absy = fabs(y);
    if (fabs(x) > absy)
    {
	const double	t = y / x;
	_c = 1 / sqrt(t*t + 1);
	_s = t * _c;
    }
    else if (absy != 0.0)
    {
	const double	t = x / y;
	_s = 1 / sqrt(t*t + 1);
	_c = t * _s;
    }
}

Rotation::Rotation(int p, int q, double theta)
    :_p(p), _q(q), _c(::cos(theta)), _s(::sin(theta))
{
}
 
}
