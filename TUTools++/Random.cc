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
 *  $Id: Random.cc,v 1.12 2009-09-09 07:06:31 ueshiba Exp $
 */
#include "TU/Random.h"
#include <cmath>
#include <cstdlib>
#include <stdexcept>
#ifdef WIN32
#  include "windows/fakeWindows.h"
#else
#  include <sys/time.h>
#endif

namespace TU
{
/************************************************************************
*  static constans & functions						*
************************************************************************/
static const int	M1  = 259200;
static const int	A1  = 7141;
static const int	C1  = 54773;
static const double	RM1 = (1.0 / M1);
static const int	M2  = 134456;
static const int	A2  = 8121;
static const int	C2  = 28411;
static const double	RM2 = (1.0 / M2);
static const int	M3  = 243000;
static const int	A3  = 4561;
static const int	C3  = 51349;

static inline long
seed()
{
    timeval tv;
    gettimeofday(&tv, 0);
    return -tv.tv_sec;
}

static inline long
congruence(long x, long a, long c, long m)
{
    return (a * x + c) % m;
}

/************************************************************************
*  class Random								*
************************************************************************/
Random::Random()
    :_seed(seed()), _x1(0), _x2(0), _x3(0), _ff(0),
     _has_extra(0), _extra(0.0)
{
    srand48(-_seed);
}

double
Random::uniform()
{
    if (_seed < 0 || _ff == 0)
    {
	_ff = 1;

	_x1 = (C1 - _seed) % M1;		// seed the first routine.
	_x1 = congruence(_x1, A1, C1, M1);
	_x2 = _x1 % M2;				// seed the second routine.
	_x1 = congruence(_x1, A1, C1, M1);
	_x3 = _x1 % M3;				// seed the third routien.
	for (int j = 0 ; j < 97; ++j)
	{
	    _x1 = congruence(_x1, A1, C1, M1);
	    _x2 = congruence(_x2, A2, C2, M2);
	    _r[j] = (_x1 + _x2 * RM2) * RM1;	// fill table.
	}
	_seed = 1;
    }
    _x1 = congruence(_x1, A1, C1, M1);
    _x2 = congruence(_x2, A2, C2, M2);
    _x3 = congruence(_x3, A3, C3, M3);
    int j = (97 * _x3) / M3;
    if (j < 0 || j >= 97)
	throw
	  std::runtime_error("TU::Random::normal: unexpected integer value!!");
    double	tmp = _r[j];
    _r[j] = (_x1 + _x2 * RM2) * RM1;		// refill table.
    
    return tmp;
}

double
Random::uniform48()
{
    return drand48();
}

double
Random::gaussian(double (Random::*uni)())
{
    using namespace	std;
    
    if (!_has_extra)
    {
	double	v0, v1, r;
	do
	{
	    v0 = 2.0 * (this->*uni)() - 1.0;		// -1 <= v0 < 1
	    v1 = 2.0 * (this->*uni)() - 1.0;		// -1 <= v1 < 1
	    r  = v0*v0 + v1*v1;
	} while (r >= 1.0);
	double	fac = sqrt(-2.0 * log(r) / r);
	_extra = v0 * fac;
	_has_extra = 1;
	return v1 * fac;
    }
    else
    {
	_has_extra = 0;
	return _extra;
    }
}
 
}
