/*
 *  $Id: Random.cc,v 1.1.1.1 2002-07-25 02:14:16 ueshiba Exp $
 */
#include <time.h>
#include <cmath>
#include "TU/Random.h"
#include <stdexcept>

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

inline long
congruence(long x, long a, long c, long m)
{
    return (a * x + c) % m;
}

/************************************************************************
*  class Random							*
************************************************************************/
Random::Random()
    :_seed(-int(time(0))), _x1(0), _x2(0), _x3(0), _ff(0),
     _has_extra(0), _extra(0.0)
{
#ifdef __APPLE__
    srandom(-_seed);
#else
    srand48(-_seed);
#endif
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
#ifdef __APPLE__
    return random();
#else
    return drand48();
#endif
}

double
Random::gaussian(double (Random::*uni)())
{
    if (!_has_extra)
    {
	double	v0, v1, r;
	do
	{
	    v0 = 2.0 * (this->*uni)() - 1.0;		// -1 <= v0 < 1
	    v1 = 2.0 * (this->*uni)() - 1.0;		// -1 <= v1 < 1
	    r  = v0*v0 + v1*v1;
	} while (r >= 1.0);
	double	fac = std::sqrt(-2.0 * std::log(r) / r);
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
