/*
 *  $Id: Random.h,v 1.1.1.1 2002-07-25 02:14:16 ueshiba Exp $
 */
namespace TU
{
/************************************************************************
*  class Random								*
************************************************************************/
class Random
{
  public:
    Random()						;
    
    double	uniform()				;
    double	gaussian()				;
    double	uniform48()				;
    double	gaussian48()				;
    
  private:
    double	gaussian(double (Random::*uni)())	;
    
    int		_seed;
    long	_x1, _x2, _x3;
    double	_r[97];
    int		_ff;
    int		_has_extra;	// flag showing existence of _extra.
    double	_extra;		// extra gaussian noise value.
};

inline double
Random::gaussian()
{
    return gaussian(&Random::uniform);
}

inline double
Random::gaussian48()
{
    return gaussian(&Random::uniform48);
}
 
}
