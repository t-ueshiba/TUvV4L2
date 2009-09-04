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
 *  $Id: Random.h,v 1.8 2009-09-04 04:01:06 ueshiba Exp $
 */
#include "TU/types.h"

namespace TU
{
/************************************************************************
*  class Random								*
************************************************************************/
class __PORT Random
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
