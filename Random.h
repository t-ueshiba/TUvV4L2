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
 *  $Id: Random.h,v 1.3 2007-11-26 07:28:09 ueshiba Exp $
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
