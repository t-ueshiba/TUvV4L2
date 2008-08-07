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
 *  $Id: CorrectIntensity.h,v 1.1 2008-08-07 11:45:05 ueshiba Exp $
 */
#ifndef	__TUCorrectIntensity_h
#define	__TUCorrectIntensity_h

#include "TU/Image++.h"

namespace TU
{
/************************************************************************
*  class CorrectIntensity						*
************************************************************************/
class CorrectIntensity
{
  public:
    CorrectIntensity(float gain=1.0, float offset=0.0)
	:_gain(gain), _offset(offset)					{}

    void	initialize(float gain, float offset)			;
    template <class T>
    void	operator()(Image<T>& image, int vs=0, int ve=0)	const	;
    
  private:
    template <class T>
    T		val(T pixel)					const	;
    
    float	_gain, _offset;
};

inline void
CorrectIntensity::initialize(float gain, float offset)
{
    _gain   = gain;
    _offset = offset;
}

}

#endif	/* !__TUCorrectIntensity_h */
