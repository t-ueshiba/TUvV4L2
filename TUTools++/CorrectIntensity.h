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
 *  $Id: CorrectIntensity.h,v 1.4 2009-07-31 07:04:44 ueshiba Exp $
 */
#ifndef	__TUCorrectIntensity_h
#define	__TUCorrectIntensity_h

#include "TU/Image++.h"

namespace TU
{
/************************************************************************
*  class CorrectIntensity						*
************************************************************************/
//! 画像の線形輝度補正を行うクラス
class CorrectIntensity
{
  public:
  //! オフセットとゲインを指定して輝度補正オブジェクトを生成する．
  /*!
    \param offset	オフセット
    \param gain		ゲイン
  */
    CorrectIntensity(float offset=0.0, float gain=1.0)
	:_offset(offset), _gain(gain) 					{}

    void	initialize(float offset, float gain)			;
    template <class T>
    void	operator()(Image<T>& image, int vs=0, int ve=0)	const	;
    
  private:
    template <class T>
    T		val(T pixel)					const	;
    
    float	_offset;
    float	_gain;
};

//! オフセットとゲインを指定して輝度補正オブジェクトを初期化する．
/*!
  \param offset		オフセット
  \param gain		ゲイン
*/
inline void
CorrectIntensity::initialize(float offset, float gain)
{
    _offset = offset;
    _gain   = gain;
}

}

#endif	/* !__TUCorrectIntensity_h */
