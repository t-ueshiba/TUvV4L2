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
 *  $Id: EdgeDetector.h,v 1.1 2008-08-07 07:26:48 ueshiba Exp $
 */
#ifndef	__TUEdgeDetector_h
#define	__TUEdgeDetector_h

#include "TU/Image++.h"

namespace TU
{
/************************************************************************
*  class EdgeDetector							*
************************************************************************/
//! エッジ検出を行うクラス
class EdgeDetector
{
  public:
    enum
    {
	TRACED	= 0x04,
	EDGE	= 0x02,					//!< 強いエッジ点
	WEAK	= 0x01					//!< 弱いエッジ点
    };
    
    EdgeDetector(float th_low=2.0, float th_high=5.0)			;
    
    EdgeDetector&	initialize(float th_low, float th_high)		;
    const EdgeDetector&
	strength(const Image<float>& edgeH,
		 const Image<float>& edgeV, Image<float>& out)	  const	;
    const EdgeDetector&
	direction4(const Image<float>& edgeH,
		   const Image<float>& edgeV, Image<u_char>& out) const	;
    const EdgeDetector&
	direction8(const Image<float>& edgeH,
		   const Image<float>& edgeV, Image<u_char>& out) const	;
    const EdgeDetector&
	suppressNonmaxima(const Image<float>& strength,
			  const Image<u_char>& direction,
			  Image<u_char>& out)			  const	;
    const EdgeDetector&
	zeroCrossing(const Image<float>& in, Image<u_char>& out)  const	;
    const EdgeDetector&
	zeroCrossing(const Image<float>& in,
		     const Image<float>& strength,
		     Image<u_char>& out)			  const	;
    const EdgeDetector&
	hysteresisThresholding(Image<u_char>& edge)		  const	;

  private:
    float		_th_low, _th_high;
};

//! エッジ検出器を生成する
/*!
  \param th_low		弱いエッジの閾値
  \param th_low		強いエッジの閾値
*/
inline
EdgeDetector::EdgeDetector(float th_low, float th_high)
{
    initialize(th_low, th_high);
}

//! エッジ検出の閾値を設定する
/*!
  \param th_low		弱いエッジの閾値
  \param th_low		強いエッジの閾値
  \return		このエッジ検出器自身
*/
inline EdgeDetector&
EdgeDetector::initialize(float th_low, float th_high)
{
    _th_low  = th_low;
    _th_high = th_high;

    return *this;
}
 
}

#endif	/* !__TUEdgeDetector_h */
