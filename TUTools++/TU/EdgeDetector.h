/*!
  \file		EdgeDetector.h
  \author	Toshio UESHIBA
  \brief	クラス TU::EdgeDetector の定義と実装
*/
#ifndef	TU_EDGEDETECTOR_H
#define	TU_EDGEDETECTOR_H

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
	TRACED	= 0x04,		//!< トレース済みのエッジ点(内部的使用のみ)
	EDGE	= 0x02,		//!< 強いエッジ点
	WEAK	= 0x01		//!< 弱いエッジ点
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
	direction4x(const Image<float>& edgeH,
		    const Image<float>& edgeV, Image<u_char>& out) const;
    const EdgeDetector&
	direction8(const Image<float>& edgeH,
		   const Image<float>& edgeV, Image<u_char>& out) const	;
    const EdgeDetector&
	direction8x(const Image<float>& edgeH,
		    const Image<float>& edgeV, Image<u_char>& out) const;
    const EdgeDetector&
	ridge(const Image<float>& edgeHH,
	      const Image<float>& edgeHV, const Image<float>& edgeVV,
	      Image<float>& strength, Image<u_char>& direction)	  const	;
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
  \param th_high	強いエッジの閾値
*/
inline
EdgeDetector::EdgeDetector(float th_low, float th_high)
{
    initialize(th_low, th_high);
}

//! エッジ検出の閾値を設定する
/*!
  \param th_low		弱いエッジの閾値
  \param th_high	強いエッジの閾値
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
#endif	// !TU_EDGEDETECTOR_H
