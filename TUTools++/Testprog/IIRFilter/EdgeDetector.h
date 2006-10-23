/*
 *  $Id: EdgeDetector.h,v 1.1 2006-10-23 01:22:31 ueshiba Exp $
 */
#ifndef __TUEdgeDetector_h
#define __TUEdgeDetector_h

#include "TU/Image++.h"

namespace TU
{
/************************************************************************
*  class EdgeDetector							*
************************************************************************/
/*!
  エッジ検出を行うクラス.
*/ 
class EdgeDetector
{
  public:
    enum	{TRACED = 0x4, EDGE = 0x02, WEAK = 0x01};
    
    EdgeDetector(float th_low=10.0, float th_high=20.0)			;
    
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

  private:
    float		_th_low, _th_high;
};

//! エッジ検出器を生成する
/*!
  \param th_low		弱いエッジの閾値.
  \param th_low		強いエッジの閾値.
*/
inline
EdgeDetector::EdgeDetector(float th_low, float th_high)
{
    initialize(th_low, th_high);
}

//! エッジ検出の閾値を設定する
/*!
  \param th_low		弱いエッジの閾値.
  \param th_low		強いエッジの閾値.
  \return		このエッジ検出器自身.
*/
inline EdgeDetector&
EdgeDetector::initialize(float th_low, float th_high)
{
    _th_low  = th_low;
    _th_high = th_high;

    return *this;
}
    
}

#endif	// !__TUEdgeDetector_h
