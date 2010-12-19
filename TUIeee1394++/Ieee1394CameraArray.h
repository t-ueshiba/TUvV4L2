/*
 *  $Id: Ieee1394CameraArray.h,v 1.3 2010-12-19 20:04:04 ueshiba Exp $
 */
#ifndef __TUIeee1394CameraArray_h
#define __TUIeee1394CameraArray_h

#ifdef HAVE_LIBTUTOOLS__
#  include "TU/Array++.h"
#  include "TU/Ieee1394++.h"

namespace TU
{
/************************************************************************
*  class Ieee1394CameraArray						*
************************************************************************/
//! IEEE1394デジタルカメラの配列を表すクラス
/*!
  TU::Ieee1394Cameraへのポインタの配列として定義される．
*/
class Ieee1394CameraArray : public Array<Ieee1394Camera*>
{
  public:
    Ieee1394CameraArray()						;
    Ieee1394CameraArray(std::istream& in, bool i1394b, int ncameras=-1)	;
    ~Ieee1394CameraArray()						;

    void	initialize(std::istream& in,
			   bool i1394b, int ncameras=-1)		;
};

}
#endif	/* HAVE_LIBTUTOOLS__		*/    
#endif	/* !__TUIeeeCameraArray_h	*/
