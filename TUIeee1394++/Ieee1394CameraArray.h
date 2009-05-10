/*
 *  $Id: Ieee1394CameraArray.h,v 1.1 2009-05-10 23:34:27 ueshiba Exp $
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
    Ieee1394CameraArray(std::istream& in, bool i1394b, int ncameras=-1)	;
    ~Ieee1394CameraArray()						;
};

}
#endif	/* HAVE_LIBTUTOOLS__		*/    
#endif	/* !__TUIeeeCameraArray_h	*/
