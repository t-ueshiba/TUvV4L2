/*
 *  $Id$
 */
#include "TU/V4L2CameraArray.h"

namespace TU
{
/************************************************************************
*  static functions							*
************************************************************************/
static void
setFormat(V4L2Camera& camera, V4L2Camera::PixelFormat pixelFormat, int val)
{
    const V4L2Camera::FrameSize&
		frameSize = camera.availableFrameSizes(pixelFormat).first[val];
    const V4L2Camera::FrameRate&
		frameRate = *frameSize.availableFrameRates().first;
    camera.setFormat(pixelFormat, frameSize.width.max, frameSize.height.max,
		     frameRate.fps_n.min, frameRate.fps_d.max);
}
    
static void
setFormat(const Array<V4L2Camera*>& cameras,
	  V4L2Camera::PixelFormat pixelFormat, int val)
{
    for (size_t i = 0; i < cameras.size(); ++i)
	setFormat(*cameras[i], pixelFormat, val);
}
    
template <class CAMERAS> static bool
handleCameraFormatsV4L2(CAMERAS& cameras, u_int id, int val)
{
    switch (id)
    {
      case V4L2Camera::BGR24:
      case V4L2Camera::RGB24:
      case V4L2Camera::BGR32:
      case V4L2Camera::RGB32:
      case V4L2Camera::GREY:
      case V4L2Camera::Y16:
      case V4L2Camera::YUYV:
      case V4L2Camera::UYVY:
      case V4L2Camera::SBGGR8:
      case V4L2Camera::SGBRG8:
      case V4L2Camera::SGRBG8:
#ifdef V4L2_PIX_FMT_SRGGB8
      case V4L2Camera::SRGGB8:
#endif
	setFormat(cameras, V4L2Camera::uintToPixelFormat(id), val);
	return true;
    }

    return false;
}
    
    
template <class CAMERAS> static bool
handleCameraFeaturesV4L2(CAMERAS& cameras, u_int id, int val, int n)
{
    switch (id)
    {
      case V4L2Camera::BRIGHTNESS:
      case V4L2Camera::BRIGHTNESS_AUTO:
      case V4L2Camera::CONTRAST:
      case V4L2Camera::GAIN:
      case V4L2Camera::GAIN_AUTO:
      case V4L2Camera::SATURATION:
      case V4L2Camera::HUE:
      case V4L2Camera::HUE_AUTO:
      case V4L2Camera::GAMMA:
      case V4L2Camera::SHARPNESS:
      case V4L2Camera::BLACK_LEVEL:
      case V4L2Camera::WHITE_BALANCE_TEMPERATURE:
      case V4L2Camera::WHITE_BALANCE_AUTO:
      case V4L2Camera::RED_BALANCE:
      case V4L2Camera::BLUE_BALANCE:
      case V4L2Camera::HFLIP:
      case V4L2Camera::VFLIP:
      case V4L2Camera::BACKLIGHT_COMPENSATION:
      case V4L2Camera::POWER_LINE_FREQUENCY:
      case V4L2Camera::EXPOSURE_AUTO:
      case V4L2Camera::EXPOSURE_AUTO_PRIORITY:
      case V4L2Camera::EXPOSURE_ABSOLUTE:
      case V4L2Camera::FOCUS_ABSOLUTE:
      case V4L2Camera::FOCUS_RELATIVE:
      case V4L2Camera::FOCUS_AUTO:
      case V4L2Camera::ZOOM_ABSOLUTE:
      case V4L2Camera::ZOOM_RELATIVE:
      case V4L2Camera::ZOOM_CONTINUOUS:
#ifdef V4L2_CID_IRIS_ABSOLUTE
      case V4L2Camera::IRIS_ABSOLUTE:
#endif
#ifdef V4L2_CID_IRIS_RELATIVE
      case V4L2Camera::IRIS_RELATIVE:
#endif
      case V4L2Camera::PAN_ABSOLUTE:
      case V4L2Camera::PAN_RELATIVE:
      case V4L2Camera::PAN_RESET:
      case V4L2Camera::TILT_ABSOLUTE:
      case V4L2Camera::TILT_RELATIVE:
      case V4L2Camera::TILT_RESET:
      case V4L2Camera::CID_PRIVATE0:
      case V4L2Camera::CID_PRIVATE1:
      case V4L2Camera::CID_PRIVATE2:
      case V4L2Camera::CID_PRIVATE3:
      case V4L2Camera::CID_PRIVATE4:
      case V4L2Camera::CID_PRIVATE5:
      case V4L2Camera::CID_PRIVATE6:
      case V4L2Camera::CID_PRIVATE7:
      case V4L2Camera::CID_PRIVATE8:
      case V4L2Camera::CID_PRIVATE9:
      case V4L2Camera::CID_PRIVATE10:
      case V4L2Camera::CID_PRIVATE11:
      case V4L2Camera::CID_PRIVATE12:
      case V4L2Camera::CID_PRIVATE13:
      case V4L2Camera::CID_PRIVATE14:
      case V4L2Camera::CID_PRIVATE15:
      case V4L2Camera::CID_PRIVATE16:
      case V4L2Camera::CID_PRIVATE17:
      case V4L2Camera::CID_PRIVATE18:
      case V4L2Camera::CID_PRIVATE19:
      case V4L2Camera::CID_PRIVATE20:
      case V4L2Camera::CID_PRIVATE21:
      case V4L2Camera::CID_PRIVATE22:
      case V4L2Camera::CID_PRIVATE23:
      case V4L2Camera::CID_PRIVATE24:
      case V4L2Camera::CID_PRIVATE25:
      case V4L2Camera::CID_PRIVATE26:
      case V4L2Camera::CID_PRIVATE27:
      case V4L2Camera::CID_PRIVATE28:
      case V4L2Camera::CID_PRIVATE29:
	exec(cameras,
	     &V4L2Camera::setValue, V4L2Camera::uintToFeature(id), val, n);
	return true;
    }

    return false;
}

/************************************************************************
*  global functions							*
************************************************************************/
bool
handleCameraFormats(V4L2Camera& camera, u_int id, int val)
{
    return handleCameraFormatsV4L2(camera, id, val);
}

bool
handleCameraFeatures(V4L2Camera& camera, u_int id, int val, int)
{
    return handleCameraFeaturesV4L2(camera, id, val, -1);
}

#ifdef HAVE_LIBTUTOOLS__
bool
handleCameraFormats(const Array<V4L2Camera*>& cameras, u_int id, int val)
{
    return handleCameraFormatsV4L2(cameras, id, val);
}

bool
handleCameraFeatures(const Array<V4L2Camera*>& cameras,
		     u_int id, int val, int n)
{
    return handleCameraFeaturesV4L2(cameras, id, val, n);
}

void
exec(const Array<V4L2Camera*>& cameras, V4L2Camera& (V4L2Camera::*mf)(), int n)
{
    if (0 <= n && n < cameras.size())
	(cameras[n]->*mf)();
    else
	for (size_t i = 0; i < cameras.size(); ++i)
	    (cameras[i]->*mf)();
}
#endif
}
