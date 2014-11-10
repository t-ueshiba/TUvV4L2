/*
 *  $Id$
 */
#include "TU/Ieee1394CameraArray.h"

namespace TU
{
/************************************************************************
*  static functions							*
************************************************************************/
template <class CAMERAS> static bool
setCameraFormat1394(CAMERAS& cameras, u_int id, u_int val)
{
    switch (id)
    {
      case Ieee1394Camera::YUV444_160x120:
      case Ieee1394Camera::YUV422_320x240:
      case Ieee1394Camera::YUV411_640x480:
      case Ieee1394Camera::YUV422_640x480:
      case Ieee1394Camera::RGB24_640x480:
      case Ieee1394Camera::MONO8_640x480:
      case Ieee1394Camera::MONO16_640x480:
      case Ieee1394Camera::YUV422_800x600:
      case Ieee1394Camera::RGB24_800x600:
      case Ieee1394Camera::MONO8_800x600:
      case Ieee1394Camera::YUV422_1024x768:
      case Ieee1394Camera::RGB24_1024x768:
      case Ieee1394Camera::MONO8_1024x768:
      case Ieee1394Camera::MONO16_800x600:
      case Ieee1394Camera::MONO16_1024x768:
      case Ieee1394Camera::YUV422_1280x960:
      case Ieee1394Camera::RGB24_1280x960:
      case Ieee1394Camera::MONO8_1280x960:
      case Ieee1394Camera::YUV422_1600x1200:
      case Ieee1394Camera::RGB24_1600x1200:
      case Ieee1394Camera::MONO8_1600x1200:
      case Ieee1394Camera::MONO16_1280x960:
      case Ieee1394Camera::MONO16_1600x1200:
	exec(cameras, &Ieee1394Camera::setFormatAndFrameRate,
	     Ieee1394Camera::uintToFormat(id),
	     Ieee1394Camera::uintToFrameRate(val));
	return true;
    }

    return false;
}
    
template <class CAMERAS> static bool
setCameraFeatureValue1394(CAMERAS& cameras, u_int id, u_int val, int n)
{
    switch (id)
    {
      case Ieee1394Camera::BRIGHTNESS:
      case Ieee1394Camera::AUTO_EXPOSURE:
      case Ieee1394Camera::SHARPNESS:
      case Ieee1394Camera::HUE:
      case Ieee1394Camera::SATURATION:
      case Ieee1394Camera::GAMMA:
      case Ieee1394Camera::SHUTTER:
      case Ieee1394Camera::GAIN:
      case Ieee1394Camera::IRIS:
      case Ieee1394Camera::FOCUS:
      case Ieee1394Camera::TEMPERATURE:
      case Ieee1394Camera::ZOOM:
      case Ieee1394Camera::PAN:
      case Ieee1394Camera::TILT:
	exec(cameras, &Ieee1394Camera::setValue,
	     Ieee1394Camera::uintToFeature(id), val, n);
	return true;

      case Ieee1394Camera::TRIGGER_MODE:
	exec(cameras, &Ieee1394Camera::setTriggerMode,
	     Ieee1394Camera::uintToTriggerMode(val), n);
	return true;

      case Ieee1394Camera::WHITE_BALANCE:
      {
	u_int	ub, vr;
	exec(cameras, &Ieee1394Camera::getWhiteBalance, ub,  vr, n);
	exec(cameras, &Ieee1394Camera::setWhiteBalance, val, vr, n);
      }
	return true;
      
      case Ieee1394Camera::WHITE_BALANCE + IEEE1394CAMERA_OFFSET_VR:
      {
	u_int	ub, vr;
	exec(cameras, &Ieee1394Camera::getWhiteBalance, ub, vr,  n);
	exec(cameras, &Ieee1394Camera::setWhiteBalance, ub, val, n);
      }
	return true;
      
      case Ieee1394Camera::BRIGHTNESS	 + IEEE1394CAMERA_OFFSET_ONOFF:
      case Ieee1394Camera::AUTO_EXPOSURE + IEEE1394CAMERA_OFFSET_ONOFF:
      case Ieee1394Camera::SHARPNESS	 + IEEE1394CAMERA_OFFSET_ONOFF:
      case Ieee1394Camera::WHITE_BALANCE + IEEE1394CAMERA_OFFSET_ONOFF:
      case Ieee1394Camera::HUE		 + IEEE1394CAMERA_OFFSET_ONOFF:
      case Ieee1394Camera::SATURATION	 + IEEE1394CAMERA_OFFSET_ONOFF:
      case Ieee1394Camera::GAMMA	 + IEEE1394CAMERA_OFFSET_ONOFF:
      case Ieee1394Camera::SHUTTER	 + IEEE1394CAMERA_OFFSET_ONOFF:
      case Ieee1394Camera::GAIN		 + IEEE1394CAMERA_OFFSET_ONOFF:
      case Ieee1394Camera::IRIS		 + IEEE1394CAMERA_OFFSET_ONOFF:
      case Ieee1394Camera::FOCUS	 + IEEE1394CAMERA_OFFSET_ONOFF:
      case Ieee1394Camera::TEMPERATURE	 + IEEE1394CAMERA_OFFSET_ONOFF:
      case Ieee1394Camera::TRIGGER_MODE	 + IEEE1394CAMERA_OFFSET_ONOFF:
      case Ieee1394Camera::ZOOM		 + IEEE1394CAMERA_OFFSET_ONOFF:
      case Ieee1394Camera::PAN		 + IEEE1394CAMERA_OFFSET_ONOFF:
      case Ieee1394Camera::TILT		 + IEEE1394CAMERA_OFFSET_ONOFF:
      {
	Ieee1394Camera::Feature	feature = Ieee1394Camera::uintToFeature(
					      id - IEEE1394CAMERA_OFFSET_ONOFF);
	if (val)
	    exec(cameras, &Ieee1394Camera::turnOn,  feature, n);
	else
	    exec(cameras, &Ieee1394Camera::turnOff, feature, n);
      }
	return true;

      case Ieee1394Camera::BRIGHTNESS	 + IEEE1394CAMERA_OFFSET_AUTO:
      case Ieee1394Camera::AUTO_EXPOSURE + IEEE1394CAMERA_OFFSET_AUTO:
      case Ieee1394Camera::SHARPNESS	 + IEEE1394CAMERA_OFFSET_AUTO:
      case Ieee1394Camera::WHITE_BALANCE + IEEE1394CAMERA_OFFSET_AUTO:
      case Ieee1394Camera::HUE		 + IEEE1394CAMERA_OFFSET_AUTO:
      case Ieee1394Camera::SATURATION	 + IEEE1394CAMERA_OFFSET_AUTO:
      case Ieee1394Camera::GAMMA	 + IEEE1394CAMERA_OFFSET_AUTO:
      case Ieee1394Camera::SHUTTER	 + IEEE1394CAMERA_OFFSET_AUTO:
      case Ieee1394Camera::GAIN		 + IEEE1394CAMERA_OFFSET_AUTO:
      case Ieee1394Camera::IRIS		 + IEEE1394CAMERA_OFFSET_AUTO:
      case Ieee1394Camera::FOCUS	 + IEEE1394CAMERA_OFFSET_AUTO:
      case Ieee1394Camera::TEMPERATURE	 + IEEE1394CAMERA_OFFSET_AUTO:
      case Ieee1394Camera::ZOOM		 + IEEE1394CAMERA_OFFSET_AUTO:
      case Ieee1394Camera::PAN		 + IEEE1394CAMERA_OFFSET_AUTO:
      case Ieee1394Camera::TILT		 + IEEE1394CAMERA_OFFSET_AUTO:
      {
	Ieee1394Camera::Feature	feature = Ieee1394Camera::uintToFeature(
					      id - IEEE1394CAMERA_OFFSET_AUTO);
	if (val)
	    exec(cameras, &Ieee1394Camera::setAutoMode,   feature, n);
	else
	    exec(cameras, &Ieee1394Camera::setManualMode, feature, n);
      }
        return true;
    }

    return false;
}

template <class CAMERAS> static u_int
getCameraFeatureValue1394(const CAMERAS& cameras, u_int id, int n)
{
    switch (id)
    {
      case Ieee1394Camera::BRIGHTNESS:
      case Ieee1394Camera::AUTO_EXPOSURE:
      case Ieee1394Camera::SHARPNESS:
      case Ieee1394Camera::HUE:
      case Ieee1394Camera::SATURATION:
      case Ieee1394Camera::GAMMA:
      case Ieee1394Camera::SHUTTER:
      case Ieee1394Camera::GAIN:
      case Ieee1394Camera::IRIS:
      case Ieee1394Camera::FOCUS:
      case Ieee1394Camera::TEMPERATURE:
      case Ieee1394Camera::ZOOM:
      case Ieee1394Camera::PAN:
      case Ieee1394Camera::TILT:
	return exec(cameras, &Ieee1394Camera::getValue,
		    Ieee1394Camera::uintToFeature(id), n);

      case Ieee1394Camera::TRIGGER_MODE:
	return exec(cameras, &Ieee1394Camera::getTriggerMode, n);
	
      case Ieee1394Camera::WHITE_BALANCE:
      {
	u_int	ub, vr;
	exec(cameras, &Ieee1394Camera::getWhiteBalance, ub, vr, n);
	return ub;
      }
      
      case Ieee1394Camera::WHITE_BALANCE + IEEE1394CAMERA_OFFSET_VR:
      {
	u_int	ub, vr;
	exec(cameras, &Ieee1394Camera::getWhiteBalance, ub, vr, n);
	return vr;
      }
	
      case Ieee1394Camera::BRIGHTNESS	 + IEEE1394CAMERA_OFFSET_ONOFF:
      case Ieee1394Camera::AUTO_EXPOSURE + IEEE1394CAMERA_OFFSET_ONOFF:
      case Ieee1394Camera::SHARPNESS	 + IEEE1394CAMERA_OFFSET_ONOFF:
      case Ieee1394Camera::WHITE_BALANCE + IEEE1394CAMERA_OFFSET_ONOFF:
      case Ieee1394Camera::HUE		 + IEEE1394CAMERA_OFFSET_ONOFF:
      case Ieee1394Camera::SATURATION	 + IEEE1394CAMERA_OFFSET_ONOFF:
      case Ieee1394Camera::GAMMA	 + IEEE1394CAMERA_OFFSET_ONOFF:
      case Ieee1394Camera::SHUTTER	 + IEEE1394CAMERA_OFFSET_ONOFF:
      case Ieee1394Camera::GAIN		 + IEEE1394CAMERA_OFFSET_ONOFF:
      case Ieee1394Camera::IRIS		 + IEEE1394CAMERA_OFFSET_ONOFF:
      case Ieee1394Camera::FOCUS	 + IEEE1394CAMERA_OFFSET_ONOFF:
      case Ieee1394Camera::TEMPERATURE	 + IEEE1394CAMERA_OFFSET_ONOFF:
      case Ieee1394Camera::TRIGGER_MODE	 + IEEE1394CAMERA_OFFSET_ONOFF:
      case Ieee1394Camera::ZOOM		 + IEEE1394CAMERA_OFFSET_ONOFF:
      case Ieee1394Camera::PAN		 + IEEE1394CAMERA_OFFSET_ONOFF:
      case Ieee1394Camera::TILT		 + IEEE1394CAMERA_OFFSET_ONOFF:
      {
	Ieee1394Camera::Feature	feature = Ieee1394Camera::uintToFeature(
					      id - IEEE1394CAMERA_OFFSET_ONOFF);
	return exec(cameras, &Ieee1394Camera::isTurnedOn, feature, n);
      }

      case Ieee1394Camera::BRIGHTNESS	 + IEEE1394CAMERA_OFFSET_AUTO:
      case Ieee1394Camera::AUTO_EXPOSURE + IEEE1394CAMERA_OFFSET_AUTO:
      case Ieee1394Camera::SHARPNESS	 + IEEE1394CAMERA_OFFSET_AUTO:
      case Ieee1394Camera::WHITE_BALANCE + IEEE1394CAMERA_OFFSET_AUTO:
      case Ieee1394Camera::HUE		 + IEEE1394CAMERA_OFFSET_AUTO:
      case Ieee1394Camera::SATURATION	 + IEEE1394CAMERA_OFFSET_AUTO:
      case Ieee1394Camera::GAMMA	 + IEEE1394CAMERA_OFFSET_AUTO:
      case Ieee1394Camera::SHUTTER	 + IEEE1394CAMERA_OFFSET_AUTO:
      case Ieee1394Camera::GAIN		 + IEEE1394CAMERA_OFFSET_AUTO:
      case Ieee1394Camera::IRIS		 + IEEE1394CAMERA_OFFSET_AUTO:
      case Ieee1394Camera::FOCUS	 + IEEE1394CAMERA_OFFSET_AUTO:
      case Ieee1394Camera::TEMPERATURE	 + IEEE1394CAMERA_OFFSET_AUTO:
      case Ieee1394Camera::ZOOM		 + IEEE1394CAMERA_OFFSET_AUTO:
      case Ieee1394Camera::PAN		 + IEEE1394CAMERA_OFFSET_AUTO:
      case Ieee1394Camera::TILT		 + IEEE1394CAMERA_OFFSET_AUTO:
      {
	Ieee1394Camera::Feature	feature = Ieee1394Camera::uintToFeature(
					      id - IEEE1394CAMERA_OFFSET_AUTO);
	return exec(cameras, &Ieee1394Camera::isAuto, feature, n);
      }
    }

    throw std::invalid_argument("getCameraFeatureValue1394(): unknown feature!!");
    
    return 0;
}

/************************************************************************
*  global functions							*
************************************************************************/
bool
setCameraFormat(Ieee1394Camera& camera, u_int id, int val)
{
    return setCameraFormat1394(camera, id, val);
}

bool
setCameraFeatureValue(Ieee1394Camera& camera, u_int id, int val, int)
{
    return setCameraFeatureValue1394(camera, id, val, -1);
}

u_int
getCameraFeatureValue(const Ieee1394Camera& camera, u_int id, int)
{
    return getCameraFeatureValue1394(camera, id, -1);
}
    
#ifdef HAVE_LIBTUTOOLS__
bool
setCameraFormat(const Array<Ieee1394Camera*>& cameras, u_int id, int val)
{
    return setCameraFormat1394(cameras, id, val);
}

bool
setCameraFeatureValue(const Array<Ieee1394Camera*>& cameras,
		      u_int id, int val, int n)
{
    return setCameraFeatureValue1394(cameras, id, val, n);
}

u_int
getCameraFeatureValue(const Array<Ieee1394Camera*>& cameras, u_int id, int n)
{
    return getCameraFeatureValue1394(cameras, id, n);
}

void
exec(const Array<Ieee1394Camera*>& cameras,
     Ieee1394Camera& (Ieee1394Camera::*mf)(), int n)
{
    if (0 <= n && n < cameras.size())
	(cameras[n]->*mf)();
    else
	for (size_t i = 0; i < cameras.size(); ++i)
	    (cameras[i]->*mf)();
}
#endif
}
