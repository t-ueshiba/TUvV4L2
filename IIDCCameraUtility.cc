/*
 *  $Id$
 */
#include "TU/IIDCCameraArray.h"
#include "TU/Heap.h"

namespace TU
{
/************************************************************************
*  static functions							*
************************************************************************/
template <class CAMERAS> static bool
setFormatIIDC(CAMERAS& cameras, u_int id, u_int val)
{
    switch (id)
    {
      case IIDCCamera::YUV444_160x120:
      case IIDCCamera::YUV422_320x240:
      case IIDCCamera::YUV411_640x480:
      case IIDCCamera::YUV422_640x480:
      case IIDCCamera::RGB24_640x480:
      case IIDCCamera::MONO8_640x480:
      case IIDCCamera::MONO16_640x480:
      case IIDCCamera::YUV422_800x600:
      case IIDCCamera::RGB24_800x600:
      case IIDCCamera::MONO8_800x600:
      case IIDCCamera::YUV422_1024x768:
      case IIDCCamera::RGB24_1024x768:
      case IIDCCamera::MONO8_1024x768:
      case IIDCCamera::MONO16_800x600:
      case IIDCCamera::MONO16_1024x768:
      case IIDCCamera::YUV422_1280x960:
      case IIDCCamera::RGB24_1280x960:
      case IIDCCamera::MONO8_1280x960:
      case IIDCCamera::YUV422_1600x1200:
      case IIDCCamera::RGB24_1600x1200:
      case IIDCCamera::MONO8_1600x1200:
      case IIDCCamera::MONO16_1280x960:
      case IIDCCamera::MONO16_1600x1200:
	exec(cameras, &IIDCCamera::setFormatAndFrameRate,
	     IIDCCamera::uintToFormat(id),
	     IIDCCamera::uintToFrameRate(val));
	return true;

      default:
	break;
    }

    return false;
}
    
template <class CAMERAS> static bool
setFeatureValueIIDC(CAMERAS& cameras, u_int id, u_int val, int n)
{
    switch (id)
    {
      case IIDCCamera::BRIGHTNESS:
      case IIDCCamera::AUTO_EXPOSURE:
      case IIDCCamera::SHARPNESS:
      case IIDCCamera::HUE:
      case IIDCCamera::SATURATION:
      case IIDCCamera::GAMMA:
      case IIDCCamera::SHUTTER:
      case IIDCCamera::GAIN:
      case IIDCCamera::IRIS:
      case IIDCCamera::FOCUS:
      case IIDCCamera::TEMPERATURE:
      case IIDCCamera::ZOOM:
      case IIDCCamera::PAN:
      case IIDCCamera::TILT:
	exec(cameras, &IIDCCamera::setValue,
	     IIDCCamera::uintToFeature(id), val, n);
	return true;

      case IIDCCamera::TRIGGER_MODE:
	exec(cameras, &IIDCCamera::setTriggerMode,
	     IIDCCamera::uintToTriggerMode(val), n);
	return true;

      case IIDCCamera::WHITE_BALANCE:
      {
	u_int	ub, vr;
	exec(cameras, &IIDCCamera::getWhiteBalance, ub,  vr, n);
	exec(cameras, &IIDCCamera::setWhiteBalance, val, vr, n);
      }
	return true;
      
      case IIDCCamera::WHITE_BALANCE + IIDCCAMERA_OFFSET_VR:
      {
	u_int	ub, vr;
	exec(cameras, &IIDCCamera::getWhiteBalance, ub, vr,  n);
	exec(cameras, &IIDCCamera::setWhiteBalance, ub, val, n);
      }
	return true;
      
      case IIDCCamera::BRIGHTNESS    + IIDCCAMERA_OFFSET_ONOFF:
      case IIDCCamera::AUTO_EXPOSURE + IIDCCAMERA_OFFSET_ONOFF:
      case IIDCCamera::SHARPNESS     + IIDCCAMERA_OFFSET_ONOFF:
      case IIDCCamera::WHITE_BALANCE + IIDCCAMERA_OFFSET_ONOFF:
      case IIDCCamera::HUE	     + IIDCCAMERA_OFFSET_ONOFF:
      case IIDCCamera::SATURATION    + IIDCCAMERA_OFFSET_ONOFF:
      case IIDCCamera::GAMMA	     + IIDCCAMERA_OFFSET_ONOFF:
      case IIDCCamera::SHUTTER	     + IIDCCAMERA_OFFSET_ONOFF:
      case IIDCCamera::GAIN	     + IIDCCAMERA_OFFSET_ONOFF:
      case IIDCCamera::IRIS	     + IIDCCAMERA_OFFSET_ONOFF:
      case IIDCCamera::FOCUS	     + IIDCCAMERA_OFFSET_ONOFF:
      case IIDCCamera::TEMPERATURE   + IIDCCAMERA_OFFSET_ONOFF:
      case IIDCCamera::TRIGGER_MODE  + IIDCCAMERA_OFFSET_ONOFF:
      case IIDCCamera::ZOOM	     + IIDCCAMERA_OFFSET_ONOFF:
      case IIDCCamera::PAN	     + IIDCCAMERA_OFFSET_ONOFF:
      case IIDCCamera::TILT	     + IIDCCAMERA_OFFSET_ONOFF:
	exec(cameras, &IIDCCamera::setActive,
	     IIDCCamera::uintToFeature(id - IIDCCAMERA_OFFSET_ONOFF),
	     bool(val), n);
	return true;

      case IIDCCamera::BRIGHTNESS    + IIDCCAMERA_OFFSET_AUTO:
      case IIDCCamera::AUTO_EXPOSURE + IIDCCAMERA_OFFSET_AUTO:
      case IIDCCamera::SHARPNESS     + IIDCCAMERA_OFFSET_AUTO:
      case IIDCCamera::WHITE_BALANCE + IIDCCAMERA_OFFSET_AUTO:
      case IIDCCamera::HUE	     + IIDCCAMERA_OFFSET_AUTO:
      case IIDCCamera::SATURATION    + IIDCCAMERA_OFFSET_AUTO:
      case IIDCCamera::GAMMA	     + IIDCCAMERA_OFFSET_AUTO:
      case IIDCCamera::SHUTTER	     + IIDCCAMERA_OFFSET_AUTO:
      case IIDCCamera::GAIN	     + IIDCCAMERA_OFFSET_AUTO:
      case IIDCCamera::IRIS	     + IIDCCAMERA_OFFSET_AUTO:
      case IIDCCamera::FOCUS	     + IIDCCAMERA_OFFSET_AUTO:
      case IIDCCamera::TEMPERATURE   + IIDCCAMERA_OFFSET_AUTO:
      case IIDCCamera::ZOOM	     + IIDCCAMERA_OFFSET_AUTO:
      case IIDCCamera::PAN	     + IIDCCAMERA_OFFSET_AUTO:
      case IIDCCamera::TILT	     + IIDCCAMERA_OFFSET_AUTO:
	exec(cameras, &IIDCCamera::setAuto,
	     IIDCCamera::uintToFeature(id - IIDCCAMERA_OFFSET_AUTO),
	     bool(val), n);
        return true;

      case IIDCCamera::TRIGGER_MODE  + IIDCCAMERA_OFFSET_AUTO:
	exec(cameras, &IIDCCamera::setTriggerPolarity, bool(val), n);
	return true;

      default:
	break;
    }

    return false;
}

template <class CAMERAS> static u_int
getFeatureValueIIDC(const CAMERAS& cameras, u_int id, int n)
{
    switch (id)
    {
      case IIDCCamera::BRIGHTNESS:
      case IIDCCamera::AUTO_EXPOSURE:
      case IIDCCamera::SHARPNESS:
      case IIDCCamera::HUE:
      case IIDCCamera::SATURATION:
      case IIDCCamera::GAMMA:
      case IIDCCamera::SHUTTER:
      case IIDCCamera::GAIN:
      case IIDCCamera::IRIS:
      case IIDCCamera::FOCUS:
      case IIDCCamera::TEMPERATURE:
      case IIDCCamera::ZOOM:
      case IIDCCamera::PAN:
      case IIDCCamera::TILT:
	return exec(cameras, &IIDCCamera::getValue,
		    IIDCCamera::uintToFeature(id), n);

      case IIDCCamera::TRIGGER_MODE:
	return exec(cameras, &IIDCCamera::getTriggerMode, n);
	
      case IIDCCamera::WHITE_BALANCE:
      {
	u_int	ub, vr;
	exec(cameras, &IIDCCamera::getWhiteBalance, ub, vr, n);
	return ub;
      }
      
      case IIDCCamera::WHITE_BALANCE + IIDCCAMERA_OFFSET_VR:
      {
	u_int	ub, vr;
	exec(cameras, &IIDCCamera::getWhiteBalance, ub, vr, n);
	return vr;
      }
	
      case IIDCCamera::BRIGHTNESS    + IIDCCAMERA_OFFSET_ONOFF:
      case IIDCCamera::AUTO_EXPOSURE + IIDCCAMERA_OFFSET_ONOFF:
      case IIDCCamera::SHARPNESS     + IIDCCAMERA_OFFSET_ONOFF:
      case IIDCCamera::WHITE_BALANCE + IIDCCAMERA_OFFSET_ONOFF:
      case IIDCCamera::HUE	     + IIDCCAMERA_OFFSET_ONOFF:
      case IIDCCamera::SATURATION    + IIDCCAMERA_OFFSET_ONOFF:
      case IIDCCamera::GAMMA	     + IIDCCAMERA_OFFSET_ONOFF:
      case IIDCCamera::SHUTTER	     + IIDCCAMERA_OFFSET_ONOFF:
      case IIDCCamera::GAIN	     + IIDCCAMERA_OFFSET_ONOFF:
      case IIDCCamera::IRIS	     + IIDCCAMERA_OFFSET_ONOFF:
      case IIDCCamera::FOCUS	     + IIDCCAMERA_OFFSET_ONOFF:
      case IIDCCamera::TEMPERATURE   + IIDCCAMERA_OFFSET_ONOFF:
      case IIDCCamera::TRIGGER_MODE  + IIDCCAMERA_OFFSET_ONOFF:
      case IIDCCamera::ZOOM	     + IIDCCAMERA_OFFSET_ONOFF:
      case IIDCCamera::PAN	     + IIDCCAMERA_OFFSET_ONOFF:
      case IIDCCamera::TILT	     + IIDCCAMERA_OFFSET_ONOFF:
	return exec(cameras, &IIDCCamera::isActive,
		    IIDCCamera::uintToFeature(id - IIDCCAMERA_OFFSET_ONOFF), n);
	
      case IIDCCamera::BRIGHTNESS    + IIDCCAMERA_OFFSET_AUTO:
      case IIDCCamera::AUTO_EXPOSURE + IIDCCAMERA_OFFSET_AUTO:
      case IIDCCamera::SHARPNESS     + IIDCCAMERA_OFFSET_AUTO:
      case IIDCCamera::WHITE_BALANCE + IIDCCAMERA_OFFSET_AUTO:
      case IIDCCamera::HUE	     + IIDCCAMERA_OFFSET_AUTO:
      case IIDCCamera::SATURATION    + IIDCCAMERA_OFFSET_AUTO:
      case IIDCCamera::GAMMA	     + IIDCCAMERA_OFFSET_AUTO:
      case IIDCCamera::SHUTTER	     + IIDCCAMERA_OFFSET_AUTO:
      case IIDCCamera::GAIN	     + IIDCCAMERA_OFFSET_AUTO:
      case IIDCCamera::IRIS	     + IIDCCAMERA_OFFSET_AUTO:
      case IIDCCamera::FOCUS	     + IIDCCAMERA_OFFSET_AUTO:
      case IIDCCamera::TEMPERATURE   + IIDCCAMERA_OFFSET_AUTO:
      case IIDCCamera::ZOOM	     + IIDCCAMERA_OFFSET_AUTO:
      case IIDCCamera::PAN	     + IIDCCAMERA_OFFSET_AUTO:
      case IIDCCamera::TILT	     + IIDCCAMERA_OFFSET_AUTO:
	return exec(cameras, &IIDCCamera::isAuto,
		    IIDCCamera::uintToFeature(id - IIDCCAMERA_OFFSET_AUTO), n);

      case IIDCCamera::TRIGGER_MODE  + IIDCCAMERA_OFFSET_AUTO:
	return exec(cameras, &IIDCCamera::getTriggerPolarity, n);

      default:
	break;
    }

    throw std::invalid_argument("getFeatureValueIIDC(): unknown feature!!");
    
    return 0;
}

/************************************************************************
*  global functions							*
************************************************************************/
bool
setFormat(IIDCCamera& camera, u_int id, int val)
{
    return setFormatIIDC(camera, id, val);
}

bool
setFeatureValue(IIDCCamera& camera, u_int id, int val, int)
{
    return setFeatureValueIIDC(camera, id, val, -1);
}

u_int
getFeatureValue(const IIDCCamera& camera, u_int id, int)
{
    return getFeatureValueIIDC(camera, id, -1);
}
    
bool
setFormat(const Array<IIDCCamera*>& cameras, u_int id, int val)
{
    return setFormatIIDC(cameras, id, val);
}

bool
setFeatureValue(const Array<IIDCCamera*>& cameras,
		u_int id, int val, int n)
{
    return setFeatureValueIIDC(cameras, id, val, n);
}

u_int
getFeatureValue(const Array<IIDCCamera*>& cameras, u_int id, int n)
{
    return getFeatureValueIIDC(cameras, id, n);
}

void
exec(const Array<IIDCCamera*>& cameras, IIDCCamera& (IIDCCamera::*mf)(), int n)
{
    if (0 <= n && n < cameras.size())
	(cameras[n]->*mf)();
    else
	for (size_t i = 0; i < cameras.size(); ++i)
	    (cameras[i]->*mf)();
}

void
syncedSnap(const Array<IIDCCamera*>& cameras, uint64_t thresh)
{
    typedef std::pair<uint64_t, IIDCCamera*>	timestamp_t;
    
    Heap<timestamp_t, std::greater<timestamp_t> >
			timestamps(cameras.size());
    
    for (size_t i = 0; i < cameras.size(); ++i)
	cameras[i]->snap();

    timestamp_t		last(0, cameras[0]);
    for (size_t i = 0; i < cameras.size(); ++i)
    {
	timestamp_t	timestamp(cameras[i]->getTimestamp(), cameras[i]);
	timestamps.push(timestamp);
	if (timestamp > last)
	    last = timestamp;
    }

    for (timestamp_t top; last.first - (top = timestamps.pop()).first > thresh;
	 timestamps.push(last))
	last = {top.second->snap().getTimestamp(), top.second};
}
    
}
