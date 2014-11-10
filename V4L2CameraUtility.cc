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
setCameraFormatV4L2(CAMERAS& cameras, u_int id, int val)
{
    V4L2Camera::PixelFormat	pixelFormat = V4L2Camera::uintToPixelFormat(id);

    if (pixelFormat == V4L2Camera::UNKNOWN_PIXEL_FORMAT)
	return false;

    setFormat(cameras, pixelFormat, val);
    return true;
}
    
template <class CAMERAS> static bool
setCameraFeatureValueV4L2(CAMERAS& cameras, u_int id, int val, int n)
{
    V4L2Camera::Feature	feature = V4L2Camera::uintToFeature(id);

    if (feature == V4L2Camera::UNKNOWN_FEATURE)
	return false;

    exec(cameras, &V4L2Camera::setValue, feature, val, n);
    return true;
}

template <class CAMERAS> static int
getCameraFeatureValueV4L2(CAMERAS& cameras, u_int id, int n)
{
    V4L2Camera::Feature	feature = V4L2Camera::uintToFeature(id);

    if (feature == V4L2Camera::UNKNOWN_FEATURE)
	throw std::invalid_argument("getCameraFeatureValueV4L2(): unknown feature!!");

    return exec(cameras, &V4L2Camera::getValue, feature, n);
}

/************************************************************************
*  global functions							*
************************************************************************/
bool
setCameraFormat(V4L2Camera& camera, u_int id, int val)
{
    return setCameraFormatV4L2(camera, id, val);
}

bool
setCameraFeatureValue(V4L2Camera& camera, u_int id, int val, int)
{
    return setCameraFeatureValueV4L2(camera, id, val, -1);
}

int
getCameraFeatureValue(const V4L2Camera& camera, u_int id, int)
{
    return getCameraFeatureValueV4L2(camera, id, -1);
}
    
#ifdef HAVE_LIBTUTOOLS__
bool
setCameraFormat(const Array<V4L2Camera*>& cameras, u_int id, int val)
{
    return setCameraFormatV4L2(cameras, id, val);
}

bool
setCameraFeatureValue(const Array<V4L2Camera*>& cameras,
		      u_int id, int val, int n)
{
    return setCameraFeatureValueV4L2(cameras, id, val, n);
}

int
getCameraFeatureValue(const Array<V4L2Camera*>& cameras, u_int id, int n)
{
    return getCameraFeatureValueV4L2(cameras, id, n);
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
