/*
 *  $Id: MyCmdWindow.cc,v 1.2 2012-06-19 08:54:04 ueshiba Exp $
 */
#include <unistd.h>
#include <sys/time.h>
#include "multicam.h"
#include "MyCmdWindow.h"

namespace TU
{
/************************************************************************
*  static functions							*
************************************************************************/
static void
countTime(int& nframes, struct timeval& start)
{
    if (nframes == 10)
    {
	struct timeval	end;
	gettimeofday(&end, NULL);
	double	interval = (end.tv_sec  - start.tv_sec) +
	    (end.tv_usec - start.tv_usec) / 1.0e6;
	std::cerr << nframes / interval << " frames/sec" << std::endl;
	nframes = 0;
    }
    if (nframes++ == 0)
	gettimeofday(&start, NULL);
}

namespace v
{
CmdDef*		createMenuCmds(const V4L2Camera& camera)	;
CmdDef*		createCaptureCmds()				;
CmdDef*		createFeatureCmds(const V4L2Camera& camera)	;
    
/************************************************************************
*  class MyCmdWindow							*
************************************************************************/
MyCmdWindow::MyCmdWindow(App& parentApp, V4L2Camera& camera)
    :CmdWindow(parentApp, "V4L2 camera controller",
	       0, Colormap::RGBColor, 16, 0, 0),
     _camera(camera),
     _image(),
     _canvas(*this, _camera.width(), _camera.height(), _image),
     _menuCmd(*this, createMenuCmds(_camera)),
     _captureCmd(*this, createCaptureCmds()),
     _featureCmd(*this, createFeatureCmds(_camera)),
     _fileSelection(*this),
     _timer(*this, 0)
{
    _menuCmd.place(0, 0, 2, 1);
    _captureCmd.place(0, 1, 1, 1);
    _featureCmd.place(1, 1, 1, 1);
    _canvas.place(0, 2, 2, 1);

    show();
}

void
MyCmdWindow::callback(CmdId id, CmdVal val)
{
    using namespace	std;

    try
    {
	switch (id)
	{
	  case M_Exit:
	    app().exit();
	    break;

	  case M_Save:
	  {
	    stopContinuousShotIfRunning();
	    
	    ofstream	out;
	    if (_fileSelection.open(out))
		_image.save(out);
	  }
	    break;
	
	  case c_BGR24:
	  case c_RGB24:
	  case c_BGR32:
	  case c_RGB32:
	  case c_GREY:
	  case c_Y16:
	  case c_YUYV:
	  case c_UYVY:
	  case c_SBGGR8:
	  case c_SGBRG8:
	  case c_SGRBG8:
	  {
	    V4L2Camera::PixelFormat
		pixelFormat = V4L2Camera::uintToPixelFormat(id);
	    V4L2Camera::FrameSizeRange
		frameSizes = _camera.availableFrameSizes(pixelFormat);
	    const V4L2Camera::FrameSize&	frameSize = *frameSizes.first;
	    u_int	w = frameSize.width.max, h = frameSize.height.max;
	    V4L2Camera::FrameRateRange
			frameRates = frameSize.availableFrameRates();
	    const V4L2Camera::FrameRate&	frameRate = *frameRates.first;
	    u_int	fps_n = frameRate.fps_n.max,
			fps_d = frameRate.fps_d.max;
	    _camera.setFormat(pixelFormat, w, h, fps_n, fps_d);
	    _canvas.resize(w, h);
	  }
	    break;

	  case c_Brightness:
	  case c_Brightness_Auto:
	  case c_Contrast:
	  case c_Gain:
	  case c_Saturation:
	  case c_Hue:
	  case c_Hue_Auto:
	  case c_Gamma:
	  case c_Sharpness:
	  case c_White_Balance_Temperature:
	  case c_White_Balance_Auto:
	  case c_Backlight_Compensation:
	  case c_Power_Frequency:
	  case c_Exposure_Auto:
	  case c_Exposure_Auto_Priority:
	  case c_Exposure_Absolute:
	  case c_Focus_Absolute:
	  case c_Focus_Relative:
	  case c_Focus_Auto:
	  case c_Zomm_Absolute:
	  case c_Zoom_Relative:
	  case c_Zoom_Continuous:
	  case c_Pan_Absolute:
	  case c_Pan_Relative:
	  case c_Pan_Reset:
	  case c_Tilt_Absolute:
	  case c_Tilt_Relative:
	  case c_Tilt_Reset:
	    _camera.setValue(V4L2Camera::uintToFeature(id), val);
	    break;

	  case c_ContinuousShot:
	    if (val)
	    {
		_camera.continuousShot();
		_timer.start(1);
	    }
	    else
	    {
		_timer.stop();
		_camera.stopContinuousShot();
	    }
	    break;
	}
    }
    catch (exception& err)
    {
	cerr << err.what();
    }
}

void
MyCmdWindow::tick()
{
    static int			nframes = 0;
    static struct timeval	start;
    countTime(nframes, start);

    _camera.snap() >> _image;

    repaintCanvas();
}
    
void
MyCmdWindow::repaintCanvas()
{
    _canvas.repaintUnderlay();
}

void
MyCmdWindow::stopContinuousShotIfRunning()
{
    if (_captureCmd.getValue(c_ContinuousShot))
    {
	_timer.stop();
	_camera.stopContinuousShot();
	_captureCmd.setValue(c_ContinuousShot, 0);
    }
}
 
}
}
