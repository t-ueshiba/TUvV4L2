/*
 *  $Id$
 */
#include <unistd.h>
#include <sys/time.h>
#include <sstream>
#include "testcam.h"
#include "MyCmdWindow.h"
#include "MyModalDialog.h"

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

static std::ostream&
printTime(std::ostream& out, u_int64_t localtime)
{
    u_int64_t	usec = localtime % 1000;
    u_int64_t	msec = (localtime / 1000) % 1000;
    u_int64_t	sec  = localtime / 1000000;
    return out << sec << '.' << msec << '.' << usec;
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
   //_captureCmd(*this, createCaptureCmds()),
     _featureCmd(*this, createFeatureCmds(_camera)),
     _fileSelection(*this),
     _timer(*this, 0)
{
    _menuCmd.place(0, 0, 2, 1);
  //_captureCmd.place(0, 1, 1, 1);
    _featureCmd.place(1, 1, 1, 1);
    _canvas.place(0, 1, 1, 1);

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
	    const V4L2Camera::FrameSize&
		frameSize = _camera.availableFrameSizes(pixelFormat).first[val];
	    u_int	w = frameSize.width.max, h = frameSize.height.max;
	    V4L2Camera::FrameRateRange
			frameRates = frameSize.availableFrameRates();
	    const V4L2Camera::FrameRate&	frameRate = *frameRates.first;
	    u_int	fps_n = frameRate.fps_n.min,
			fps_d = frameRate.fps_d.max;
	    _camera.setFormat(pixelFormat, w, h, fps_n, fps_d);
	    _canvas.resize(w, h);
	  }
	    break;

	  case c_ROI:
	  {
	    size_t	u0, v0, width, height;
	    if (_camera.getROI(u0, v0, width, height))
	    {
		MyModalDialog	modalDialog(*this, _camera);
		modalDialog.getROI(u0, v0, width, height);
		_camera.setROI(u0, v0, width, height);
		_canvas.resize(_camera.width(), _camera.height());
	    }
	  }
	    break;
	  
	  case c_Brightness:
	  case c_Brightness_Auto:
	  case c_Contrast:
	  case c_Gain:
	  case c_Gain_Auto:
	  case c_Saturation:
	  case c_Hue:
	  case c_Hue_Auto:
	  case c_Gamma:
	  case c_Sharpness:
	  case c_Black_Level:
	  case c_White_Balance_Temperature:
	  case c_White_Balance_Auto:
	  case c_Red_Balance:
	  case c_Blue_Balance:
	  case c_HFlip:
	  case c_VFlip:
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
#ifdef V4L2_CID_IRIS_ABSOLUTE
	  case c_Iris_Absolute:
#endif
#ifdef V4L2_CID_IRIS_RELATIVE
	  case c_Iris_Relative:
#endif
	  case c_Pan_Absolute:
	  case c_Pan_Relative:
	  case c_Pan_Reset:
	  case c_Tilt_Absolute:
	  case c_Tilt_Relative:
	  case c_Tilt_Reset:
	  case c_Cid_Private0:
	  case c_Cid_Private1:
	  case c_Cid_Private2:
	  case c_Cid_Private3:
	  case c_Cid_Private4:
	  case c_Cid_Private5:
	  case c_Cid_Private6:
	  case c_Cid_Private7:
	  case c_Cid_Private8:
	  case c_Cid_Private9:
	  case c_Cid_Private10:
	  case c_Cid_Private11:
	  case c_Cid_Private12:
	  case c_Cid_Private13:
	  case c_Cid_Private14:
	  case c_Cid_Private15:
	  case c_Cid_Private16:
	  case c_Cid_Private17:
	  case c_Cid_Private18:
	  case c_Cid_Private19:
	  case c_Cid_Private20:
	  case c_Cid_Private21:
	  case c_Cid_Private22:
	  case c_Cid_Private23:
	  case c_Cid_Private24:
	  case c_Cid_Private25:
	  case c_Cid_Private26:
	  case c_Cid_Private27:
	  case c_Cid_Private28:
	  case c_Cid_Private29:
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

	  case Id_MouseMove:
	  {
	    ostringstream	s;
	    s << '(' << val.u << ',' << val.v << ')';
	    _menuCmd.setString(c_Cursor, s.str().c_str());
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

  //_camera.snap() >> _image;
    _camera.snap().captureRGBImage(_image);
  //printTime(std::cerr, _camera.arrivaltime()) << std::endl;
    
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
    if (_menuCmd.getValue(c_ContinuousShot))
    {
	_timer.stop();
	_camera.stopContinuousShot();
	_menuCmd.setValue(c_ContinuousShot, 0);
    }
}
 
}
}
