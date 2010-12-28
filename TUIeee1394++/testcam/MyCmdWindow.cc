/*
 *  $Id: MyCmdWindow.cc,v 1.2 2010-12-28 11:47:48 ueshiba Exp $
 */
#include <unistd.h>
#include <sys/time.h>
#include "multicam.h"
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

namespace v
{
CmdDef*		createMenuCmds(Ieee1394Camera& camera)		;
CmdDef*		createCaptureCmds()				;
CmdDef*		createFeatureCmds(const Ieee1394Camera& camera)	;
    
/************************************************************************
*  static data								*
************************************************************************/
static int	movieProp[3];

/************************************************************************
*  class MyCmdWindow							*
************************************************************************/
MyCmdWindow::MyCmdWindow(App&			parentApp,
			 Ieee1394Camera&	camera
#ifdef UseTrigger
			 , TriggerGenerator&	trigger
#endif
			)
    :CmdWindow(parentApp, "Ieee1394 camera controller",
	       0, Colormap::RGBColor, 16, 0, 0),
     _camera(camera),
#ifdef UseTrigger
     _trigger(trigger),
#endif
     _movie(1),
     _canvas(*this, _camera.width(), _camera.height(), _movie.image()),
     _menuCmd(*this, createMenuCmds(_camera)),
     _captureCmd(*this, createCaptureCmds()),
     _featureCmd(*this, createFeatureCmds(_camera)),
     _fileSelection(*this),
     _timer(*this, 0)
{
    _camera.turnOff(Ieee1394Camera::TRIGGER_MODE)
	   .setTriggerMode(Ieee1394Camera::Trigger_Mode0);
	   
    _menuCmd.place(0, 0, 2, 1);
    _captureCmd.place(0, 1, 1, 1);
    _featureCmd.place(1, 1, 1, 1);
    _canvas.place(0, 2, 2, 1);

    show();

    initializeMovie();
    _movie.setCircularMode(true);
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
	    setFrame();
	    
	    ofstream	out;
	    if (_fileSelection.open(out))
		_movie.image().save(out);
	  }
	    break;
	
	  case c_YUV444_160x120:
	  case c_YUV422_320x240:
	  case c_YUV411_640x480:
	  case c_YUV422_640x480:
	  case c_RGB24_640x480:
	  case c_MONO8_640x480:
	  case c_MONO16_640x480:
	  case c_YUV422_800x600:
	  case c_RGB24_800x600:
	  case c_MONO8_800x600:
	  case c_YUV422_1024x768:
	  case c_RGB24_1024x768:
	  case c_MONO8_1024x768:
	  case c_MONO16_800x600:
	  case c_MONO16_1024x768:
	  case c_YUV422_1280x960:
	  case c_RGB24_1280x960:
	  case c_MONO8_1280x960:
	  case c_YUV422_1600x1200:
	  case c_RGB24_1600x1200:
	  case c_MONO8_1600x1200:
	  case c_MONO16_1280x960:
	  case c_MONO16_1600x1200:
	    _camera.setFormatAndFrameRate(Ieee1394Camera::uintToFormat(id),
					  Ieee1394Camera::uintToFrameRate(val));
	    initializeMovie();
	    break;

	  case c_Format_7_0:
	  case c_Format_7_1:
	  case c_Format_7_2:
	  case c_Format_7_3:
	  case c_Format_7_4:
	  case c_Format_7_5:
	  case c_Format_7_6:
	  case c_Format_7_7:
	  {
	    Ieee1394Camera::Format	format7  = Ieee1394Camera::uintToFormat(id);
	    Ieee1394Camera::Format_7_Info
		fmt7info = _camera.getFormat_7_Info(format7);
	    MyModalDialog		modalDialog(*this, fmt7info);
	    u_int			u0, v0, width, height;
	    Ieee1394Camera::PixelFormat
		pixelFormat = modalDialog.getROI(u0, v0, width, height);
	    _camera.setFormat_7_ROI(format7, u0, v0, width, height)
		   .setFormat_7_PixelFormat(format7, pixelFormat)
		   .setFormatAndFrameRate(format7,
					  Ieee1394Camera::uintToFrameRate(val));
	    initializeMovie();
	  }
	    break;
	
	  case c_Brightness:
	  case c_AutoExposure:
	  case c_Sharpness:
	  case c_Hue:
	  case c_Saturation:
	  case c_Gamma:
	  case c_Shutter:
	  case c_Gain:
	  case c_Iris:
	  case c_Focus:
	  case c_Temperature:
	  case c_Zoom:
	    _camera.setValue(id2feature(id), val);
	    break;

	  case c_WhiteBalance_UB:
	    _camera.setWhiteBalance(val, _featureCmd.getValue(c_WhiteBalance_VR));
	    break;
	  case c_WhiteBalance_VR:
	    _camera.setWhiteBalance(_featureCmd.getValue(c_WhiteBalance_UB), val);
	    break;
      
	  case c_Brightness	 + OFFSET_ONOFF:
	  case c_AutoExposure    + OFFSET_ONOFF:
	  case c_Sharpness	 + OFFSET_ONOFF:
	  case c_WhiteBalance_UB + OFFSET_ONOFF:
	  case c_WhiteBalance_VR + OFFSET_ONOFF:
	  case c_Hue		 + OFFSET_ONOFF:
	  case c_Saturation	 + OFFSET_ONOFF:
	  case c_Gamma		 + OFFSET_ONOFF:
	  case c_Shutter	 + OFFSET_ONOFF:
	  case c_Gain		 + OFFSET_ONOFF:
	  case c_Iris		 + OFFSET_ONOFF:
	  case c_Focus		 + OFFSET_ONOFF:
	  case c_Temperature     + OFFSET_ONOFF:
	  case c_Zoom		 + OFFSET_ONOFF:
	  {
	    Ieee1394Camera::Feature feature = id2feature(id - OFFSET_ONOFF);
	    if (val)
		_camera.turnOn(feature);
	    else
		_camera.turnOff(feature);
	  }
	    break;
      
	  case c_Brightness	 + OFFSET_AUTO:
	  case c_AutoExposure    + OFFSET_AUTO:
	  case c_Sharpness	 + OFFSET_AUTO:
	  case c_WhiteBalance_UB + OFFSET_AUTO:
	  case c_WhiteBalance_VR + OFFSET_AUTO:
	  case c_Hue		 + OFFSET_AUTO:
	  case c_Saturation	 + OFFSET_AUTO:
	  case c_Gamma		 + OFFSET_AUTO:
	  case c_Shutter	 + OFFSET_AUTO:
	  case c_Gain		 + OFFSET_AUTO:
	  case c_Iris		 + OFFSET_AUTO:
	  case c_Focus		 + OFFSET_AUTO:
	  case c_Temperature     + OFFSET_AUTO:
	  case c_Zoom		 + OFFSET_AUTO:
	  {
	    Ieee1394Camera::Feature feature = id2feature(id - OFFSET_AUTO);
	    if (val)
		_camera.setAutoMode(feature);
	    else
	    {
		_camera.setManualMode(feature);
		if (feature == Ieee1394Camera::WHITE_BALANCE)
		    _camera
			.setWhiteBalance(_featureCmd.getValue(c_WhiteBalance_UB),
					 _featureCmd.getValue(c_WhiteBalance_VR));
		else
		    _camera.setValue(feature,
				     _featureCmd.getValue(id - OFFSET_AUTO));
	    }
	  }
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
	
	  case c_OneShot:
	    stopContinuousShotIfRunning();
	    _camera.oneShot();
	    tick();
	    break;

	  case c_Trigger:
	    if (val)
	    {
#ifdef UseTrigger
		_trigger.selectChannel(0xffffffff);
#endif
		_camera.turnOn(Ieee1394Camera::TRIGGER_MODE);
	    }
	    else
		_camera.turnOff(Ieee1394Camera::TRIGGER_MODE);
	    break;

	  case c_NFrames:
	    stopContinuousShotIfRunning();
	    initializeMovie();
	    break;
	    
	  case c_PlayMovie:
	    stopContinuousShotIfRunning();
	    if (val)
		_timer.start(10);
	    else
		_timer.stop();
	    break;

	  case c_ForwardMovie:
	    stopContinuousShotIfRunning();
	    if (!++_movie)
		--_movie;
	    repaintCanvas();
	    break;
	
	  case c_BackwardMovie:
	    stopContinuousShotIfRunning();
	    if (!--_movie)
		_movie.rewind();
	    repaintCanvas();
	    break;
	
	  case c_StatusMovie:
	    stopContinuousShotIfRunning();
	    _movie.setFrame(val);
	    _canvas.repaintUnderlay();
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

    if (!_captureCmd.getValue(c_PlayMovie))
    {
#ifdef UseTrigger
	if (_captureCmd.getValue(c_Trigger))
	    _trigger.oneShot();
#endif
	_camera.snap() >> _movie.image();
    }
    
    repaintCanvas();

    ++_movie;
}

void
MyCmdWindow::initializeMovie()
{
    Array<Movie<PixelType>::Size >	sizes(1);
    sizes[0] = std::make_pair(_camera.width(), _camera.height());
    _movie.setSizes(sizes);
    _movie.insert(_menuCmd.getValue(c_NFrames));

    _canvas.resize();

    movieProp[0] = 0;
    movieProp[1] = _movie.nframes() - 1;
    movieProp[2] = 1;
    _captureCmd.setProp(c_StatusMovie, movieProp);
    
    repaintCanvas();
}
    
void
MyCmdWindow::repaintCanvas()
{
    _canvas.repaintUnderlay();
    _captureCmd.setValue(c_StatusMovie, int(_movie.currentFrame()));
}

void
MyCmdWindow::setFrame()
{
    _movie.setFrame(_captureCmd.getValue(c_StatusMovie));
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
