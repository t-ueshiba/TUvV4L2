/*
 *  $Id: MyCmdWindow.cc,v 1.7 2012-06-30 20:00:51 ueshiba Exp $
 */
#include <unistd.h>
#include <sys/time.h>
#include "multicam.h"
#include "MyCmdWindow.h"
#include "MyModalDialog.h"
#include <iomanip>

namespace TU
{
/************************************************************************
*  static functions							*
************************************************************************/

static void
countTime(int& nframes, timeval& start)
{
    using namespace	std;

    if (nframes == 10)
    {
	timeval	end;
	gettimeofday(&end, NULL);
	double	interval = (end.tv_sec  - start.tv_sec) +
	    (end.tv_usec - start.tv_usec) / 1.0e6;
	cerr << nframes / interval << " frames/sec" << endl;
	nframes = 0;
    }
    if (nframes++ == 0)
	gettimeofday(&start, NULL);
}

inline void
displayTime(const timeval& time)
{
    using namespace	std;
    
    time_t	sec = time.tv_sec;
    tm*		tm  = localtime(&sec);
    cerr << setfill('0')
	 << setw(2) << tm->tm_hour << ':'
	 << setw(2) << tm->tm_min  << ':'
	 << setw(2) << tm->tm_sec  << '.'
	 << setw(3) << time.tv_usec / 1000;
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
MyCmdWindow::MyCmdWindow(App&				parentApp,
			 const Array<Ieee1394Camera*>&	cameras,
			 bool				sync
#ifdef UseTrigger
			 , TriggerGenerator&		trigger
#endif
			)
    :CmdWindow(parentApp, "Ieee1394 camera controller",
	       0, Colormap::RGBColor, 16, 0, 0),
     _cameras(cameras), _sync(sync),
#ifdef UseTrigger
     _trigger(trigger),
#endif
     _movie(_cameras.dim()),
     _canvases(0),
     _menuCmd(*this, createMenuCmds(*_cameras[0])),
     _captureCmd(*this, createCaptureCmds()),
     _featureCmd(*this, createFeatureCmds(*_cameras[0])),
     _fileSelection(*this),
     _timer(*this, 0)
{
    for (int i = 0; i < cameras.dim(); ++i)
	_cameras[i]->turnOff(Ieee1394Camera::TRIGGER_MODE)
		    .setTriggerMode(Ieee1394Camera::Trigger_Mode0);

    _menuCmd.place(0, 0, 2, 1);
    _captureCmd.place(0, 1, 1, 1);
    _featureCmd.place(1, 1, 1, 1);

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
	    {
		for (int i = 0; i < _movie.nviews(); ++i)
		    _movie.image(i).save(out);
	    }
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
	    for (int i = 0; i < _cameras.dim(); ++i)
		_cameras[i]
		  ->setFormatAndFrameRate(Ieee1394Camera::uintToFormat(id),
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
	    Ieee1394Camera::Format
			format7 = Ieee1394Camera::uintToFormat(id);
	    Ieee1394Camera::Format_7_Info
			fmt7info = _cameras[0]->getFormat_7_Info(format7);
	    MyModalDialog	modalDialog(*this, fmt7info);
	    u_int		u0, v0, width, height;
	    Ieee1394Camera::PixelFormat
		pixelFormat = modalDialog.getROI(u0, v0, width, height);
	    for (int i = 0; i < _cameras.dim(); ++i)
		_cameras[i]->setFormat_7_ROI(format7, u0, v0, width, height)
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
	  case c_Zoom:
	    for (int i = 0; i < _cameras.dim(); ++i)
		_cameras[i]->setValue(id2feature(id), val);
	    break;
      
	  case c_WhiteBalance_UB:
	    for (int i = 0; i < _cameras.dim(); ++i)
		_cameras[i]->setWhiteBalance(
		    val, _featureCmd.getValue(c_WhiteBalance_VR));
	    break;
	  case c_WhiteBalance_VR:
	    for (int i = 0; i < _cameras.dim(); ++i)
		_cameras[i]->setWhiteBalance(
		    _featureCmd.getValue(c_WhiteBalance_UB), val);
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
	  case c_Zoom		 + OFFSET_ONOFF:
	  {
	    Ieee1394Camera::Feature feature = id2feature(id - OFFSET_ONOFF);
	    if (val)
		for (int i = 0; i < _cameras.dim(); ++i)
		    _cameras[i]->turnOn(feature);
	    else
		for (int i = 0; i < _cameras.dim(); ++i)
		    _cameras[i]->turnOff(feature);
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
	  case c_Zoom		 + OFFSET_AUTO:
	  {
	    Ieee1394Camera::Feature feature = id2feature(id - OFFSET_AUTO);
	    if (val)
		for (int i = 0; i < _cameras.dim(); ++i)
		    _cameras[i]->setAutoMode(feature);
	    else
		for (int i = 0; i < _cameras.dim(); ++i)
		{
		    _cameras[i]->setManualMode(feature);
		    if (feature == Ieee1394Camera::WHITE_BALANCE)
			_cameras[i]->setWhiteBalance(
			    _featureCmd.getValue(c_WhiteBalance_UB),
			    _featureCmd.getValue(c_WhiteBalance_VR));
		    else
			_cameras[i]->setValue(
			    feature, _featureCmd.getValue(id - OFFSET_AUTO));
		}
	  }
	    break;

	  case c_ContinuousShot:
	    if (val)
	    {
		for (int i = 0; i < _cameras.dim(); ++i)
		    _cameras[i]->continuousShot();
		_timer.start(1);
	    }
	    else
	    {
		_timer.stop();
		for (int i = 0; i < _cameras.dim(); ++i)
		    _cameras[i]->stopContinuousShot();
	    }
	    break;
	
	  case c_OneShot:
	    stopContinuousShotIfRunning();
	    for (int i = 0; i < _cameras.dim(); ++i)
		_cameras[i]->oneShot();
#ifdef UseTrigger
	    if (_captureCmd.getValue(c_Trigger))
		_trigger.oneShot();
#endif
	    tick();
	    break;

	  case c_Trigger:
	    if (val)
	    {
#ifdef UseTrigger
		_trigger.selectChannel(0xffffffff);
#endif
		for (int i = 0; i < _cameras.dim(); ++i)
		    _cameras[i]->turnOn(Ieee1394Camera::TRIGGER_MODE);
	    }
	    else
		for (int i = 0; i < _cameras.dim(); ++i)
		    _cameras[i]->turnOff(Ieee1394Camera::TRIGGER_MODE);
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
	    repaintCanvases();
	    break;
	
	  case c_BackwardMovie:
	    stopContinuousShotIfRunning();
	    if (!--_movie)
		_movie.rewind();
	    repaintCanvases();
	    break;
	
	  case c_StatusMovie:
	    stopContinuousShotIfRunning();
	    _movie.setFrame(val);
	    for (u_int i = 0; i < _canvases.dim(); ++i)
		_canvases[i]->repaintUnderlay();
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
	syncronizedSnap();
	for (int i = 0; i < _cameras.dim(); ++i)
	    *_cameras[i] >> _movie.image(i);
    }

    repaintCanvases();

    ++_movie;
}

void
MyCmdWindow::initializeMovie()
{
    Array<Movie<PixelType>::Size >	sizes(_cameras.dim());
    for (u_int i = 0; i < sizes.dim(); ++i)
	sizes[i] = std::make_pair(_cameras[i]->width(), _cameras[i]->height());
    _movie.setSizes(sizes);
    _movie.insert(_menuCmd.getValue(c_NFrames));

    if (_canvases.dim() != _movie.nviews())
    {
	for (u_int i = 0; i < _canvases.dim(); ++i)
	    delete _canvases[i];

	_canvases.resize(_movie.nviews());
	for (u_int i = 0; i < _canvases.dim(); ++i)
	{
	    Image<PixelType>&	image = _movie.image(i);
	    _canvases[i] = new MyCanvasPane(*this,
					    image.width(), image.height(),
					    image);
	    _canvases[i]->place(i % 2, 2 + i / 2, 1, 1);
	}
    }
    else
    {
	for (u_int i = 0; i < _canvases.dim(); ++i)
	    _canvases[i]->resize();
    }

    movieProp[0] = 0;
    movieProp[1] = _movie.nframes() - 1;
    movieProp[2] = 1;
    _captureCmd.setProp(c_StatusMovie, movieProp);
    
    repaintCanvases();
}
    
void
MyCmdWindow::repaintCanvases()
{
    for (u_int i = 0; i < _canvases.dim(); ++i)
	_canvases[i]->repaintUnderlay();
    _captureCmd.setValue(c_StatusMovie, int(_movie.currentFrame()));
}

void
MyCmdWindow::setFrame()
{
    _movie.setFrame(_captureCmd.getValue(c_StatusMovie));
    for (u_int i = 0; i < _canvases.dim(); ++i)
	_canvases[i]->repaintUnderlay();
}

void
MyCmdWindow::stopContinuousShotIfRunning()
{
    if (_captureCmd.getValue(c_ContinuousShot))
    {
	_timer.stop();
	for (int i = 0; i < _cameras.dim(); ++i)
	    _cameras[i]->stopContinuousShot();
	_captureCmd.setValue(c_ContinuousShot, 0);
    }
}

void
MyCmdWindow::syncronizedSnap()
{
#if !defined(__APPLE__)
    const u_int64_t	margin = 2000;
    u_int64_t		last = 0;
    for (u_int i = 0; i < _cameras.dim(); ++i)
    {
	u_int64_t	arrivaltime = _cameras[i]->snap().arrivaltime();
	if (last + margin < arrivaltime)
	{
	    last = arrivaltime;
	    for (u_int j = 0; j < i; ++j)
		do
		{
		    arrivaltime = _cameras[j]->snap().arrivaltime();
		} while (arrivaltime + margin < last);
	}
	else if (arrivaltime + margin < last)
	    do
	    {
		arrivaltime = _cameras[i]->snap().arrivaltime();
	    } while (arrivaltime + margin < last);
    }
#else
    for (int i = 0; i < _cameras.dim(); ++i)
	_cameras[i]->snap();
#endif
}

}
}
