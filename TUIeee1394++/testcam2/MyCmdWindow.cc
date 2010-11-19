/*
 *  $Id: MyCmdWindow.cc,v 1.2 2010-11-19 02:14:36 ueshiba Exp $
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

inline u_int64_t
timeval2u_int64(const timeval& time)
{
    return u_int64_t(time.tv_sec) * 1000000LL + u_int64_t(time.tv_usec);
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
     _movie(),
     _menuCmd(*this, createMenuCmds(*_cameras[0])),
     _captureCmd(*this, createCaptureCmds()),
     _featureCmd(*this, createFeatureCmds(*_cameras[0])),
     _fileSelection(*this),
     _images(_cameras.dim() > 3 ? _cameras.dim() : 3),
     _canvasC(*this, 320, 240, _images[0]),
     _canvasH(*this, 320, 240, _images[1]),
     _canvasV(*this, 320, 240, _images[2]),
     _timer(*this, 0)
{
    for (int i = 0; i < cameras.dim(); ++i)
	_cameras[i]->turnOff(Ieee1394Camera::TRIGGER_MODE)
		    .setTriggerMode(Ieee1394Camera::Trigger_Mode0);

    _menuCmd.place(0, 0, 2, 1);
    _captureCmd.place(0, 1, 1, 1);
    _featureCmd.place(1, 1, 1, 2);
    _canvasC.place(0, 3, 1, 1);
    _canvasH.place(1, 3, 1, 1);
    _canvasV.place(0, 2, 1, 1);

    show();
}

void
MyCmdWindow::callback(CmdId id, CmdVal val)
{
    using namespace	std;
    
    switch (id)
    {
      case M_Exit:
	app().exit();
	break;

      case M_Save:
      {
	ofstream	out;
	if (_fileSelection.open(out))
	{
	    for (int i = 0; i < _cameras.dim(); ++i)
		_images[i].save(out);
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
	_canvasC.resize(_cameras[0]->width(), _cameras[0]->height());
	_canvasH.resize(_cameras[0]->width(), _cameras[0]->height());
	_canvasV.resize(_cameras[0]->width(), _cameras[0]->height());
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
	Ieee1394Camera::Format	format7 = Ieee1394Camera::uintToFormat(id);
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
      }
	_canvasC.resize(_cameras[0]->width(), _cameras[0]->height());
	_canvasH.resize(_cameras[0]->width(), _cameras[0]->height());
	_canvasV.resize(_cameras[0]->width(), _cameras[0]->height());
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
	    _cameras[i]
	      ->setWhiteBalance(val, _featureCmd.getValue(c_WhiteBalance_VR));
	break;
      case c_WhiteBalance_VR:
	for (int i = 0; i < _cameras.dim(); ++i)
	    _cameras[i]
	      ->setWhiteBalance(_featureCmd.getValue(c_WhiteBalance_UB), val);
	break;
      
      case c_Brightness	     + OFFSET_ONOFF:
      case c_AutoExposure    + OFFSET_ONOFF:
      case c_Sharpness	     + OFFSET_ONOFF:
      case c_WhiteBalance_UB + OFFSET_ONOFF:
      case c_WhiteBalance_VR + OFFSET_ONOFF:
      case c_Hue	     + OFFSET_ONOFF:
      case c_Saturation	     + OFFSET_ONOFF:
      case c_Gamma	     + OFFSET_ONOFF:
      case c_Shutter	     + OFFSET_ONOFF:
      case c_Gain	     + OFFSET_ONOFF:
      case c_Iris	     + OFFSET_ONOFF:
      case c_Focus	     + OFFSET_ONOFF:
      case c_Zoom	     + OFFSET_ONOFF:
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
      
      case c_Brightness	     + OFFSET_AUTO:
      case c_AutoExposure    + OFFSET_AUTO:
      case c_Sharpness	     + OFFSET_AUTO:
      case c_WhiteBalance_UB + OFFSET_AUTO:
      case c_WhiteBalance_VR + OFFSET_AUTO:
      case c_Hue	     + OFFSET_AUTO:
      case c_Saturation	     + OFFSET_AUTO:
      case c_Gamma	     + OFFSET_AUTO:
      case c_Shutter	     + OFFSET_AUTO:
      case c_Gain	     + OFFSET_AUTO:
      case c_Iris	     + OFFSET_AUTO:
      case c_Focus	     + OFFSET_AUTO:
      case c_Zoom	     + OFFSET_AUTO:
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
		    _cameras[i]->
		      setWhiteBalance(_featureCmd.getValue(c_WhiteBalance_UB),
				      _featureCmd.getValue(c_WhiteBalance_VR));
		else
		    _cameras[i]->
		      setValue(feature,
			       _featureCmd.getValue(id - OFFSET_AUTO));
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
#ifdef UseTrigger
	if (_captureCmd.getValue(c_Trigger))
	    _trigger.oneShot();
#endif
	snapMulti();
	for (int i = 0; i < _cameras.dim(); ++i)
	{
	    MyCanvasPane&	canvas = (i == 0 ? _canvasC :
					  i == 1 ? _canvasH : _canvasV);
	    *_cameras[i] >> _images[i];
	    canvas.repaintUnderlay();
	}
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
	
      case c_PlayMovie:
	stopContinuousShotIfRunning();
	_canvasC.resize(_movie.width(), _movie.height());
	_canvasH.resize(_movie.width(), _movie.height());
	_canvasV.resize(_movie.width(), _movie.height());
	for (_movie.rewind(); _movie; ++_movie)
	{
	    static int			nframes = 0;
	    static struct timeval	start;
	    countTime(nframes, start);

	    showMovieFrames();
	    _captureCmd.setValue(c_StatusMovie, int(_movie.currentFrame()));
	}
	break;
	
      case c_RecordMovie:
      {
	stopContinuousShotIfRunning();
#ifdef MONO_IMAGE
	if (_cameras[0]->pixelFormat() != Ieee1394Camera::MONO_8)
	{
	    cerr << "Only MONO(8 bits) format is supported for movie!" << endl;
	    break;
	}
#else
	if (_cameras[0]->pixelFormat() != Ieee1394Camera::YUV_422)
	{
	    cerr << "Only YUV(4:2:2) format is supported for movie!" << endl;
	    break;
	}
#endif
	Array<pair<u_int, u_int> >	sizes(_cameras.dim());
	for (int i = 0; i < sizes.dim(); ++i)
	    sizes[i] = make_pair<u_int, u_int>(_cameras[i]->width(),
					       _cameras[i]->height());
	_movie.alloc(sizes, _menuCmd.getValue(c_NFrames));
	movieProp[0] = 0;
	movieProp[1] = _movie.nframes() - 1;
	movieProp[2] = 1;
	_captureCmd.setProp(c_StatusMovie, movieProp);
      	for (int i = 0; i < _cameras.dim(); ++i)
	    _cameras[i]->continuousShot();
	for (_movie.rewind(); _movie; ++_movie)
	{
	    static int			nframes = 0;
	    static struct timeval	start;
	    countTime(nframes, start);
#ifdef UseTrigger
	    if (_captureCmd.getValue(c_Trigger))
		_trigger.oneShot();
#endif
	    snapMulti();
	    for (int i = 0; i < _cameras.dim(); ++i)
	    {
		timeval	filltime = _cameras[i]->filltime();
		cerr << ' ';
		displayTime(filltime);
	    }
	    cerr << endl;
	    for (int i = 0; i < _cameras.dim(); ++i)
	    {
		_movie.setView(i);
		*_cameras[i] >> _movie.image();
	    }
	    _captureCmd.setValue(c_StatusMovie, int(_movie.currentFrame()));
	}
	for (int i = 0; i < _cameras.dim(); ++i)
	    _cameras[i]->stopContinuousShot();
      }
	break;

      case c_ForwardMovie:
      {
	int	frame = _captureCmd.getValue(c_StatusMovie) + 1;
	if (frame < _movie.nframes())
	{
	    stopContinuousShotIfRunning();
	    _canvasC.resize(_movie.width(), _movie.height());
	    _canvasH.resize(_movie.width(), _movie.height());
	    _canvasV.resize(_movie.width(), _movie.height());
	    _movie.setFrame(frame);
	    showMovieFrames();
	    _captureCmd.setValue(c_StatusMovie, frame);
	}
      }
        break;
	
      case c_BackwardMovie:
      {
	int	frame = _captureCmd.getValue(c_StatusMovie) - 1;
	if (frame >= 0)
	{
	    stopContinuousShotIfRunning();
	    _canvasC.resize(_movie.width(), _movie.height());
	    _canvasH.resize(_movie.width(), _movie.height());
	    _canvasV.resize(_movie.width(), _movie.height());
	    _movie.setFrame(frame);
	    showMovieFrames();
	    _captureCmd.setValue(c_StatusMovie, frame);
	}
      }
        break;
	
      case c_StatusMovie:
	stopContinuousShotIfRunning();
	_canvasC.resize(_movie.width(), _movie.height());
	_canvasH.resize(_movie.width(), _movie.height());
	_canvasV.resize(_movie.width(), _movie.height());
	_movie.setFrame(val);
	showMovieFrames();
	break;
    }
}

void
MyCmdWindow::tick()
{
    static int			nframes = 0;
    static struct timeval	start;
    countTime(nframes, start);
#ifdef UseTrigger
    if (_captureCmd.getValue(c_Trigger))
	_trigger.oneShot();
#endif
    snapMulti();
    for (int i = 0; i < _cameras.dim(); ++i)
    {
	MyCanvasPane&	canvas = (i == 0 ? _canvasC :
				  i == 1 ? _canvasH : _canvasV);
	*_cameras[i] >> _images[i];
	canvas.repaintUnderlay();
/*	timeval	filltime = _cameras[i]->filltime();
	std::cerr << ' ';
	displayTime(filltime);*/
    }
//    std::cerr << std::endl;
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
MyCmdWindow::showMovieFrames()
{
    for (int i = 0; i < _movie.nviews(); ++i)
    {
	MyCanvasPane&	canvas = (i == 0 ? _canvasC :
				  i == 1 ? _canvasH : _canvasV);
	_movie.setView(i);
	_images[i] = _movie.image();
	canvas.repaintUnderlay();
    }
}

void
MyCmdWindow::snapMulti()
{
    if (_sync)
    {
	const u_int64_t	margin = 2000;
	u_int64_t		last = 0;
	for (int i = 0; i < _cameras.dim(); ++i)
	{
	    u_int64_t	filltime = timeval2u_int64(_cameras[i]
						   ->snap().filltime());
	    if (last + margin < filltime)
	    {
		last = filltime;
		for (int j = 0; j < i; ++j)
		    do
		    {
			filltime = timeval2u_int64(_cameras[j]
						   ->snap().filltime());
		    } while (filltime + margin < last);
	    }
	    else if (filltime + margin < last)
		do
		{
		    filltime = timeval2u_int64(_cameras[i]->snap().filltime());
		} while (filltime + margin < last);
	}
    }
    else
	for (int i = 0; i < _cameras.dim(); ++i)
	    _cameras[i]->snap();
}

}
}
