/*
 *  $Id: MyCmdWindow.h,v 1.2 2011-01-05 02:05:22 ueshiba Exp $
 */
#include "TU/v/App.h"
#include "TU/v/CmdWindow.h"
#include "TU/v/CmdPane.h"
#include "TU/v/FileSelection.h"
#include "TU/v/Timer.h"
#include "TU/v/vIeee1394++.h"
#include "TU/Movie.h"
#include "testcam.h"
#include "MyCanvasPane.h"

namespace TU
{
namespace v
{
/************************************************************************
*  class MyCmdWindow<PIXEL>						*
************************************************************************/
template <class PIXEL>
class MyCmdWindow : public CmdWindow
{
  public:
    MyCmdWindow(App&			parentApp,
		Ieee1394Camera&		camera,
		Ieee1394Camera::Type	type)				;

    virtual void	callback(CmdId, CmdVal)				;
    virtual void	tick()						;
    
  private:
    void	initializeMovie()					;
    void	repaintCanvases()					;
    void	setFrame()						;
    void	stopContinuousShotIfRunning()				;
    void	separateChannels(const Image<YUV422>& image)		;
    void	separateChannels(const Image<RGB>& image)		;
    
    Ieee1394Camera&		_camera;
    Movie<PIXEL>		_movie;
    Array<MyCanvasPane<PIXEL>*>	_canvases;
    CmdPane			_menuCmd;
    CmdPane			_captureCmd;
    CmdPane			_featureCmd;
    Timer			_timer;
};

template <class PIXEL>
MyCmdWindow<PIXEL>::MyCmdWindow(App&			parentApp,
				Ieee1394Camera&		camera,
				Ieee1394Camera::Type	type)
    :CmdWindow(parentApp, "Camera controller", Colormap::RGBColor, 16, 0, 0),
     _camera(camera),
     _movie(type == Ieee1394Camera::Monocular ? 1 :
	    type == Ieee1394Camera::Binocular ? 2 : 3),
     _canvases(0),
     _menuCmd(*this, createMenuCmds(_camera)),
     _captureCmd(*this, createCaptureCmds()),
     _featureCmd(*this, createFeatureCmds(_camera)),
     _timer(*this, 0)
{
    _menuCmd.place(0, 0, 2, 1);
    _captureCmd.place(0, 1, 1, 1);
    _featureCmd.place(1, 1, 1, 1);

    show();

    initializeMovie();
    _movie.setCircularMode(true);
}

template <class PIXEL> void
MyCmdWindow<PIXEL>::callback(CmdId id, CmdVal val)
{
    using namespace	std;

    try
    {
	if (handleCameraFormats(_camera, id, val))
	{
	    initializeMovie();
	    return;
	}
	else if (handleCameraSpecialFormats(_camera, id, val, *this))
	{
	    initializeMovie();
	    return;
	}
	else if (handleCameraFeatures(_camera, id, val))
	    return;
	
	switch (id)
	{
	  case M_Exit:
	    app().exit();
	    break;

	  case M_Save:
	  {
	    stopContinuousShotIfRunning();
	    setFrame();

	    FileSelection	fileSelection(*this);
	    ofstream		out;
	    if (fileSelection.open(out))
	    {
		for (size_t i = 0; i < _movie.nviews(); ++i)
		    _movie.image(i).save(out);
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
	    for (size_t i = 0; i < _canvases.size(); ++i)
		_canvases[i]->repaintUnderlay();
	    break;
	}
    }
    catch (exception& err)
    {
	cerr << err.what();
    }
}

template <class PIXEL> void
MyCmdWindow<PIXEL>::tick()
{
    static int			nframes = 0;
    static struct timeval	start;
    countTime(nframes, start);

    if (!_captureCmd.getValue(c_PlayMovie))
    {
	_camera.snap();
	switch (_movie.nviews())
	{
	  case 2:
	  {
	    Image<YUV422>	image;
	    _camera.captureDirectly(image);
	    separateChannels(image);
	  }
	    break;
	  case 3:
	  {
	    Image<RGB>		image;
	    _camera.captureDirectly(image);
	    separateChannels(image);
	  }
	    break;
	  default:
	    _camera >> _movie.image(0);
	    break;
	}
    }

    repaintCanvases();

    ++_movie;
}

template <class PIXEL> void
MyCmdWindow<PIXEL>::initializeMovie()
{
    Array<typename Movie<PIXEL>::Size>	sizes(_movie.nviews());
    for (size_t i = 0; i < sizes.size(); ++i)
	sizes[i] = std::make_pair(_camera.width(), _camera.height());
    _movie.setSizes(sizes);
    _movie.insert(_menuCmd.getValue(c_NFrames));

    if (_canvases.size() != _movie.nviews())
    {
	for (size_t i = 0; i < _canvases.size(); ++i)
	    delete _canvases[i];

	_canvases.resize(_movie.nviews());
	for (size_t i = 0; i < _canvases.size(); ++i)
	{
	    Image<PIXEL>&	image = _movie.image(i);
	    _canvases[i] = new MyCanvasPane<PIXEL>(*this, image.width(),
						   image.height(), image);
	    _canvases[i]->place(i % 2, 2 + i / 2, 1, 1);
	}
    }
    else
    {
	for (size_t i = 0; i < _canvases.size(); ++i)
	    _canvases[i]->resize();
    }

    int	props[] = {0, _movie.nframes() - 1, 1};
    _captureCmd.setProp(c_StatusMovie, props);

    repaintCanvases();
}

template <class PIXEL> void
MyCmdWindow<PIXEL>::repaintCanvases()
{
    for (size_t i = 0; i < _canvases.size(); ++i)
	_canvases[i]->repaintUnderlay();
    _captureCmd.setValue(c_StatusMovie, int(_movie.currentFrame()));
}

template <class PIXEL> void
MyCmdWindow<PIXEL>::setFrame()
{
    _movie.setFrame(_captureCmd.getValue(c_StatusMovie));
    for (size_t i = 0; i < _canvases.size(); ++i)
	_canvases[i]->repaintUnderlay();
}

template <class PIXEL> void
MyCmdWindow<PIXEL>::separateChannels(const Image<YUV422>& image)
{
    Image<PIXEL>&	imageL = _movie.image(0);
    Image<PIXEL>&	imageR = _movie.image(1);
    
    for (size_t v = 0; v < image.height(); ++v)
    {
	const YUV422*	p = image[v].data();
	PIXEL*		q = imageL[v].data();
	PIXEL*		r = imageR[v].data();
	for (size_t n = image.width(); n > 0; --n)
	{
	    *q++ = p->y;
	    *r++ = p->x;
	    ++p;
	}
    }
}

template <class PIXEL> void
MyCmdWindow<PIXEL>::separateChannels(const Image<RGB>& image)
{
    Image<PIXEL>&	imageC = _movie.image(0);
    Image<PIXEL>&	imageH = _movie.image(1);
    Image<PIXEL>&	imageV = _movie.image(2);

    for (size_t v = 0; v < image.height(); ++v)
    {
	const RGB*	p = image[v].data();
	PIXEL*		q = imageC[v].data();
	PIXEL*		r = imageH[v].data();
	PIXEL*		s = imageV[v].data();
	for (size_t n = image.width(); n > 0; --n)
	{
	    *q++ = p->r;
	    *r++ = p->g;
	    *s++ = p->b;
	    ++p;
	}
    }
}

template <class PIXEL> void
MyCmdWindow<PIXEL>::stopContinuousShotIfRunning()
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
