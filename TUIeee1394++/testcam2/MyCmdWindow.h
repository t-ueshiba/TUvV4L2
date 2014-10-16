/*
 *  $Id: MyCmdWindow.h,v 1.3 2011-01-05 02:04:50 ueshiba Exp $
 */
#include <sys/time.h>
#include "TU/v/App.h"
#include "TU/v/CmdWindow.h"
#include "TU/v/CmdPane.h"
#include "TU/v/FileSelection.h"
#include "TU/v/Timer.h"
#include "TU/Movie.h"
#include "testcam.h"
#include "MyCanvasPane.h"

namespace TU
{
namespace v
{
/************************************************************************
*  class MyCmdWindow<CAMERA, PIXEL>					*
************************************************************************/
template <class CAMERA, class PIXEL>
class MyCmdWindow : public CmdWindow
{
  public:
    MyCmdWindow(App& parentApp, const Array<CAMERA*>& cameras)	;

    virtual void	callback(CmdId, CmdVal)			;
    virtual void	tick()					;
    
  private:
    void		initializeMovie()			;
    void		repaintCanvases()			;
    void		setFrame()				;
    void		stopContinuousShotIfRunning()		;

  private:
    const Array<CAMERA*>&	_cameras;
    Movie<PIXEL>		_movie;
    Array<MyCanvasPane<PIXEL>*>	_canvases;
    CmdPane			_menuCmd;
    CmdPane			_captureCmd;
    CmdPane			_featureCmd;
    Timer			_timer;
    int				_movieProp[3];
};

template <class CAMERA, class PIXEL>
MyCmdWindow<CAMERA, PIXEL>::MyCmdWindow(App& parentApp,
					const Array<CAMERA*>& cameras)
    :CmdWindow(parentApp, "Camera controller", Colormap::RGBColor, 16, 0, 0),
     _cameras(cameras),
     _movie(_cameras.size()),
     _canvases(0),
     _menuCmd(*this, createMenuCmds(*_cameras[0])),
     _captureCmd(*this, createCaptureCmds()),
     _featureCmd(*this, createFeatureCmds(_cameras)),
     _timer(*this, 0)
{
    _menuCmd.place(0, 0, 2, 1);
    _captureCmd.place(0, 1, 1, 1);
    _featureCmd.place(1, 1, 1, 1);

    show();

    initializeMovie();
    _movie.setCircularMode(true);
}

template <class CAMERA, class PIXEL> void
MyCmdWindow<CAMERA, PIXEL>::callback(CmdId id, CmdVal val)
{
    using namespace	std;

    try
    {
	if (handleCameraFormats(_cameras, id, val))
	{
	    initializeMovie();
	    return;
	}
	else if (handleCameraSpecialFormats(_cameras, id, val, *this))
	{
	    initializeMovie();
	    return;
	}
	else if (handleCameraFeatures(_cameras, id, val, _featureCmd))
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
		exec(_cameras, &CAMERA::continuousShot);
		_timer.start(1);
	    }
	    else
	    {
		_timer.stop();
		exec(_cameras, &CAMERA::stopContinuousShot);
	    }
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

template <class CAMERA, class PIXEL> void
MyCmdWindow<CAMERA, PIXEL>::tick()
{
    static int		nframes = 0;
    static timeval	start;
    countTime(nframes, start);

    if (!_captureCmd.getValue(c_PlayMovie))
    {
	exec(_cameras, &CAMERA::snap);
	for (size_t i = 0; i < _cameras.size(); ++i)
	    *_cameras[i] >> _movie.image(i);
    }

    repaintCanvases();

    ++_movie;
}

template <class CAMERA, class PIXEL> void
MyCmdWindow<CAMERA, PIXEL>::initializeMovie()
{
    Array<typename Movie<PIXEL>::Size>	sizes(_cameras.size());
    for (size_t i = 0; i < sizes.size(); ++i)
	sizes[i] = std::make_pair(_cameras[i]->width(), _cameras[i]->height());
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
	for (u_int i = 0; i < _canvases.size(); ++i)
	    _canvases[i]->resize();
    }

    _movieProp[0] = 0;
    _movieProp[1] = _movie.nframes() - 1;
    _movieProp[2] = 1;
    _captureCmd.setProp(c_StatusMovie, _movieProp);
    
    repaintCanvases();
}
    
template <class CAMERA, class PIXEL> void
MyCmdWindow<CAMERA, PIXEL>::repaintCanvases()
{
    for (size_t i = 0; i < _canvases.size(); ++i)
	_canvases[i]->repaintUnderlay();
    _captureCmd.setValue(c_StatusMovie, int(_movie.currentFrame()));
}

template <class CAMERA, class PIXEL> void
MyCmdWindow<CAMERA, PIXEL>::setFrame()
{
    _movie.setFrame(_captureCmd.getValue(c_StatusMovie));
    for (size_t i = 0; i < _canvases.size(); ++i)
	_canvases[i]->repaintUnderlay();
}

template <class CAMERA, class PIXEL> void
MyCmdWindow<CAMERA, PIXEL>::stopContinuousShotIfRunning()
{
    if (_captureCmd.getValue(c_ContinuousShot))
    {
	_timer.stop();
	exec(_cameras, &CAMERA::stopContinuousShot);
	_captureCmd.setValue(c_ContinuousShot, 0);
    }
}
    
}
}
