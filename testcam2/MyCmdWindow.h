/*
 *  $Id: MyCmdWindow.h,v 1.3 2011-01-05 02:04:50 ueshiba Exp $
 */
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
*  class MyCmdWindow<CAMERAS, PIXEL>					*
************************************************************************/
template <class CAMERAS, class PIXEL>
class MyCmdWindow : public CmdWindow
{
  private:
    typedef typename CAMERAS::value_type	camera_type;
    
  public:
    MyCmdWindow(App& parentApp, CAMERAS& cameras)		;

    virtual void	callback(CmdId, CmdVal)			;
    virtual void	tick()					;
    
  private:
    void		initializeMovie()			;
    void		repaintCanvases()			;
    void		setFrame()				;
    void		stopContinuousShotIfRunning()		;

  private:
    CAMERAS&			_cameras;
    Movie<PIXEL>		_movie;
    Array<MyCanvasPane<PIXEL>*>	_canvases;
    CmdPane			_menuCmd;
    CmdPane			_captureCmd;
    CmdPane			_featureCmd;
    Timer			_timer;
};

template <class CAMERAS, class PIXEL>
MyCmdWindow<CAMERAS, PIXEL>::MyCmdWindow(App& parentApp, CAMERAS& cameras)
    :CmdWindow(parentApp, "Camera controller", Colormap::RGBColor, 16, 0, 0),
     _cameras(cameras),
     _movie(size(_cameras)),
     _canvases(0),
     _menuCmd(*this, createMenuCmds(*std::begin(_cameras))),
     _captureCmd(*this, createCaptureCmds()),
     _featureCmd(*this, createFeatureCmds(*std::begin(_cameras), size(cameras))),
     _timer(*this, 0)
{
    _menuCmd.place(0, 0, 2, 1);
    _captureCmd.place(0, 1, 1, 1);
    _featureCmd.place(1, 1, 1, 1);

    show();

    refreshFeatureCmds(_cameras, _featureCmd);	// カメラ属性の初期値をGUIにセット
    
    initializeMovie();
    _movie.setCircularMode(true);
}

template <class CAMERAS, class PIXEL> void
MyCmdWindow<CAMERAS, PIXEL>::callback(CmdId id, CmdVal val)
{
    try
    {
	if (setFormat(_cameras, id, val, *this))
	{
	    initializeMovie();
	    return;
	}
	else if (setFeature(_cameras, id, val, _featureCmd))
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
	    std::ofstream	out;
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
		std::for_each(std::begin(_cameras), std::end(_cameras),
			      std::bind(&camera_type::continuousShot,
					std::placeholders::_1, true));
		_timer.start(1);
	    }
	    else
	    {
		_timer.stop();
		std::for_each(std::begin(_cameras), std::end(_cameras),
			      std::bind(&camera_type::continuousShot,
					std::placeholders::_1, false));
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
	    for (auto canvas : _canvases)
		canvas->repaintUnderlay();
	    break;
	}
    }
    catch (std::exception& err)
    {
	std::cerr << err.what() << std::endl;
    }
}

template <class CAMERAS, class PIXEL> void
MyCmdWindow<CAMERAS, PIXEL>::tick()
{
    countTime();

    if (!_captureCmd.getValue(c_PlayMovie))
    {
	std::for_each(std::begin(_cameras), std::end(_cameras),
		      std::bind(&camera_type::snap, std::placeholders::_1));
	for (size_t i = 0; i < size(_cameras); ++i)
	    _cameras[i] >> _movie.image(i);
    }

    repaintCanvases();

    ++_movie;
}

template <class CAMERAS, class PIXEL> void
MyCmdWindow<CAMERAS, PIXEL>::initializeMovie()
{
    Array<typename Movie<PIXEL>::Size>	sizes(size(_cameras));
    auto				camera = std::cbegin(_cameras);
    for (auto& size : sizes)
    {
	size = std::make_pair(camera->width(), camera->height());
	++camera;
    }
    _movie.setSizes(sizes);
    _movie.insert(_menuCmd.getValue(c_NFrames));

    if (_canvases.size() != _movie.nviews())
    {
	for (auto canvas : _canvases)
	    delete canvas;

	_canvases.resize(_movie.nviews());
	for (size_t i = 0; i < _canvases.size(); ++i)
	{
	    auto&	image = _movie.image(i);
	    _canvases[i] = new MyCanvasPane<PIXEL>(*this, image.width(),
						   image.height(), image);
	    _canvases[i]->place(i % 2, 2 + i / 2, 1, 1);
	}
    }
    else
    {
	for (auto canvas : _canvases)
	    canvas->resize();
    }

    float	props[] = {0, float(_movie.nframes()) - 1, 1};
    _captureCmd.setProp(c_StatusMovie, props);
    
    repaintCanvases();
}
    
template <class CAMERAS, class PIXEL> void
MyCmdWindow<CAMERAS, PIXEL>::repaintCanvases()
{
    for (auto canvas : _canvases)
	canvas->repaintUnderlay();
    _captureCmd.setValue(c_StatusMovie, _movie.currentFrame());
}

template <class CAMERAS, class PIXEL> void
MyCmdWindow<CAMERAS, PIXEL>::setFrame()
{
    _movie.setFrame(_captureCmd.getValue(c_StatusMovie));
    for (auto canvas : _canvases)
	canvas->repaintUnderlay();
}

template <class CAMERAS, class PIXEL> void
MyCmdWindow<CAMERAS, PIXEL>::stopContinuousShotIfRunning()
{
    if (_captureCmd.getValue(c_ContinuousShot))
    {
	_timer.stop();
	std::for_each(std::begin(_cameras), std::end(_cameras),
		      std::bind(&camera_type::continuousShot,
				std::placeholders::_1, false));
	_captureCmd.setValue(c_ContinuousShot, 0);
    }
}
    
}
}
