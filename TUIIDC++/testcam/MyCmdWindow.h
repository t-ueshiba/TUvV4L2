/*
 *  $Id: MyCmdWindow.h,v 1.3 2011-01-05 02:06:09 ueshiba Exp $
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
*  class MyCmdWindow<CAMERA, PIXEL>					*
************************************************************************/
template <class CAMERA, class PIXEL>
class MyCmdWindow : public CmdWindow
{
  public:
    MyCmdWindow(App& parentApp, CAMERA& camera)			;

    virtual void	callback(CmdId, CmdVal)			;
    virtual void	tick()					;
    
  private:
    void		initializeMovie()			;
    void		repaintCanvas()				;
    void		setFrame()				;
    void		stopContinuousShotIfRunning()		;

  private:
    CAMERA&		_camera;
    Movie<PIXEL>	_movie;
    MyCanvasPane<PIXEL>	_canvas;
    CmdPane		_menuCmd;
    CmdPane		_captureCmd;
    CmdPane		_featureCmd;
    Timer		_timer;
};
 
template <class CAMERA, class PIXEL>
MyCmdWindow<CAMERA, PIXEL>::MyCmdWindow(App& parentApp, CAMERA& camera)
    :CmdWindow(parentApp, "Camera controller", Colormap::RGBColor, 16, 0, 0),
     _camera(camera),
     _movie(1),
     _canvas(*this, _camera.width(), _camera.height(), _movie.image(0)),
     _menuCmd(*this, createMenuCmds(_camera)),
     _captureCmd(*this, createCaptureCmds()),
     _featureCmd(*this, createFeatureCmds(_camera)),
     _timer(*this, 0)
{
    _menuCmd.place(0, 0, 2, 1);
    _captureCmd.place(0, 1, 1, 1);
    _featureCmd.place(1, 1, 1, 1);
    _canvas.place(0, 2, 2, 1);

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
	cerr << "id = " << hex << id << dec << ", fval = " << val.f << endl;
	
	if (setFormat(_camera, id, val))
	{
	    initializeMovie();
	    return;
	}
	else if (setSpecialFormat(_camera, id, val, *this))
	{
	    initializeMovie();
	    return;
	}
<<<<<<< .mine
	else if (setFeature(_camera, id, int(val), val.f(), _featureCmd))
||||||| .r2044
	else if (setFeature(_camera, id, int(val), float(val)))
=======
	else if (setFeature(_camera, id, int(val), val.f))
>>>>>>> .r2054
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
		_movie.image(0).save(out);
	  }
	    break;

	  case c_ContinuousShot:
	    if (val)
	    {
		_camera.continuousShot(true);
		_timer.start(1);
	    }
	    else
	    {
		_timer.stop();
		_camera.continuousShot(false);
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

template <class CAMERA, class PIXEL> void
MyCmdWindow<CAMERA, PIXEL>::tick()
{
    countTime();

    if (!_captureCmd.getValue(c_PlayMovie))
    {
	_camera.snap() >> _movie.image(0);
#if 0
	printTime(std::cerr, _camera.arrivaltime()) << std::endl;
#endif
    }
    
    repaintCanvas();

    ++_movie;
}

template <class CAMERA, class PIXEL> void
MyCmdWindow<CAMERA, PIXEL>::initializeMovie()
{
    Array<typename Movie<PIXEL>::Size>	sizes(1);
    sizes[0] = std::make_pair(_camera.width(), _camera.height());
    _movie.setSizes(sizes);
    _movie.insert(_menuCmd.getValue(c_NFrames));

    _canvas.resize();

<<<<<<< .mine
    const float	props[] = {0, static_cast<float>(_movie.nframes() - 1), 1};
||||||| .r2044
    int	props[] = {0, static_cast<int>(_movie.nframes() - 1), 1};
=======
    float	props[] = {0, static_cast<float>(_movie.nframes() - 1), 1};
>>>>>>> .r2054
    _captureCmd.setProp(c_StatusMovie, props);
    
    repaintCanvas();
}
    
template <class CAMERA, class PIXEL> void
MyCmdWindow<CAMERA, PIXEL>::repaintCanvas()
{
    _canvas.repaintUnderlay();
<<<<<<< .mine
    _captureCmd.setValue(c_StatusMovie, _movie.currentFrame());
||||||| .r2044
    _captureCmd.setValue(c_StatusMovie, int(_movie.currentFrame()));
=======
    _captureCmd.setValue(c_StatusMovie, float(_movie.currentFrame()));
>>>>>>> .r2054
}

template <class CAMERA, class PIXEL> void
MyCmdWindow<CAMERA, PIXEL>::setFrame()
{
    _movie.setFrame(_captureCmd.getValue(c_StatusMovie));
    _canvas.repaintUnderlay();
}

template <class CAMERA, class PIXEL> void
MyCmdWindow<CAMERA, PIXEL>::stopContinuousShotIfRunning()
{
    if (_captureCmd.getValue(c_ContinuousShot))
    {
	_timer.stop();
	_camera.continuousShot(false);
	_captureCmd.setValue(c_ContinuousShot, 0);
    }
}

}
}
