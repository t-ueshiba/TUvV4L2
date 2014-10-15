/*
 *  $Id: MyCmdWindow.h,v 1.3 2011-01-05 02:06:09 ueshiba Exp $
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
/************************************************************************
*  static functions							*
************************************************************************/
static inline void
countTime(int& nframes, timeval& start)
{
    if (nframes == 10)
    {
	timeval	end;
	gettimeofday(&end, NULL);
	double	interval = (end.tv_sec  - start.tv_sec) +
	    (end.tv_usec - start.tv_usec) / 1.0e6;
	std::cerr << nframes / interval << " frames/sec" << std::endl;
	nframes = 0;
    }
    if (nframes++ == 0)
	gettimeofday(&start, NULL);
}

static inline std::ostream&
printTime(std::ostream& out, u_int64_t localtime)
{
    u_int64_t	usec = localtime % 1000;
    u_int64_t	msec = (localtime / 1000) % 1000;
    u_int64_t	sec  = localtime / 1000000;
    return out << sec << '.' << msec << '.' << usec;
}

namespace v
{
/************************************************************************
*  local data								*
************************************************************************/
static MenuDef nframesMenu[] =
{
    {" 10",  10, false, noSub},
    {"100", 100, true,  noSub},
    {"300", 300, false, noSub},
    {"600", 600, false, noSub},
    EndOfMenu
};

static MenuDef fileMenu[] =
{
    {"Save",			M_Save,		 false, noSub},
    {"Restore camera config.",	c_RestoreConfig, false, noSub},
    {"Save camera config.",	c_SaveConfig,	 false, noSub},
    {"-",			M_Line,		 false, noSub},
    {"Quit",			M_Exit,		 false, noSub},
    EndOfMenu
};

static CmdDef menuCmds[] =
{
    {C_MenuButton, M_File,   0, "File",   fileMenu, CA_None, 0, 0, 1, 1, 0},
    {C_MenuButton, M_Format, 0, "Format", noProp,   CA_None, 1, 0, 1, 1, 0},
    {C_ChoiceMenuButton, c_NFrames, 100, "# of movie frames", nframesMenu,
     CA_None, 2, 0, 1, 1, 0},
    EndOfCmds
};

static CmdDef captureCmds[] =
{
    {C_ToggleButton,  c_ContinuousShot, 0, "Continuous shot", noProp, CA_None,
     0, 0, 1, 1, 0},
    {C_Button,	      c_OneShot,        0, "One shot",	      noProp, CA_None,
     0, 1, 1, 1, 0},
    {C_ToggleButton,  c_PlayMovie,	0, "Play",	      noProp, CA_None,
     1, 0, 1, 1, 0},
    {C_Button,  c_BackwardMovie, 0, "<",    noProp, CA_None, 2, 0, 1, 1, 0},
    {C_Button,  c_ForwardMovie,  0, ">",    noProp, CA_None, 3, 0, 1, 1, 0},
    {C_Slider,  c_StatusMovie,   0, "",     noProp, CA_None, 1, 1, 3, 1, 0},
    EndOfCmds
};

template <class CAMERA> static CmdDef*
createMenuCmds(const CAMERA& camera)
{
    menuCmds[1].prop = createFormatMenu(camera);
    
    return menuCmds;
}

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
    int			_movieProp[3];
};
 
template <class CAMERA, class PIXEL>
MyCmdWindow<CAMERA, PIXEL>::MyCmdWindow(App& parentApp, CAMERA& camera)
    :CmdWindow(parentApp, "Camera controller", Colormap::RGBColor, 16, 0, 0),
     _camera(camera),
     _movie(1),
     _canvas(*this, _camera.width(), _camera.height(), _movie.image(0)),
     _menuCmd(*this, createMenuCmds(_camera)),
     _captureCmd(*this, captureCmds),
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
	if (handleCameraFormats(_camera, id, val))
	{
	    initializeMovie();
	    return;
	}
	else if (handleCameraSpecialFormat(_camera, id, val, *this))
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
		_movie.image(0).save(out);
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
	  /*
	  case c_OneShot:
	    stopContinuousShotIfRunning();
	    _camera.oneShot();
	    tick();
	    break;
	  */
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
    static int		nframes = 0;
    static timeval	start;
    countTime(nframes, start);

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

    _movieProp[0] = 0;
    _movieProp[1] = _movie.nframes() - 1;
    _movieProp[2] = 1;
    _captureCmd.setProp(c_StatusMovie, _movieProp);
    
    repaintCanvas();
}
    
template <class CAMERA, class PIXEL> void
MyCmdWindow<CAMERA, PIXEL>::repaintCanvas()
{
    _canvas.repaintUnderlay();
    _captureCmd.setValue(c_StatusMovie, int(_movie.currentFrame()));
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
	_camera.stopContinuousShot();
	_captureCmd.setValue(c_ContinuousShot, 0);
    }
}

}
}
