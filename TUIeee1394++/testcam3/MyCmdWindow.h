/*
 *  $Id: MyCmdWindow.h,v 1.2 2011-01-05 02:05:22 ueshiba Exp $
 */
#include "TU/v/App.h"
#include "TU/v/CmdWindow.h"
#include "TU/v/CmdPane.h"
#include "TU/v/FileSelection.h"
#include "TU/v/Timer.h"
#include "MyCanvasPane.h"
#include "TU/Ieee1394++.h"
#include "TU/TriggerGenerator.h"
#include "TU/Movie.h"

namespace TU
{
namespace v
{
/************************************************************************
*  class MyCmdWindow							*
************************************************************************/
class MyCmdWindow : public CmdWindow
{
  public:
    MyCmdWindow(App&			parentApp,
		Ieee1394Camera&		camera,
		Ieee1394Camera::Type	type
#ifdef UseTrigger
		, TriggerGenerator&	trigger
#endif
	       )							;

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
#ifdef UseTrigger
    TriggerGenerator&		_trigger;
#endif
    Movie<PixelType>		_movie;
    Array<MyCanvasPane*>	_canvases;
    CmdPane			_menuCmd;
    CmdPane			_captureCmd;
    CmdPane			_featureCmd;
    FileSelection		_fileSelection;
    Timer			_timer;
};
 
}
}
