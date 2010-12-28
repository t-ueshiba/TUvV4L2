/*
 *  $Id: MyCmdWindow.h,v 1.2 2010-12-28 11:47:48 ueshiba Exp $
 */
#include "TU/v/App.h"
#include "TU/v/CmdWindow.h"
#include "TU/v/CmdPane.h"
#include "TU/v/FileSelection.h"
#include "TU/v/Timer.h"
#include "MyCanvasPane.h"
#include "TU/Ieee1394++.h"
#include "TU/Serial.h"
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
    MyCmdWindow(App& parentApp, Ieee1394Camera& camera
#ifdef UseTrigger
		, TriggerGenerator& trigger
#endif
	       )						;

    virtual void	callback(CmdId, CmdVal)			;
    virtual void	tick()					;
    
  private:
    void		initializeMovie()			;
    void		repaintCanvas()				;
    void		setFrame()				;
    void		stopContinuousShotIfRunning()		;
    
    Ieee1394Camera&	_camera;
#ifdef UseTrigger
    TriggerGenerator&	_trigger;
#endif
    Movie<PixelType>	_movie;
    MyCanvasPane	_canvas;
    CmdPane		_menuCmd;
    CmdPane		_captureCmd;
    CmdPane		_featureCmd;
    FileSelection	_fileSelection;
    Timer		_timer;
};
 
}
}
