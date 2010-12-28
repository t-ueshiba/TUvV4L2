/*
 *  $Id: MyCmdWindow.h,v 1.2 2010-12-28 11:48:41 ueshiba Exp $
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
    MyCmdWindow(App&				parentApp,
		const Array<Ieee1394Camera*>&	cameras,
		bool				sync
#ifdef UseTrigger
		, TriggerGenerator&		trigger
#endif
	       )						;

    virtual void	callback(CmdId, CmdVal)			;
    virtual void	tick()					;
    
  private:
    void		initializeMovie()			;
    void		repaintCanvases()			;
    void		setFrame()				;
    void		stopContinuousShotIfRunning()		;
    void		syncronizedSnap()			;
    
    const Array<Ieee1394Camera*>&	_cameras;
    bool				_sync;
#ifdef UseTrigger
    TriggerGenerator&			_trigger;
#endif
    Movie<PixelType>			_movie;
    Array<MyCanvasPane*>		_canvases;
    CmdPane				_menuCmd;
    CmdPane				_captureCmd;
    CmdPane				_featureCmd;
    FileSelection			_fileSelection;
    Timer				_timer;
};
 
}
}
