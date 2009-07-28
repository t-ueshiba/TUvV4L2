/*
 *  $Id: MyCmdWindow.h,v 1.1 2009-07-28 00:15:17 ueshiba Exp $
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
typedef YUV422	BinocularPixelType;
typedef RGB	TrinocularPixelType;
    
namespace v
{
/************************************************************************
*  class MyCmdWindow							*
************************************************************************/
class MyCmdWindow : public CmdWindow
{
  public:
    MyCmdWindow(App&				parentApp,
		Ieee1394Camera&			camera,
		Ieee1394Camera::Type		type
#ifdef UseTrigger
		, TriggerGenerator&		trigger
#endif
	       )							;

    virtual void	callback(CmdId, CmdVal)				;
    virtual void	tick()						;
    
  private:
    void	stopContinuousShotIfRunning()				;
    void	separateAndDisplay(const Image<BinocularPixelType>&)	;
    void	separateAndDisplay(const Image<TrinocularPixelType>&)	;
    
    Ieee1394Camera&			_camera;
    const u_int				_nviews;
    bool				_sync;
#ifdef UseTrigger
    TriggerGenerator&			_trigger;
#endif
    Movie<BinocularPixelType>		_movie;
    CmdPane				_menuCmd;
    CmdPane				_captureCmd;
    CmdPane				_featureCmd;
    FileSelection			_fileSelection;
    Image<BinocularPixelType>		_binocularImage;
    Image<PixelType>			_images[3];
    MyCanvasPane			_canvasC;
    MyCanvasPane			_canvasH;
    MyCanvasPane			_canvasV;
    Timer				_timer;
};
 
}
}
