/*
 *  $Id: MyCmdWindow.h,v 1.1 2012-06-19 06:14:31 ueshiba Exp $
 */
#include "TU/v/App.h"
#include "TU/v/CmdWindow.h"
#include "TU/v/CmdPane.h"
#include "TU/v/FileSelection.h"
#include "TU/v/Timer.h"
#include "MyCanvasPane.h"
#include "TU/V4L2++.h"

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
    MyCmdWindow(App& parentApp, V4L2Camera& camera)		;

    virtual void	callback(CmdId, CmdVal)			;
    virtual void	tick()					;
    
  private:
    void		repaintCanvas()				;
    void		setFrame()				;
    void		stopContinuousShotIfRunning()		;
    
    V4L2Camera&		_camera;
    Image<PixelType>	_image;
    MyCanvasPane	_canvas;
    CmdPane		_menuCmd;
    CmdPane		_captureCmd;
    CmdPane		_featureCmd;
    FileSelection	_fileSelection;
    Timer		_timer;
};
 
}
}
