/*
 *  $Id: MyCmdWindow.h 1246 2012-11-30 06:23:09Z ueshiba $
 */
#include "TU/StereoUtility.h"
#include "TU/v/App.h"
#include "TU/v/CmdWindow.h"
#include "TU/v/CmdPane.h"
#include "MyCanvasPane.h"

namespace TU
{
namespace v
{
/************************************************************************
*  class MyCmdWindow<PIXEL, DISP>					*
************************************************************************/
template <class PIXEL=u_char, class DISP=float>
class MyCmdWindow : public CmdWindow
{
  public:
    typedef PIXEL	pixel_type;
    typedef DISP	disparity_type;
    typedef float	score_type;
    
  public:
    MyCmdWindow(App& parentApp)						;

    virtual void	callback(CmdId, CmdVal)				;
    
  private:
    void		initializeRectification()			;
    void		stereoMatch()					;
    template <class OUT>
    void		stereoMatch(int algo, OUT out)			;
    
  private:
  // Stereo stuffs.
    StereoParameters			_params;
    Image<pixel_type>			_imageL, _originalImageR, _imageR;
    Image<disparity_type>		_disparityMap;

  // GUI stuffs.
    CmdPane				_menuCmd;
    MyCanvasPane<pixel_type>		_canvasL;
    MyCanvasPane<pixel_type>		_canvasR;
    MyCanvasPane<disparity_type>	_canvasD;
};
 
}
}
