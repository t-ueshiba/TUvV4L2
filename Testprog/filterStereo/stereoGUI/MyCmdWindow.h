/*
 *  $Id: MyCmdWindow.h 1246 2012-11-30 06:23:09Z ueshiba $
 */
#include "TU/StereoUtility.h"
#include "TU/v/App.h"
#include "TU/v/CmdWindow.h"
#include "TU/v/CmdPane.h"
#include "TU/Rectify.h"
#include "MyCanvasPane.h"
#include "MyOglCanvasPane.h"

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
    template <class OUT>
    void		stereoMatch(int algo, OUT out)			;
    void		stereoMatch()					;
    void		refineDisparity()				;
    
  private:
  // Stereo stuffs.
    StereoParameters				_params;
    Rectify					_rectify;
    Image<pixel_type>				_imageL;
    Image<pixel_type>				_imageR;
    Image<pixel_type>				_rectifiedImageL;
    Image<pixel_type>				_rectifiedImageR;
    Image<disparity_type>			_disparityMap;

  // GUI stuffs.
    float					_b;
    CmdPane					_menuCmd;
    MyCanvasPane<pixel_type>			_canvasL;
    MyCanvasPane<pixel_type>			_canvasR;
    MyCanvasPane<disparity_type>		_canvasD;
    MyOglCanvasPane<disparity_type, pixel_type>	_canvas3D;
};
 
}
}
