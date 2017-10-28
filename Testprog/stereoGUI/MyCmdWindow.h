/*
 *  $Id: MyCmdWindow.h 1246 2012-11-30 06:23:09Z ueshiba $
 */
#include "TU/v/App.h"
#include "TU/v/CmdWindow.h"
#include "TU/v/CmdPane.h"
#include "TU/v/Timer.h"
#include "TU/Rectify.h"
#include "MyCanvasPane.h"
#include "MyOglCanvasPane.h"

namespace TU
{
namespace v
{
/************************************************************************
*  class MyCmdWindow<STEREO, PIXEL, DISP>				*
************************************************************************/
template <class STEREO, class PIXEL=u_char, class DISP=float>
class MyCmdWindow : public CmdWindow
{
  public:
    typedef STEREO			stereo_type;
    typedef PIXEL			pixel_type;
    typedef DISP			disparity_type;
    
  private:
    typedef typename STEREO::Parameters	params_type;

  public:
    MyCmdWindow(App&			parentApp,
		const XVisualInfo*	vinfo,
		bool			textureMapping,
		double			parallax,
		const params_type&	params,
		double			scale)				;

    virtual void	callback(CmdId, CmdVal)				;
    
  private:
    void		initializeRectification()			;
    void		stereoMatch()					;
    void		putThreeD(std::ostream& out)		const	;
    void		putThreeDImage(std::ostream& out)	const	;

  // Stereo stuffs.
    const double				_scale;
    Rectify					_rectify;
    stereo_type					_stereo;
    u_int					_nimages;
    Image<pixel_type>				_images[3];
    Image<pixel_type>				_rectifiedImages[3];
    Image<disparity_type>			_disparityMap;

  // GUI stuffs.
    float					_b;
    CmdPane					_menuCmd;
    MyCanvasPane<pixel_type>			_canvasL;
    MyCanvasPane<pixel_type>			_canvasR;
    MyCanvasPane<pixel_type>			_canvasV;
    MyCanvasPane<disparity_type>		_canvasD;
    const double				_parallax;
    MyOglCanvasPane<disparity_type, pixel_type>	_canvas3D;
};
 
}
}
