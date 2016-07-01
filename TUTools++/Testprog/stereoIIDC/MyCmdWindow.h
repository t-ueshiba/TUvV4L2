/*
 *  平成14-19年（独）産業技術総合研究所 著作権所有
 *  
 *  創作者：植芝俊夫
 *
 *  本プログラムは（独）産業技術総合研究所の職員である植芝俊夫が創作し，
 *  （独）産業技術総合研究所が著作権を所有する秘密情報です．創作者によ
 *  る許可なしに本プログラムを使用，複製，改変，第三者へ開示する等の著
 *  作権を侵害する行為を禁止します．
 *  
 *  このプログラムによって生じるいかなる損害に対しても，著作権所有者お
 *  よび創作者は責任を負いません。
 *
 *  Copyright 2002-2007.
 *  National Institute of Advanced Industrial Science and Technology (AIST)
 *
 *  Creator: Toshio UESHIBA
 *
 *  [AIST Confidential and all rights reserved.]
 *  This program is confidential. Any using, copying, changing or
 *  giving any information concerning with this program to others
 *  without permission by the creator are strictly prohibited.
 *
 *  [No Warranty.]
 *  The copyright holders or the creator are not responsible for any
 *  damages in the use of this program.
 *  
 *  $Id: MyCmdWindow.h 1495 2014-02-27 15:07:51Z ueshiba $
 */
#include "TU/v/App.h"
#include "TU/v/CmdWindow.h"
#include "TU/v/CmdPane.h"
#include "TU/v/Timer.h"
#include "TU/IIDCCameraArray.h"
#include "TU/Rectify.h"
#include "MyCanvasPane.h"
#if defined(DISPLAY_3D)
#  if defined(DEMO)
#    include "MyOglWindow.h"
#  else
#    include "MyOglCanvasPane.h"
#  endif
#endif

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
    MyCmdWindow(App&				parentApp,
#if defined(DISPLAY_3D)
		const XVisualInfo*		vinfo,
		bool				textureMapping,
		double				parallax,
#endif
		const IIDCCameraArray&	cameras,
		const params_type&		params,
		double				scale)			;

    virtual void	callback(CmdId, CmdVal)				;
    virtual void	tick()						;
    
  private:
    void		syncronizedSnap()				;
    void		restoreCalibration()				;
    void		initializeRectification()			;
    void		stopContinuousShotIfRunning()			;
    void		stereoMatch()					;
#if defined(DISPLAY_2D)
    void		scaleDisparity()				;
#endif
    void		putThreeD(std::ostream& out)		const	;
#if defined(DISPLAY_3D)
    void		putThreeDImage(std::ostream& out)	const	;
#endif

  private:
  // Stereo stuffs.
    const IIDCCameraArray&			_cameras;
    const double				_initialWidth;
    const double				_initialHeight;
    const double				_scale;
    Rectify					_rectify;
    stereo_type					_stereo;
    size_t					_nimages;
    Image<pixel_type>				_images[3];
    Image<pixel_type>				_rectifiedImages[3];
    Image<disparity_type>			_disparityMap;

  // GUI stuffs.
    float					_b;
    CmdPane					_menuCmd;
    CmdPane					_captureCmd;
    CmdPane					_featureCmd;
#if defined(DISPLAY_2D)
    MyCanvasPane<pixel_type>			_canvasL;
#  if !defined(NO_RV)
    MyCanvasPane<pixel_type>			_canvasR;
    MyCanvasPane<pixel_type>			_canvasV;
#  endif
    Image<u_char>				_disparityMapUC;
    MyCanvasPane<u_char>			_canvasD;
#endif
#if defined(DISPLAY_3D)
    const double				_parallax;
#  if defined(DEMO)
    MyOglWindow<pixel_type>			_window3D;
#  else
    MyOglCanvasPane<disparity_type, pixel_type>	_canvas3D;
#  endif
#endif
    Timer					_timer;
};

}
}
