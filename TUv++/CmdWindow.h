/*
 *  平成14-19年（独）産業技術総合研究所 著作権所有
 *  
 *  創作者：植芝俊夫
 *
 *  本プログラムは（独）産業技術総合研究所の職員である植芝俊夫が創作し，
 *  （独）産業技術総合研究所が著作権を所有する秘密情報です．著作権所有
 *  者による許可なしに本プログラムを使用，複製，改変，第三者へ開示する
 *  等の行為を禁止します．
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
 *  without permission by the copyright holder are strictly prohibited.
 *
 *  [No Warranty.]
 *  The copyright holder or the creator are not responsible for any
 *  damages caused by using this program.
 *
 *  $Id: CmdWindow.h,v 1.7 2008-09-10 05:12:03 ueshiba Exp $  
 */
#ifndef __TUvCmdWindow_h
#define __TUvCmdWindow_h

#include "TU/v/TUv++.h"
#include "TU/v/Colormap.h"

namespace TU
{
namespace v
{
/************************************************************************
*  class CmdWindow							*
************************************************************************/
class CmdWindow : public Window
{
  private:
    class Paned : public Object
    {
      public:
	Paned(CmdWindow&)					;
	virtual			~Paned()			;
	
	virtual const Widget&	widget()		const	;

      private:
	const Widget	_widget;			// gridboxWidget
    };

  public:
    CmdWindow(Window&			parentWindow,
	      const char*		myName,
	      Colormap::Mode		mode,
	      u_int			resolution,
	      u_int			underlayCmapDim,
	      u_int			overlayDepth,
	      int			screen=-1,
	      bool			fullScreen=false)	;
    CmdWindow(Window&			parentWindow,
	      const char*		myName,
	      const XVisualInfo*	vinfo,
	      Colormap::Mode		mode,
	      u_int			resolution,
	      u_int			underlayCmapDim,
	      u_int			overlayDepth,
	      bool			fullScreen=false)	;
    virtual			~CmdWindow()			;

    virtual const Widget&	widget()		const	;
    virtual Colormap&		colormap()			;
    virtual void		show()				;

  protected:
    virtual Object&		paned()				;

  private:
    friend void		EVcmdWindow(::Widget widget, XtPointer cmdWindowPtr,
				    XEvent* event, Boolean*);
    
    Colormap		_colormap;
    const Widget	_widget;		// applicationShellWidget
    Paned		_paned;
};

}
}
#endif	// !__TUvCmdWindow_h
