/*
 *  $Id: Vision++.h,v 1.1.1.1 2002-07-25 02:14:16 ueshiba Exp $
 */
#ifndef __TUVisionPP_h
#define __TUVisionPP_h

#include "TU/Brep/Brep++.h"
#include "TU/v/CmdPane.h"
#include "TU/v/CanvasPaneDC.h"

namespace TU
{
namespace v
{
/************************************************************************
*  color indices							*
************************************************************************/
const int	Color_BG	= 0;
const int	Color_WHITE	= 1;
const int	Color_RED	= 2;
const int	Color_GREEN	= 3;
const int	Color_BLUE	= 4;
const int	Color_CYAN	= 5;
const int	Color_MAGENDA	= 6;
const int	Color_YELLOW	= 7;

/************************************************************************
*  class BrepCmdPane							*
************************************************************************/
class BrepCanvasPane;
class BrepCmdPane : public CmdPane
{
  public:
    BrepCmdPane(Window& parentWindow, BrepCanvasPane& canvas)		;

    virtual void		callback(CmdId id, CmdVal val)		;

    void			setGeometry(const Brep::Geometry* g)	;
    const Brep::Geometry*	getGeometry()			const	;
    const Brep::HalfEdge*	findHalfEdge(u_int n)		const	;
    
  private:
    const Brep::Geometry*	_g;
    BrepCanvasPane&		_canvas;
};

inline const Brep::Geometry*
BrepCmdPane::getGeometry() const
{
    return _g;
}

/************************************************************************
*  class BrepCanvasPane							*
************************************************************************/
class BrepCanvasPane : public CanvasPane
{
  public:
    enum DrawMode	{Erase, Draw, Highlight1, Highlight2, Highlight3};

    BrepCanvasPane(Window& parentWindow,
		   u_int width, u_int height, Brep::Root* root)		;
    ~BrepCanvasPane()							;
    
    virtual void	callback(CmdId id, CmdVal val)			;

    CanvasPaneDC&	dc()						;
    Brep::Root*		root()					const	;

    void	drawDescendants(const Brep::Loop* l, DrawMode mode)	;
    void	draw(const Brep::Loop* l, DrawMode mode)		;
    void	draw(const Brep::HalfEdge* h, DrawMode mode)		;
    void	draw(const Brep::Geometry* g, DrawMode mode)		;
    void	sync()							;
    
    virtual void	repaintUnderlay(int x, int y, int w, int h)	;
    virtual void	repaintOverlay(int x, int y, int w, int h)	;
    
  private:
    Brep::Geometry*	findGeometry(int u, int v)	const	;
    void		setCmdPane(BrepCmdPane* cmd)		{_cmd = cmd;}

    CanvasPaneDC		_dc;
    Brep::Root*			_root;
    const Brep::HalfEdge*	_h_cache;
    BrepCmdPane*		_cmd;

    friend BrepCmdPane::BrepCmdPane(Window&, BrepCanvasPane&)	;
};

inline CanvasPaneDC&
BrepCanvasPane::dc()
{
    return _dc;
}

inline Brep::Root*
BrepCanvasPane::root() const
{
    return _root;
}

inline void
BrepCanvasPane::sync()
{
    _dc.sync();
}

}
}
#endif	/* !__TUVisionPP_h	*/

