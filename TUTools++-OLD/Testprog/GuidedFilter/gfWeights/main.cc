/*
 *  $Id$
 */
#include <sstream>
#include "TU/v/App.h"
#include "TU/v/CmdWindow.h"
#include "TU/v/CmdPane.h"
#include "TU/v/CanvasPane.h"
#include "TU/v/CanvasPaneDC.h"
#include "TU/GuidedFilter.h"

namespace TU
{
namespace v
{
/************************************************************************
*  static data								*
************************************************************************/
enum	{c_WinSize, c_Regularization, c_Saturation, c_Cursor};

static float	range[][3] = {{1, 64, 1}, {0, 255, 1}, {1, 64, 4}};
static CmdDef	Cmds[] =
{
    {C_Slider, c_WinSize,	 11, "Window size:",	range[0], CA_None,
     0, 0, 1, 1, 0},
    {C_Slider, c_Regularization,  5, "Regularization:",	range[1], CA_None,
     1, 0, 1, 1, 0},
    {C_Slider, c_Saturation,     12, "Saturation:",     range[2], CA_None,
     0, 1, 1, 1, 0},
    {C_Label,  c_Cursor,	  0, "         ",	noProp,	  CA_None,
     1, 1, 1, 1, 0},
    EndOfCmds
};

/************************************************************************
*  class MyCanvasPane<T>						*
************************************************************************/
template <class T>
class MyCanvasPane : public CanvasPane
{
  public:
    MyCanvasPane(Window& parentWin, const Image<T>& image,
		 size_t width, size_t height)
	:CanvasPane(parentWin, width, height),
	 _dc(*this, width, height), _image(image)	{}

    virtual void	repaintUnderlay()		{ _dc << _image; }
    void		clear()				{ _dc.clear(); }
    void		drawPoint(int u, int v)
			{
			    _dc << foreground(BGR(255, 255, 0))
				<< Point2<int>(u, v);
			}
    void		setZoom(float zoom)
			{
			    _dc.setZoom(zoom);
			}
    void		setSize(size_t width, size_t height)
			{
			    _dc.setSize(width, height, _dc.zoom());
			}
    virtual void	callback(CmdId id, CmdVal val)	;
    
  private:
    CanvasPaneDC	_dc;
    const Image<T>&	_image;
};

template <class T> void
MyCanvasPane<T>::callback(CmdId id, CmdVal val)
{
    switch (id)
    {
      case Id_MouseMove:
      case Id_MouseButton1Press:
      case Id_MouseButton1Drag:
      case Id_MouseButton1Release:
      {
	CmdVal	logicalPosition(_dc.dev2logU(val.u()), _dc.dev2logV(val.v()));
	parent().callback(id, logicalPosition);
      }
        return;
    }

    parent().callback(id, val);
}
    
/************************************************************************
*  class MyCmdWindow<T>							*
************************************************************************/
template <class T>
class MyCmdWindow : public CmdWindow
{
  public:
    MyCmdWindow(App& parentApp, const char* name, const Image<T>& guide);

    void		showWeights(size_t u, size_t v)			;
    virtual void	callback(CmdId id, CmdVal val)			;

  private:
    const Image<T>&		_guide;
    Image<float>		_weights;
    GuidedFilter2<float>	_gf2;
    CmdPane			_cmd;
    MyCanvasPane<T>		_canvas;
    MyCanvasPane<float>		_weightsCanvas;
};

template <class T>
MyCmdWindow<T>::MyCmdWindow(App& parentApp, const char* name,
			    const Image<T>& guide)
    :CmdWindow(parentApp, name, 0, Colormap::RGBColor, 16, 0, 0),
     _guide(guide),
     _weights(),
     _gf2(5, 5, 1),
     _cmd(*this, Cmds),
     _canvas(*this, _guide, _guide.width(), _guide.height()),
     _weightsCanvas(*this, _weights, 256, 256)
{
    
    _cmd.place(0, 0, 2, 1);
    _canvas.place(0, 1, 1, 1);
    _weightsCanvas.place(1, 1, 1, 1);

    size_t	w = _cmd.getValue(c_WinSize);
    float	s = _cmd.getValue(c_Regularization).f();
    _gf2.setRowWinSize(w);
    _gf2.setColWinSize(w);
    _gf2.setEpsilon(s*s);
    colormap().setSaturationF(_cmd.getValue(c_Saturation).f());
    _weightsCanvas.setZoom(8);
    _weightsCanvas.setSize(2*w - 1, 2*w - 1);

    show();
}

template <class T> void
MyCmdWindow<T>::showWeights(size_t u, size_t v)
{
    if (u >= _guide.width() || v >= _guide.height())
	return;
    
    const size_t	w = _cmd.getValue(c_WinSize);
#if 0
    const size_t	uc = (u < w - 1 ? w - 1 :
			      u > _guide.width() - w + 1 ?
			      _guide.width() - w + 1 : u);
    const size_t	vc = (v < w - 1 ? w - 1 :
			      v > _guide.height() - w + 1 ?
			      _guide.height() - w + 1 : v);
    Image<float>	in(_guide.width(), _guide.height()),
			out(in.width(), in.height());
    in[v][u] = 255;
    _gf2.convolve(in.begin(), in.end(),
		  _guide.begin(), _guide.end(), out.begin());
    _weights = out(uc - w + 1, vc - w + 1, 2*w - 1, 2*w - 1);
#else
    const size_t	ub = std::max(u,  2*w - 2) - 2*w + 2,
			ue = std::min(u + 2*w - 1, _guide.width()),
			vb = std::max(v,  2*w - 2) - 2*w + 2,
			ve = std::min(v + 2*w - 1, _guide.height());
    Image<float>	in(ue - ub, ve - vb), out(in.width(), in.height());
    const size_t	uc = (u < w - 1 ? w - 1 : u < 2*w - 2 ? u : 2*w - 2),
			vc = (v < w - 1 ? w - 1 : v < 2*w - 2 ? v : 2*w - 2);
    in[vc][uc] = 255;
    const auto		guide = _guide(ub, vb, in.width(), in.height());
    _gf2.convolve(in.begin(), in.end(),
		  guide.begin(), guide.end(), out.begin());
    _weights = out(uc - w + 1, vc - w + 1,
		   out.width() - w + 1, out.height() - w + 1);
#endif
    _weightsCanvas.setSize(2*w - 1, 2*w - 1);
    _weightsCanvas.repaintUnderlay();
}
    
template <class T> void
MyCmdWindow<T>::callback(CmdId id, CmdVal val)
{
    switch (id)
    {
      case M_Exit:
	app().exit();
	break;

      case c_WinSize:
	_gf2.setRowWinSize(val);
	_gf2.setColWinSize(val);
	break;

      case c_Regularization:
	_gf2.setEpsilon(val.f()*val.f());
	break;
	
      case c_Saturation:
	colormap().setSaturationF(val.f());
	_weightsCanvas.repaintUnderlay();
	break;
	
      case Id_MouseButton1Press:
      case Id_MouseButton1Drag:
	showWeights(val.u(), val.v());
      case Id_MouseMove:
      {
	int	w = _cmd.getValue(c_WinSize);
	_weightsCanvas.drawPoint(w - 1, w - 1);
	  
	std::ostringstream	s;
	s << '(' << val.u() << ',' << val.v() << ')';
	_cmd.setString(c_Cursor, s.str().c_str());
      }
        break;

      case Id_MouseButton1Release:
	_weightsCanvas.clear();
	break;
    }
}

}
}

int
main(int argc, char* argv[])
{
    using namespace	std;
    using namespace	TU;

    try
    {
	typedef u_char	pixel_type;
	
	v::App			vapp(argc, argv);
	Image<pixel_type>	guide;
	guide.restore(cin);

      // GUIオブジェクトを作り，イベントループを起動．
	v::MyCmdWindow<pixel_type>	myWin(vapp, "Weights of guided filter",
					      guide);
	vapp.run();
    }
    catch (exception& err)
    {
	cerr << err.what() << endl;
	return 1;
    }
    
    return 0;
}
