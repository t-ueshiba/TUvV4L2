/*
 *  $Id$
 */
#include <sstream>
#include "TU/v/App.h"
#include "TU/v/CmdWindow.h"
#include "TU/v/CmdPane.h"
#include "TU/v/CanvasPane.h"
#include "TU/v/CanvasPaneDC.h"
#include "TU/TreeFilter.h"

namespace TU
{
/************************************************************************
*  class Diff<S, T>							*
************************************************************************/
template <class S, class T>
struct Diff
{
    typedef S	argument_type;
    typedef T	result_type;

    result_type	operator ()(argument_type x, argument_type y) const
		{
		    return std::abs(x - y);
		}
};

namespace v
{
/************************************************************************
*  static data								*
************************************************************************/
enum	{c_Sigma, c_Normalization, c_Saturation, c_Cursor};

static float	range[][3] = {{1, 255, 2}, {1, 256, 4}};
static CmdDef	Cmds[] =
{
    {C_Slider, c_Sigma,		11, "Sigma:",		range[0], CA_None,
     0, 0, 1, 1, 0},
    {C_ToggleButton, c_Normalization, 1, "Normalize",	0,	  CA_None,
     1, 0, 1, 1, 0},
    {C_Slider, c_Saturation,    20, "Saturation:",      range[1], CA_None,
     2, 0, 1, 1, 0},
    {C_Label,  c_Cursor,	 0,  "         ",	noProp,	  CA_None,
     3, 0, 1, 1, 0},
    EndOfCmds
};

/************************************************************************
*  class MyCanvasPane<T>						*
************************************************************************/
template <class T>
class MyCanvasPane : public CanvasPane
{
  public:
    MyCanvasPane(Window& parentWin, const Image<T>& image)
	:CanvasPane(parentWin, image.width(), image.height()),
	 _image(image), _dc(*this)			{}

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
    void		moveDC(size_t u, size_t v)
			{
			    CanvasPane::moveDC(_dc.log2devU(u),
					       _dc.log2devV(v));
			}
    
    virtual void	callback(CmdId id, CmdVal val)	;
    
  private:
    const Image<T>&	_image;
    CanvasPaneDC	_dc;
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
*  class MyCmdWindow<T, G>						*
************************************************************************/
template <class T, class G>
class MyCmdWindow : public CmdWindow
{
  public:
    MyCmdWindow(App& parentApp, const Image<G>& guide)			;

    void		showWeights(size_t u, size_t v)			;
    virtual void	callback(CmdId id, CmdVal val)			;

  private:
    CmdPane					_cmd;
    const Image<G>&				_guide;
    Image<T>					_weights;
    Diff<G, float>				_wfunc;
    boost::TreeFilter<T, Diff<G, float> >	_tf;
    MyCanvasPane<G>				_guideCanvas;
    MyCanvasPane<T>				_weightsCanvas;
};

template <class T, class G>
MyCmdWindow<T, G>::MyCmdWindow(App& parentApp, const Image<G>& guide)
    :CmdWindow(parentApp, "Tree filter", Colormap::RGBColor, 16, 0, 0),
     _cmd(*this, Cmds),
     _guide(guide),
     _weights(_guide.width(), _guide.height()),
     _wfunc(),
     _tf(_wfunc, 0.5),
     _guideCanvas(*this, _guide),
     _weightsCanvas(*this, _weights)
{
    _cmd.place(0, 0, 2, 1);
    _guideCanvas.place(0, 1, 1, 1);
    _weightsCanvas.place(1, 1, 1, 1);
    
    _tf.setSigma(_cmd.getValue(c_Sigma).f());
    colormap().setSaturationF(_cmd.getValue(c_Saturation).f());
    _weightsCanvas.setZoom(4);
    
    show();
}

template <class T, class G> void
MyCmdWindow<T, G>::showWeights(size_t u, size_t v)
{
    bool	norm = _cmd.getValue(c_Normalization);
    Image<T>	in(_guide.width(), _guide.height());
    in[v][u] = 255;

    _tf.convolve(in.cbegin(), in.cend(),
		 _guide.cbegin(), _guide.cend(), _weights.begin(), norm);

    _weightsCanvas.moveDC(u, v);
    _weightsCanvas.repaintUnderlay();
}
    
template <class T, class G> void
MyCmdWindow<T, G>::callback(CmdId id, CmdVal val)
{
    switch (id)
    {
      case M_Exit:
	app().exit();
	break;

      case c_Sigma:
	_tf.setSigma(val.f());
	break;

      case c_Saturation:
	colormap().setSaturationF(val.f());
	_weightsCanvas.repaintUnderlay();
	break;

      case Id_MouseButton1Press:
      case Id_MouseButton1Drag:
	showWeights(val.u(), val.v());
	_weightsCanvas.drawPoint(val.u(), val.v());
      case Id_MouseMove:
      {
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
	typedef float	weight_type;
	typedef u_char	guide_type;
	
	v::App			vapp(argc, argv);
	Image<guide_type>	guide;
	guide.restore(cin);

      // GUIオブジェクトを作り，イベントループを起動．
	v::MyCmdWindow<weight_type, guide_type>	myWin(vapp, guide);
	vapp.run();
    }
    catch (exception& err)
    {
	cerr << err.what() << endl;
	return 1;
    }
    
    return 0;
}
