/*
 *  $Id$
 */
#include "TU/Image++.h"
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
enum	{c_Sigma, c_Normalization, c_Saturation};

static float	range[][3] = {{1, 255, 2}, {1, 255, 1}};
static CmdDef	Cmds[] =
{
    {C_Slider, c_Sigma,		11, "Sigma:",		range[0], CA_None,
     0, 0, 1, 1, 0},
    {C_ToggleButton, c_Normalization, 1, "Normalize",	0,	  CA_None,
     1, 0, 1, 1, 0},
    {C_Slider, c_Saturation,   256, "Saturation:",      range[1], CA_None,
     2, 0, 1, 1, 0},
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
    
  private:
    const Image<T>&	_image;
    CanvasPaneDC	_dc;
};

/************************************************************************
*  class MyCmdWindow<T>							*
************************************************************************/
template <class T, class G>
class MyCmdWindow : public CmdWindow
{
  public:
    MyCmdWindow(App& parentApp,
		const Image<T>& image, const Image<G>& guide)		;

    void		filter()					;
    virtual void	callback(CmdId id, CmdVal val)			;

  private:
    CmdPane					_cmd;
    const Image<T>&				_image;
    const Image<G>&				_guide;
    Image<float>				_result;
    Diff<G, float>				_wfunc;
    boost::TreeFilter<float, Diff<G, float> >	_tf;
    MyCanvasPane<T>				_imageCanvas;
    MyCanvasPane<float>				_resultCanvas;
};

template <class T, class G>
MyCmdWindow<T, G>::MyCmdWindow(App& parentApp,
			       const Image<T>& image, const Image<G>& guide)
    :CmdWindow(parentApp, "Joint tree filter", Colormap::RGBColor, 16, 0, 0),
     _cmd(*this, Cmds),
     _image(image),
     _guide(guide),
     _result(_image.width(), _image.height()),
     _wfunc(),
     _tf(_wfunc, 0.5),
     _imageCanvas(*this, _image),
     _resultCanvas(*this, _result)
{
    _cmd.place(0, 0, 2, 1);
    _imageCanvas.place(0, 1, 1, 1);
    _resultCanvas.place(1, 1, 1, 1);
    show();

    _tf.setSigma(_cmd.getValue(c_Sigma).f());
    filter();
}

template <class T, class G> void
MyCmdWindow<T, G>::callback(CmdId id, CmdVal val)
{
    using namespace	std;
    
    switch (id)
    {
      case M_Exit:
	app().exit();
	break;

      case c_Sigma:
	_tf.setSigma(val.f());
      case c_Normalization:
	filter();
	break;

      case c_Saturation:
	colormap().setSaturation(val);
	colormap().setSaturationF(val.f());
	_imageCanvas.repaintUnderlay();
	_resultCanvas.repaintUnderlay();
	break;
    }
}

template <class T, class G> void
MyCmdWindow<T, G>::filter()
{
    bool	norm = _cmd.getValue(c_Normalization);
    
    _tf.convolve(_image.cbegin(), _image.cend(),
		 _guide.cbegin(), _guide.cend(), _result.begin(), norm);
    _imageCanvas.repaintUnderlay();
    _resultCanvas.repaintUnderlay();
}
    
}
}

/************************************************************************
*  global functions							*
************************************************************************/
int
main(int argc, char* argv[])
{
    using namespace	std;
    using namespace	TU;

    typedef u_char	pixel_type;
    typedef u_char	guide_type;
    
    try
    {
	v::App			vapp(argc, argv);

	Image<pixel_type>	image;
	image.restore(cin);
	Image<guide_type>	guide;
	if (!guide.restore(cin))
	    guide = image;
	else if (image.width()  != guide.width() ||
		 image.height() != guide.height())
	    throw runtime_error("Mismatched image sizes!");
	
	v::MyCmdWindow<pixel_type, guide_type>	myWin(vapp, image, guide);
	vapp.run();
    }
    catch (exception& err)
    {
	cerr << err.what() << endl;
	return 1;
    }

    return 0;
}
