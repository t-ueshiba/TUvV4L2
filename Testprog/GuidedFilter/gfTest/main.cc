/*
 *  $Id: main.cc,v 1.7 2012-06-19 08:38:46 ueshiba Exp $
 */
#include "TU/v/App.h"
#include "TU/v/CmdWindow.h"
#include "TU/v/CmdPane.h"
#include "TU/v/CanvasPane.h"
#include "TU/v/CanvasPaneDC.h"
#include "TU/v/Timer.h"
#include "TU/GuidedFilter.h"

namespace TU
{
namespace v
{
/************************************************************************
*  static data								*
************************************************************************/
enum	{c_WinSize, c_Regularization, c_Saturation};

static float	range[][3] = {{1, 64, 1}, {0, 32, 0.25}, {1, 255, 1}};
static CmdDef	Cmds[] =
{
    {C_Slider, c_WinSize,	 11, "Window size:",	range[0], CA_None,
     0, 0, 1, 1, 0},
    {C_Slider, c_Regularization, 32, "Regularization:",	range[1], CA_None,
     1, 0, 1, 1, 0},
    {C_Slider, c_Saturation,    255, "Saturation:",     range[2], CA_None,
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
    MyCmdWindow(App& parentApp, const Image<T>& image,
		const Image<G>& guide, bool movie)			;

    void		filter()					;
    virtual void	callback(CmdId id, CmdVal val)			;
    void		tick()						;

  private:
    CmdPane			_cmd;
    Image<T>			_image;
    const Image<G>&		_guide;
    Image<float>		_filteredImage;
    MyCanvasPane<T>		_imageCanvas;
    MyCanvasPane<float>		_resultCanvas;
    Timer			_timer;
};

template <class T, class G>
MyCmdWindow<T, G>::MyCmdWindow(App& parentApp, const Image<T>& image,
			    const Image<G>& guide, bool movie)
    :CmdWindow(parentApp, "Guided filter", Colormap::RGBColor, 16, 0, 0),
     _cmd(*this, Cmds),
     _image(image),
     _guide(movie ? _image : guide),
     _filteredImage(_image.width(), _image.height()),
     _imageCanvas(*this,  _image),
     _resultCanvas(*this, _filteredImage),
     _timer(*this, 0)
{
    _cmd.place(0, 0, 2, 1);
    _imageCanvas.place(0, 1, 1, 1);
    _resultCanvas.place(1, 1, 1, 1);
    
    show();

    if (movie)
	_timer.start(5);
    else
	filter();
}

template <class T, class G> void
MyCmdWindow<T, G>::callback(CmdId id, CmdVal val)
{
    switch (id)
    {
      case M_Exit:
	app().exit();
	break;

      case c_WinSize:
      case c_Regularization:
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
    _filteredImage = 0;

    size_t	w = _cmd.getValue(c_WinSize);
    float	s = _cmd.getValue(c_Regularization).f();
    GuidedFilter2<float>	gf2(w, w, s*s);
    gf2.convolve(_image.begin(), _image.end(), _guide.begin(), _guide.end(),
		 _filteredImage.begin(), true);

    _resultCanvas.repaintUnderlay();
}

template <class T, class G> void
MyCmdWindow<T, G>::tick()
{
    _image.restoreData(std::cin, ImageFormat::U_CHAR);
    filter();
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
	v::App	vapp(argc, argv);

	char	c;
	if (!cin.get(c))
	    throw runtime_error("Failed to read from stdin!!");

      // 画像／ムービーを判定して，画像データまたはヘッダを読み込む．
	Image<pixel_type>	image;
	Image<guide_type>	guide;
	switch (c)
	{
	  case 'P':	// 画像
	  case 'B':
	    cin.putback(c);
	    image.restore(cin);
	    if (!guide.restore(cin))
		guide = image;
	    else if (image.width()  != guide.width() ||
		     image.height() != guide.height())
		throw runtime_error("Mismatched image sizes!");
	    break;
	  case 'M':	// ムービー
	    cin >> skipl;
	    image.restoreHeader(cin);
	    break;
	  default:
	    throw runtime_error("Neither image nor movie file!!");
	    break;
	}

      // GUIオブジェクトを作り，イベントループを起動．
	v::MyCmdWindow<pixel_type, guide_type>
	    myWin(vapp, image, guide, (c == 'M'));
	vapp.run();
    }
    catch (exception& err)
    {
	cerr << err.what() << endl;
	return 1;
    }
    
    return 0;
}
