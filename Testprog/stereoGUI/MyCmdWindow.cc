/*
 *  $Id: MyCmdWindow.cc 1454 2013-11-11 11:13:37Z ueshiba $
 */
#include <stdexcept>
#include "TU/v/FileSelection.h"
#include "TU/v/Notify.h"
#include "TU/SADStereo.h"
#include "TU/GFStereo.h"
#include "stereoGUI.h"
#include "MyCmdWindow.h"
#include "ComputeThreeD.h"

namespace TU
{
/************************************************************************
*  static functions							*
************************************************************************/
static inline u_int	align16(u_int n)	{return 16*((n-1)/16 + 1);}
    
/************************************************************************
*  struct Epsilon<STEREO>						*
************************************************************************/
template <class STEREO>
struct Epsilon
{
    typedef typename STEREO::Parameters			params_type;

    float	get(const params_type& params)		const	{ return 0; }
    void	set(float val, params_type& params)	const	{}
};
template <class SCORE, class DISP>
struct Epsilon<GFStereo<SCORE, DISP> >
{
    typedef typename GFStereo<SCORE, DISP>::Parameters	params_type;
	
    float	get(const params_type& params) const
		{
		    return params.epsilon;
		}
    void	set(float val, params_type& params) const
		{
		    params.epsilon = val;
		}
};
    
namespace v
{
CmdDef*		createMenuCmds()					;

/************************************************************************
*  class MyCmdWindow<STEREO, PIXEL, DISP>				*
************************************************************************/
template <class STEREO, class PIXEL, class DISP>
MyCmdWindow<STEREO, PIXEL, DISP>::MyCmdWindow(App&		 parentApp,
					      const XVisualInfo* vinfo,
					      bool		 textureMapping,
					      double		 parallax,
					      const params_type& params,
					      double		 scale)
    :CmdWindow(parentApp, "Stereo vision", vinfo,
	       Colormap::RGBColor, 256, 0, 0),
   // Stereo stuffs.
     _scale(scale),
     _rectify(),
     _stereo(params),
     _typeInfo(ImageBase::DEFAULT),
     _nimages(0),
     _disparityMap(),
   // GUI stuffs.
     _b(0.0),
     _menuCmd(*this, createMenuCmds()),
     _canvasL(*this, 320, 240, _rectifiedImages[0]),
     _canvasR(*this, 320, 240, _rectifiedImages[1]),
     _canvasV(*this, 320, 240, _rectifiedImages[2]),
     _canvasD(*this, 320, 240, _disparityMap),
     _parallax(parallax),
     _canvas3D(*this, 640, 480, _disparityMap,
	       (textureMapping ? _images[0] : _rectifiedImages[0]),
	       (textureMapping ? &_rectify.warp(0) : 0))
{
    using namespace	std;
    
    _menuCmd.place(0, 0, 3, 1);
    _canvasL.place(0, 2, 1, 1);
    _canvasR.place(1, 2, 1, 1);
    _canvasV.place(0, 1, 1, 1);
    _canvasD.place(1, 1, 1, 1);
    _canvas3D.place(2, 1, 1, 2);
    show();

    const params_type&	p = _stereo.getParameters();
    _menuCmd.setValue(c_DoHorizontalBackMatch, p.doHorizontalBackMatch);
    _menuCmd.setValue(c_DoVerticalBackMatch, p.doVerticalBackMatch);
    _menuCmd.setValue(c_WindowSize, float(p.windowSize));
    _menuCmd.setValue(c_DisparitySearchWidth, float(p.disparitySearchWidth));
    _menuCmd.setValue(c_DisparityMax, float(p.disparityMax));
    _menuCmd.setValue(c_DisparityInconsistency, float(p.disparityInconsistency));
    _menuCmd.setValue(c_IntensityDiffMax, float(p.intensityDiffMax));
    _menuCmd.setValue(c_DerivativeDiffMax, float(p.derivativeDiffMax));
    _menuCmd.setValue(c_Regularization, Epsilon<stereo_type>().get(p));
    _menuCmd.setValue(c_Blend, float(p.blend));
    
    _canvasL.setZoom(0.5);
    _canvasR.setZoom(0.5);
    _canvasV.setZoom(0.5);
    _canvasD.setZoom(0.5);
    colormap().setSaturationF(p.disparityMax);
}

template <class STEREO, class PIXEL, class DISP> void
MyCmdWindow<STEREO, PIXEL, DISP>::callback(CmdId id, CmdVal val)
{
    using namespace	std;
    
    static int		u_prev, v_prev;

    try
    {
	switch (id)
	{
	  case M_Exit:
	    app().exit();
	    break;

	  case M_Open:
	  {
	    FileSelection	fileSelection(*this);
	    ifstream		in;
	    if (fileSelection.open(in))
	    {
		for (_nimages = 0; _nimages < 3; ++_nimages)
		    if (!_images[_nimages].restore(in))
			break;
		initializeRectification();
		stereoMatch();
	    }
	  }
	    break;

	  case M_Save:
	  {
	    FileSelection	fileSelection(*this);
	    ofstream		out;
	    if (fileSelection.open(out))
		_disparityMap.save(out);
	  }
	    break;
	
	  case c_SaveRectifiedImages:
	  {
	    FileSelection	fileSelection(*this);
	    ofstream		out;
	    if (fileSelection.open(out))
	    {
		u_int	nimages = (_menuCmd.getValue(c_Binocular) ? 2 : 3);
		for (u_int i = 0; i < nimages; ++i)
		    _rectifiedImages[i].save(out);
	    }
	  }
	    break;

	  case c_SaveThreeD:
	  {
	    FileSelection	fileSelection(*this);
	    ofstream		out;
	    if (fileSelection.open(out))
		putThreeD(out);
	  }
	    break;

	  case c_SaveThreeDImage:
	  {
	    FileSelection	fileSelection(*this);
	    ofstream		out;
	    if (fileSelection.open(out))
		putThreeDImage(out);
	  }
	  break;

	  case c_Binocular:
	    initializeRectification();
	    stereoMatch();
	    break;
	
	  case c_DoHorizontalBackMatch:
	  {
	    params_type	params = _stereo.getParameters();
	    params.doHorizontalBackMatch = val;
	    _stereo.setParameters(params);

	    stereoMatch();
	  }
	    break;
	    
	  case c_DoVerticalBackMatch:
	  {
	    params_type	params = _stereo.getParameters();
	    params.doVerticalBackMatch = val;
	    _stereo.setParameters(params);

	    stereoMatch();
	  }
	    break;
	    
	  case c_WindowSize:
	  {
	    params_type	params = _stereo.getParameters();
	    params.windowSize = val;
	    _stereo.setParameters(params);

	    stereoMatch();
	  }
	    break;
	
	  case c_DisparitySearchWidth:
	  {
	    params_type	params = _stereo.getParameters();
	    params.disparitySearchWidth = val;
	    _stereo.setParameters(params);
	    initializeRectification();
	    params = _stereo.getParameters();
	    _menuCmd.setValue(c_DisparitySearchWidth,
			      float(params.disparitySearchWidth));
	    _menuCmd.setValue(c_DisparityMax, float(params.disparityMax));

	    initializeRectification();
	    stereoMatch();
	  }
	    break;
	
	  case c_DisparityMax:
	  {
	    params_type	params = _stereo.getParameters();
	    params.disparityMax = val;
	    _stereo.setParameters(params);
	    initializeRectification();
	    params = _stereo.getParameters();
	    _menuCmd.setValue(c_DisparityMax, float(params.disparityMax));
	    colormap().setSaturationF(params.disparityMax);

	    initializeRectification();
	    stereoMatch();
	  }
	    break;

	  case c_Regularization:
	  {
	    params_type	params = _stereo.getParameters();
	    Epsilon<stereo_type>().set(val.f(), params);
	    _stereo.setParameters(params);

	    stereoMatch();
	  }
	    break;

	  case c_DisparityInconsistency:
	  {
	    params_type	params = _stereo.getParameters();
	    params.disparityInconsistency = val;
	    _stereo.setParameters(params);

	    stereoMatch();
	  }
	    break;

	  case c_IntensityDiffMax:
	  {
	    params_type	params = _stereo.getParameters();
	    params.intensityDiffMax = val;
	    _stereo.setParameters(params);

	    stereoMatch();
	  }
	    break;

	  case c_DerivativeDiffMax:
	  {
	    params_type	params = _stereo.getParameters();
	    params.derivativeDiffMax = val;
	    _stereo.setParameters(params);

	    stereoMatch();
	  }
	    break;

	  case c_Blend:
	  {
	    params_type	params = _stereo.getParameters();
	    params.blend = val.f();
	    _stereo.setParameters(params);

	    stereoMatch();
	  }
	    break;

	  case Id_MouseButton1Drag:
	    _canvasL.repaintUnderlay();
	    _canvasR.repaintUnderlay();
	    _canvasV.repaintUnderlay();
	    _canvasD.repaintUnderlay();
	  // 次の case に継続
	  case Id_MouseButton1Press:
	  {
	    _canvasL.drawEpipolarLine(val.v());
	    _canvasL.drawEpipolarLineV(val.u());
	    _canvasR.drawEpipolarLine(val.v());
	    _canvasV.drawEpipolarLine(val.u());
	    _canvasD.drawEpipolarLine(val.v());
	    _canvasD.drawEpipolarLineV(val.u());
	    ostringstream	s;
	    float		d;
	    if (0 <= val.u() && val.u() < _disparityMap.width() &&
		0 <= val.v() && val.v() < _disparityMap.height() &&
		(d = _disparityMap[val.v()][val.u()]) != 0)
	    {
		s.precision(4);
		s << d;
		s.precision(4);
		s << " (" << _b / d << "m)";
		int	dc = int(_stereo.getParameters().disparityMax - d + 0.5);
		_canvasR.drawPoint(val.u() + dc, val.v());
		_canvasV.drawPoint(_rectifiedImages[0].height() - 1
				   - val.v() + dc,
				   val.u());
		_canvas3D.setCursor(val.u(), val.v(), d);
	    }
	    else
		_canvas3D.setCursor(0, 0, 0.0);
	    _canvas3D.repaintUnderlay();
	    _menuCmd.setString(c_Disparity, s.str().c_str());
	  }

	  case Id_MouseMove:
	  {
	    ostringstream	s;
	    s << '(' << val.u() << ',' << val.v() << ')';
	    _menuCmd.setString(c_Cursor, s.str().c_str());
	    u_prev = val.u();
	    v_prev = val.v();
	  }
	    break;
	
	  case Id_MouseButton1Release:
	    _canvasL.repaintUnderlay();
	    _canvasR.repaintUnderlay();
	    _canvasV.repaintUnderlay();
	    _canvasD.repaintUnderlay();
	    _canvas3D.setCursor(0, 0, 0.0);
	    _canvas3D.repaintUnderlay();
	    break;

	  case c_DrawMode:
	    switch (val)
	    {
	      case c_Texture:
		_canvas3D.setDrawMode(
		    MyOglCanvasPaneBase<disparity_type>::Texture);
		_canvas3D.repaintUnderlay();
		break;
	
	      case c_Polygon:
		_canvas3D.setDrawMode(
		    MyOglCanvasPaneBase<disparity_type>::Polygon);
		_canvas3D.repaintUnderlay();
		break;
	
	      case c_Mesh:
		_canvas3D.setDrawMode(
		    MyOglCanvasPaneBase<disparity_type>::Mesh);
		_canvas3D.repaintUnderlay();
		break;
	    }
	    break;
	
	  case c_GazeDistance:
	    _canvas3D.setDistance(1000.0 * val.f());
	    break;
	}
    }
    catch (exception& err)
    {
	Notify	notify(*this);
	notify << err.what();
	notify.show();
    }
}

template <class STEREO, class PIXEL, class DISP> void
MyCmdWindow<STEREO, PIXEL, DISP>::initializeRectification()
{
    using namespace	std;

    if (_nimages < 2)
	throw runtime_error("Two or more images needed!!");
    if (_nimages == 2)
	_menuCmd.setValue(c_Binocular, 1);
    
    if (_menuCmd.getValue(c_Binocular))
    {
	_rectify.initialize(_images[0], _images[1],
			    _scale,
			    _stereo.getParameters().disparitySearchWidth,
			    _stereo.getParameters().disparityMax);
	_rectifiedImages[2] = 0;
	_canvasV.repaintUnderlay();
    }
    else
    {
	_rectify.initialize(_images[0], _images[1], _images[2],
			    _scale,
			    _stereo.getParameters().disparitySearchWidth,
			    _stereo.getParameters().disparityMax);
	_rectifiedImages[2].P  = _rectify.H(2) * _images[2].P;
	_rectifiedImages[2].P /= length(slice(_rectifiedImages[2].P[2], 0, 3));
	_canvasV.resize(_rectify.width(2), _rectify.height(2));
    }
    _rectifiedImages[0].P  = _rectify.H(0) * _images[0].P;
    _rectifiedImages[0].P /= length(slice(_rectifiedImages[0].P[2], 0, 3));
    _disparityMap.P = _rectifiedImages[0].P;
    _rectifiedImages[1].P  = _rectify.H(1) * _images[1].P;
    _rectifiedImages[1].P /= length(slice(_rectifiedImages[1].P[2], 0, 3));
    _canvasL.resize(_rectify.width(0), _rectify.height(0));
    _canvasR.resize(_rectify.width(1), _rectify.height(1));
    _canvasD.resize(_rectify.width(0), _rectify.height(0));
    _canvas3D.resize(_rectify.width(0), _rectify.height(0));
    _canvas3D.initialize(_rectifiedImages[0].P, _rectifiedImages[1].P,
			 1.0 / _scale);
    _canvas3D.resize(_images[0].width(), _images[0].height());
  //    _canvas3D.setLighting();

    const Matrix34d&	Pr = _rectifiedImages[1].P;
    Vector4d		tR;
    tR[0] = -Pr[0][3];
    tR[1] = -Pr[1][3];
    tR[2] = -Pr[2][3];
    tR[3] = 1.0;
  // the right camera center    
    solve(transpose(slice(Pr, 0, 3, 0, 3)), slice(tR, 0, 3));
    const Matrix34d&	Pl = _rectifiedImages[0].P;
    _b  = (Pl[0]*tR) / length(slice(Pl[2], 0, 3)) / 1000.0;

  // Display depth range of measurement.
    int			range[3];
    range[0] = 100 * _b / _stereo.getParameters().disparityMax;
    range[1] = 100 * _b / _stereo.getParameters().disparityMin() - range[0];
    range[2] = 100;
    _menuCmd.setProp(c_GazeDistance, range);

    std::ostringstream	s;
    s.precision(3);
    s << "Depth range: " << _b / _stereo.getParameters().disparityMax
      << " -- "		 << _b / _stereo.getParameters().disparityMin()
      << "(m)";
    _menuCmd.setString(c_DepthRange, s.str().c_str());

    std::cerr << "--- Stereo matching parameters ---\n";
    _stereo.getParameters().put(std::cerr);
}

template <class STEREO, class PIXEL, class DISP> void
MyCmdWindow<STEREO, PIXEL, DISP>::stereoMatch()
{
    if (_menuCmd.getValue(c_Binocular))
    {
	_rectify(_images[0], _images[1],
		 _rectifiedImages[0], _rectifiedImages[1]);
	_disparityMap.resize(_rectify.height(0), _rectify.width(0));
	_stereo(_rectifiedImages[0].cbegin(), _rectifiedImages[0].cend(),
		_rectifiedImages[1].cbegin(), _disparityMap.begin());
    }
    else
    {
	_rectify(_images[0], _images[1], _images[2],
		 _rectifiedImages[0], _rectifiedImages[1], _rectifiedImages[2]);
	_disparityMap.resize(_rectify.height(0), _rectify.width(0));
	_stereo(_rectifiedImages[0].cbegin(), _rectifiedImages[0].cend(),
		_rectifiedImages[0].cend(),   _rectifiedImages[1].cbegin(),
		_rectifiedImages[2].cbegin(), _disparityMap.begin());
	_canvasV.repaintUnderlay();	// rectifyされた上画像を表示．
    }
    _canvasL.repaintUnderlay();		// rectifyされた左画像を表示．
    _canvasR.repaintUnderlay();		// rectifyされた右画像を表示．
    _canvasD.repaintUnderlay();		// 計算された視差画像を表示．
    _canvas3D.repaintUnderlay();	// 3次元復元結果を表示．
}

template <class STEREO, class PIXEL, class DISP> void
MyCmdWindow<STEREO, PIXEL, DISP>::putThreeD(std::ostream& out) const
{
    using namespace	std;

  // Put header.
    int	npoints = 0;
    for (int v = 0; v < _disparityMap.height(); ++v)
	for (int u = 0; u < _disparityMap.width(); ++u)
	    if (_disparityMap[v][u] != 0.0)
		++npoints;
    const Matrix34d&	P = _rectifiedImages[0].P;
    out << "PX" << endl
	<< "# PinHoleParameterH11: " << P[0][0] << endl
	<< "# PinHoleParameterH12: " << P[0][1] << endl
	<< "# PinHoleParameterH13: " << P[0][2] << endl
	<< "# PinHoleParameterH14: " << P[0][3] << endl
	<< "# PinHoleParameterH21: " << P[1][0] << endl
	<< "# PinHoleParameterH22: " << P[1][1] << endl
	<< "# PinHoleParameterH23: " << P[1][2] << endl
	<< "# PinHoleParameterH24: " << P[1][3] << endl
	<< "# PinHoleParameterH31: " << P[2][0] << endl
	<< "# PinHoleParameterH32: " << P[2][1] << endl
	<< "# PinHoleParameterH33: " << P[2][2] << endl
	<< "# PinHoleParameterH34: " << P[2][3] << endl
	<< _disparityMap.width() << ' ' << _disparityMap.height()
	<< '\n' << npoints << endl;

  // Put the 2D image coordinates, pixel intensity and the 3D coordinates.
    ComputeThreeD	toThreeD(_rectifiedImages[0].P, _rectifiedImages[1].P);
    for (int v = 0; v < _disparityMap.height(); ++v)
	for (int u = 0; u < _disparityMap.width(); ++u)
	    if (_disparityMap[v][u] != 0.0)
	    {
		const Point3d&	x = toThreeD(u, v, _disparityMap[v][u]);
		out << u << ' ' << v
		    << '\t' << int(_rectifiedImages[0][v][u])
		    << '\t' << x;
	    }
}

template <class STEREO, class PIXEL, class DISP> void
MyCmdWindow<STEREO, PIXEL, DISP>::putThreeDImage(std::ostream& out) const
{
    const Image<RGB>&	threeDImage = _canvas3D.template getImage<RGB>();
    
    threeDImage.save(out);
}

/************************************************************************
*  instantiations							*
************************************************************************/
  //template class MyCmdWindow<SADStereo<short, u_char> >;
  //template class MyCmdWindow<SADStereo<int,   u_short> >;
template class MyCmdWindow<SADStereo<float, u_char> >;
  //template class MyCmdWindow<SADStereo<float, u_short> >;
template class MyCmdWindow<GFStereo<float,  u_char> >;
  //template class MyCmdWindow<GFStereo<float,  u_short> >;

  //template class MyCmdWindow<SADStereo<short, u_char>, u_char, u_char>;
  //template class MyCmdWindow<GFStereo<float,  u_char>, u_char, u_char>;
}
}
