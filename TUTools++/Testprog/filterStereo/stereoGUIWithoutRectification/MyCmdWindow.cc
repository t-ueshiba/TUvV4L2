/*
 *  $Id: MyCmdWindow.cc 1454 2013-11-11 11:13:37Z ueshiba $
 */
#include <stdexcept>
#include "TU/GuidedFilter.h"
#include "TU/TreeFilter.h"
#include "TU/WeightedMedianFilter.h"
#include "TU/v/FileSelection.h"
#include "TU/v/Notify.h"
#include "stereo1394.h"
#include "MyCmdWindow.h"
#include "ComputeThreeD.h"

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
		    return diff(x, y);
		}
};

namespace v
{
CmdDef*		createMenuCmds()					;

/************************************************************************
*  class MyCmdWindow<PIXEL, DISP>					*
************************************************************************/
template <class PIXEL, class DISP>
MyCmdWindow<PIXEL, DISP>::MyCmdWindow(App& parentApp)
    :CmdWindow(parentApp, "Stereo vision", Colormap::RGBColor, 256, 0, 0),
   // Stereo stuffs.
     _params(),
     _imageL(),
     _originalImageR(),
     _imageR(),
     _disparityMap(),
   // GUI stuffs.
     _menuCmd(*this, createMenuCmds()),
     _canvasL(*this, 512, 384, _imageL),
     _canvasR(*this, 512, 384, _imageR),
     _canvasD(*this, 512, 384, _disparityMap)
{
    using namespace	std;
    
    _menuCmd.place(0, 0, 1, 1);
    _canvasL.place(0, 1, 1, 1);
    _canvasR.place(1, 1, 1, 1);
    _canvasD.place(1, 0, 1, 1);
    show();

    _menuCmd.setValue(c_DoHorizontalBackMatch,
		      int(_params.doHorizontalBackMatch));
    _menuCmd.setValue(c_DisparitySearchWidth,
		      int(_params.disparitySearchWidth));
    _menuCmd.setValue(c_WindowSize, int(_params.windowSize));
    _menuCmd.setValue(c_Regularization, _params.sigma);
    _menuCmd.setValue(c_DisparityInconsistency,
		      int(_params.disparityInconsistency));
    _menuCmd.setValue(c_IntensityDiffMax, int(_params.intensityDiffMax));
  //_menuCmd.setValue(c_DerivativeDiffMax, int(_params.derivativeDiffMax));
  //_menuCmd.setValue(c_Blend, float(_params.blend));
}

template <class PIXEL, class DISP> void
MyCmdWindow<PIXEL, DISP>::callback(CmdId id, CmdVal val)
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
		if (!_imageL.restore(in) || !_originalImageR.restore(in))
		    throw runtime_error("Two or more images needed!!");
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

	  case c_Algorithm:
	    stereoMatch();
	    break;
	    
	  case c_DoHorizontalBackMatch:
	  {
	    _params.doHorizontalBackMatch = val;
	    stereoMatch();
	  }
	    break;

	  case c_DisparitySearchWidth:
	  {
	    _params.disparitySearchWidth = val;
	    initializeRectification();
	    stereoMatch();
	  }
	    break;
	
	  case c_WindowSize:
	  {
	    _params.windowSize = val;
	    stereoMatch();
	  }
	    break;
	
	  case c_Regularization:
	  {
	    _params.sigma = val.f();
	    stereoMatch();
	  }
	    break;

	  case c_DisparityInconsistency:
	  {
	    _params.disparityInconsistency = val;
	    stereoMatch();
	  }
	    break;

	  case c_IntensityDiffMax:
	  {
	    _params.intensityDiffMax = val;
	    stereoMatch();
	  }
	    break;
	  /*
	  case c_DerivativeDiffMax:
	  {
	    _params.derivativeDiffMax = val;
	    stereoMatch();
	  }
	    break;

	  case c_Blend:
	  {
	    _params.blend = val.f();
	    stereoMatch();
	  }
	    break;
	  */

	  case c_WMF:
	  case c_WMFWindowSize:
	  case c_WMFSigma:
	    stereoMatch();
	    break;
	  
	  case Id_MouseButton1Drag:
	    _canvasL.repaintUnderlay();
	    _canvasR.repaintUnderlay();
	    _canvasD.repaintUnderlay();
	  // 次の case に継続
	  case Id_MouseButton1Press:
	  {
	    _canvasL.drawEpipolarLine(val.v);
	    _canvasL.drawEpipolarLineV(val.u);
	    _canvasR.drawEpipolarLine(val.v);
	    _canvasD.drawEpipolarLine(val.v);
	    _canvasD.drawEpipolarLineV(val.u);
	    ostringstream	s;
	    float		d;
	    if (0 <= val.u && val.u < _disparityMap.width() &&
		0 <= val.v && val.v < _disparityMap.height() &&
		(d = _disparityMap[val.v][val.u]) != 0)
	    {
		s.precision(4);
		s << d;
		int dc = int(_params.disparityMax - d + 0.5);
		_canvasR.drawPoint(val.u + dc, val.v);
	    }
	    _menuCmd.setString(c_Disparity, s.str().c_str());
	  }

	  case Id_MouseMove:
	  {
	    ostringstream	s;
	    s << '(' << val.u << ',' << val.v << ')';
	    _menuCmd.setString(c_Cursor, s.str().c_str());
	    u_prev = val.u;
	    v_prev = val.v;
	  }
	    break;
	
	  case Id_MouseButton1Release:
	    _canvasL.repaintUnderlay();
	    _canvasR.repaintUnderlay();
	    _canvasD.repaintUnderlay();
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

template <class PIXEL, class DISP> void
MyCmdWindow<PIXEL, DISP>::initializeRectification()
{
    using namespace	std;

    _params.disparityMax = _params.disparitySearchWidth;
    
    colormap().setSaturationF(_params.disparityMax);

    _imageR.resize(_originalImageR.height(),
		   _originalImageR.width() + _params.disparitySearchWidth - 1);
    for (size_t v = 0; v < _originalImageR.height(); ++v)
	for (size_t u = 0; u < _originalImageR.width(); ++u)
	    _imageR[v][u + _params.disparitySearchWidth - 1]
		= _originalImageR[v][u];
    
    _disparityMap.resize(_imageL.height(), _imageL.width());
    
    _canvasL.resize(_imageL.width(), _imageL.height());
    _canvasR.resize(_imageR.width(), _imageR.height());
    _canvasD.resize(_disparityMap.width(), _disparityMap.height());

    std::cerr << "--- Stereo matching parameters ---\n";
    _params.put(std::cerr);
}

template <class PIXEL, class DISP> void
MyCmdWindow<PIXEL, DISP>::stereoMatch()
{
    typedef diff_iterator<
		typename ImageLine<pixel_type>::const_iterator,
		score_type>					in_iterator;
    typedef typename std::iterator_traits<in_iterator>::value_type
								ScoreArray;

    size_t	hoffset = 0, voffset = 0;
    int		algo = _menuCmd.getValue(c_Algorithm);
    switch (algo)
    {
      case c_SAD:
	hoffset = voffset = _params.windowSize/2;
	break;
      case c_GuidedFilter:
	if (_params.doHorizontalBackMatch)
	    hoffset = _params.windowSize - 1;
	break;
    }
    
    if (_params.doHorizontalBackMatch)
    {
	typedef matching_iterator<
	    typename ImageLine<disparity_type>::iterator,
	    ScoreArray>						out_iterator;

	stereoMatch(algo,
		    make_row_iterator<out_iterator>(
			hoffset, size_t(0),
			_disparityMap.begin() + voffset,
			_params.disparitySearchWidth,
			_params.disparityMax,
			_params.disparityInconsistency));
    }
    else
    {
	typedef assignment_iterator<
	    MinIdx<ScoreArray>,
	    typename ImageLine<disparity_type>::iterator>	out_iterator;

	stereoMatch(algo,
		    make_row_iterator<out_iterator>(
			hoffset, size_t(0),
			_disparityMap.begin() + voffset,
			MinIdx<ScoreArray>(_params.disparityMax)));
    }

    if (_menuCmd.getValue(c_WMF))
    {
	typedef ExpDiff<pixel_type, float>	wfunc_type;
	
	WeightedMedianFilter2<disparity_type, wfunc_type>
	    wmf(wfunc_type(_menuCmd.getValue(c_WMFSigma).f()));
	wmf.setWinSize(int(_menuCmd.getValue(c_WMFWindowSize)));
	wmf.convolve(_disparityMap.cbegin(), _disparityMap.cend(),
		     _imageL.cbegin(), _imageL.cend(), _disparityMap.begin());
    }

    _canvasL.repaintUnderlay();		// rectifyされた左画像を表示．
    _canvasR.repaintUnderlay();		// rectifyされた右画像を表示．
    _canvasD.repaintUnderlay();		// 計算された視差画像を表示．
}

template <class PIXEL, class DISP> template <class OUT> void
MyCmdWindow<PIXEL, DISP>::stereoMatch(int algo, OUT out)
{
    typedef diff_iterator<
		typename ImageLine<pixel_type>::const_iterator,
		score_type>					in_iterator;
    typedef typename std::iterator_traits<in_iterator>::value_type
								ScoreArray;

    auto	ib = make_row_iterator<in_iterator>(
			 make_fast_zip_iterator(
			     boost::make_tuple(_imageL.cbegin(),
					       _imageR.cbegin())),
			 _params.disparitySearchWidth,
			 _params.intensityDiffMax);
    auto	ie = make_row_iterator<in_iterator>(
			 make_fast_zip_iterator(
			     boost::make_tuple(_imageL.cend(),
					       _imageR.cend())),
			 _params.disparitySearchWidth,
			 _params.intensityDiffMax);

    switch (algo)
    {
      case c_SAD:
      {
	BoxFilter2	filter(_params.windowSize, _params.windowSize);
	  
	filter.convolve(ib, ie, out);
      }
	break;
      
      case c_GuidedFilter:
      {
	GuidedFilter2<ScoreArray>	filter(_params.windowSize,
					       _params.windowSize,
					       _params.sigma);
	
	filter.convolve(ib, ie, _imageL.cbegin(), _imageL.cend(), out);
      }
	break;

      case c_TreeFilter:
      {
	typedef Diff<pixel_type, float>		wfunc_type;

	boost::TreeFilter<ScoreArray, wfunc_type>	filter(wfunc_type(),
							       _params.sigma);
	
	filter.convolve(ib, ie, _imageL.cbegin(), _imageL.cend(), out, true);
      }
	break;

      default:
	break;
    }
}

/************************************************************************
*  instantiations							*
************************************************************************/
template class MyCmdWindow<u_char, float>;
template class MyCmdWindow<RGB,	   float>;
}
}
