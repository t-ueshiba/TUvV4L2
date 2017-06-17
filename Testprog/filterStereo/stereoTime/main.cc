/*
 *  $Id: main.cc 1246 2012-11-30 06:23:09Z ueshiba $
 */
#include <cstdlib>
#include "TU/algorithm.h"
#include "TU/io.h"
#include "TU/Rectify.h"
#include "TU/GuidedFilter.h"
#include "TU/TreeFilter.h"
#include "TU/StereoUtility.h"
#include "TU/Profiler.h"

#define DEFAULT_PARAM_FILE	"stereo"
#define DEFAULT_CONFIG_DIRS	".:/usr/local/etc/cameras"
#define DEFAULT_SCALE		1.0
#define DEFAULT_GRAINSIZE	50

//#define SCORE_ARRAY2

namespace TU
{
enum Algorithm	{SAD, GF, WMF, TF};
    
/************************************************************************
*  class Diff<S, T>							*
************************************************************************/
  /*
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
  */

/************************************************************************
*  static functions							*
************************************************************************/
template <class S, class T> static void
doJob(const Image<T>& imageL, const Image<T>& imageR,
      const StereoParameters& params, double scale, Algorithm algo,
      size_t ntrials)
{
    using	disparity_type	= float;
    
  // 画像を平行化する．
    Rectify		rectify;
    Image<T>		rectifiedImageL, rectifiedImageR;
    rectify.initialize(imageL, imageR, scale,
		       params.disparitySearchWidth, params.disparityMax);
    rectify(imageL, imageR, rectifiedImageL, rectifiedImageR);

  // ステレオマッチングを行う．
    Image<disparity_type>	disparityMap(rectify.width(0),
					     rectify.height(0));
    const auto			rowL  = std::cbegin(rectifiedImageL);
    const auto			rowR  = std::cbegin(rectifiedImageR);
    const auto			rowLe = std::cend(rectifiedImageL);
    const auto			rowRe = std::cend(rectifiedImageR);
    const auto			rowD  = std::begin(disparityMap);
#ifdef SCORE_ARRAY2
    Array2<Array<S> >		scores(disparityMap.height(),
				       disparityMap.width());
#endif
    Profiler<>			profiler(2);
    
    switch (algo)
    {
      case SAD:
      {
	BoxFilter2	filter(params.windowSize, params.windowSize);

	for (size_t i = 0; i < ntrials; ++i)
	{
	    for (size_t j = 0; j < 10; ++j)
	    {
		profiler.start(0);
		rectify(imageL, imageR, rectifiedImageL, rectifiedImageR);
		profiler.start(1);
		filter.convolve(
		    make_range_iterator(
			make_diff_iterator<S>(std::cbegin(*rowL),
					      std::cbegin(*rowR),
					      params.disparitySearchWidth,
					      params.intensityDiffMax),
			std::make_tuple(stride(rowL), stride(rowR)),
			std::size(*rowL)),
		    make_range_iterator(
			make_diff_iterator<S>(std::cbegin(*rowLe),
					      std::cbegin(*rowRe),
					      params.disparitySearchWidth,
					      params.intensityDiffMax),
			std::make_tuple(stride(rowLe), stride(rowRe)),
			std::size(*rowLe)),
#ifdef SCORE_ARRAY2
		    scores.begin());
#else
		make_range_iterator(
		    make_matching_iterator<S>(
			std::begin(*(rowD + params.windowSize/2))
			+ params.windowSize/2,
			params.disparitySearchWidth,
			params.disparityMax,
			params.disparityInconsistency,
			params.doHorizontalBackMatch),
		    stride(rowD), std::size(*rowD)));
#endif
	        profiler.nextFrame();
	    }
	    std::cerr << "-------------------------------------------" << std::endl;
	    profiler.print(std::cerr);
	}
      }
	break;

      case GF:
      {
	GuidedFilter2<S>
	    filter(params.windowSize, params.windowSize, params.sigma);
	
	for (size_t i = 0; i < ntrials; ++i)
	{
	    for (size_t j = 0; j < 10; ++j)
	    {
		profiler.start(0);
		rectify(imageL, imageR, rectifiedImageL, rectifiedImageR);
		profiler.start(1);
		filter.convolve(
		    make_range_iterator(
			make_diff_iterator<S>(std::cbegin(*rowL),
					      std::cbegin(*rowR),
					      params.disparitySearchWidth,
					      params.intensityDiffMax),
			std::make_tuple(stride(rowL), stride(rowR)),
			std::size(*rowL)),
		    make_range_iterator(
			make_diff_iterator<S>(std::cbegin(*rowLe),
					      std::cbegin(*rowRe),
					      params.disparitySearchWidth,
					      params.intensityDiffMax),
			std::make_tuple(stride(rowLe), stride(rowRe)),
			std::size(*rowLe)),
		    rowL, rowLe,
#ifdef SCORE_ARRAY2
		    scores.begin());
#else
		make_range_iterator(
		    make_matching_iterator<S>(
			std::begin(*rowD),
			params.disparitySearchWidth,
			params.disparityMax,
			params.disparityInconsistency,
			params.doHorizontalBackMatch),
		    stride(rowD), std::size(*rowD)));
#endif
		profiler.nextFrame();
	    }
	    std::cerr << "-------------------------------------------" << std::endl;
	    profiler.print(std::cerr);
	}
      }
	break;
  /*
      case TF:
      {
	typedef Diff<T, float>	wfunc_type;

	boost::TreeFilter<ScoreArray, wfunc_type>	filter(wfunc_type(),
							       params.sigma);
	
	for (size_t i = 0; i < ntrials; ++i)
	{
	    for (size_t j = 0; j < 10; ++j)
	    {
		profiler.start(0);
		rectify(imageL, imageR, rectifiedImageL, rectifiedImageR);
		profiler.start(1);
		filter.convolve(
		    make_row_iterator<in_iterator>(
			make_fast_zip_iterator(
			    boost::make_tuple(rectifiedImageL.cbegin(),
					      rectifiedImageR.cbegin())),
			params.disparitySearchWidth,
			params.intensityDiffMax),
		    make_row_iterator<in_iterator>(
			make_fast_zip_iterator(
			    boost::make_tuple(rectifiedImageL.cend(),
					      rectifiedImageR.cend())),
			params.disparitySearchWidth,
			params.intensityDiffMax),
		    rectifiedImageL.cbegin(),
		    rectifiedImageL.cend(),
		    make_row_iterator<out_iterator>(
			disparityMap.begin(),
#ifdef NO_BACKMATCH
			MinIdx<ScoreArray>(params.disparityMax)
#else
			params.disparitySearchWidth,
			params.disparityMax,
			params.disparityInconsistency
#endif
			),
		    true);
		profiler.nextFrame();
	    }
	    std::cerr << "-------------------------------------------" << std::endl;
	    profiler.print(std::cerr);
	    filter.print(std::cerr);
	}
      }
	break;
  */      
      default:
	break;
    }

#ifdef SCORE_ARRAY2
    std::copy(scores.cbegin(), scores.cend(),
	      make_range_iterator(make_matching_iterator<S>(
				      std::begin(*rowD),
				      params.disparitySearchWidth,
				      params.disparityMax,
				      params.disparityInconsistency,
				      params.doHorizontalBackMatch),
				  stride(rowD), std::size(*rowD)));
#endif
    disparityMap.save(std::cout);
}

}
/************************************************************************
*  global functions							*
************************************************************************/
int
main(int argc, char* argv[])
{
    using namespace	TU;

    typedef u_char	pixel_type;
    typedef float	score_type;
    
    Algorithm	algo			= SAD;
    std::string	paramFile		= DEFAULT_PARAM_FILE;
    std::string	configDirs		= DEFAULT_CONFIG_DIRS;
    double	scale			= DEFAULT_SCALE;
    size_t	windowSize		= 0;
    size_t	disparitySearchWidth	= 0;
    size_t	disparityMax		= 0;
    size_t	grainSize		= DEFAULT_GRAINSIZE;
    size_t	ntrials			= 5;
    
  // コマンド行の解析．
    extern char*	optarg;
    for (int c; (c = getopt(argc, argv, "GMTp:c:s:w:d:m:g:n:")) != EOF; )
	switch (c)
	{
	  case 'G':
	    algo = GF;
	    break;
	  case 'M':
	    algo = WMF;
	    break;
	  case 'T':
	    algo = TF;
	    break;
	  case 'p':
	    paramFile = optarg;
	    break;
	  case 'c':
	    configDirs = optarg;
	    break;
	  case 's':
	    scale = atof(optarg);
	    break;
	  case 'w':
	    windowSize = atoi(optarg);
	    break;
	  case 'd':
	    disparitySearchWidth = atoi(optarg);
	    break;
	  case 'm':
	    disparityMax = atoi(optarg);
	    break;
	  case 'g':
	    grainSize = atoi(optarg);
	    break;
	  case 'n':
	    ntrials = atoi(optarg);
	    break;
	}
    
  // 本当のお仕事．
    try
    {
	std::ifstream		in;
	openFile(in, paramFile, configDirs, ".params");
	StereoParameters	params;
	params.get(in);
	    
	if (windowSize != 0)
	    params.windowSize = windowSize;
	if (disparityMax != 0)
	    params.disparityMax = disparityMax;
	if (disparitySearchWidth != 0)
	    params.disparitySearchWidth = disparitySearchWidth;
	params.grainSize = grainSize;

      // ステレオマッチングパラメータを表示．
	std::cerr << "--- Stereo matching parameters ---\n";
	params.put(std::cerr);

      // 画像を読み込む．
	Image<pixel_type>	imageL, imageR;
	imageL.restore(std::cin);
	imageR.restore(std::cin);

	doJob<score_type>(imageL, imageR, params, scale, algo, ntrials);
    }
    catch (const std::exception& err)
    {
	std::cerr << err.what() << std::endl;
	return 1;
    }

    return 0;
}
