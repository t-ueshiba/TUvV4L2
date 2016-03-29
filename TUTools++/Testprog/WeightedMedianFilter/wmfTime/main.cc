/*
 *  $Id$
 */
#include <cstdlib>
#include "TU/Image++.h"
#include "TU/WeightedMedianFilter.h"
#include "TU/Profiler.h"

namespace TU
{
/************************************************************************
*  static functions							*
************************************************************************/
template <class T, class G> static void
doJob(const Image<T>& in, const Image<G>& guide,
      float sigma, size_t winSize, size_t grainSize)
{
    typedef ExpDiff<G, float>	wfunc_type;

    wfunc_type	wfunc(sigma);
    WeightedMedianFilter2<T, wfunc_type, std::chrono::system_clock>
		wmf(wfunc, winSize, 256, 256);
    Image<T>	out(in.width(), in.height());
    Profiler<>	profiler(1);

    wmf.setGrainSize(grainSize);
    
    for (size_t n = 0; n < 10; ++n)
    {
	profiler.start(0);
	wmf.convolve(in.begin(), in.end(),
		     guide.begin(), guide.end(), out.begin());
	profiler.nextFrame();
    }
    wmf.print(std::cerr);
    profiler.print(std::cerr);
    
    out.save(std::cout);
}
    
}

/************************************************************************
*  global functions							*
************************************************************************/
int
main(int argc, char* argv[])
{
    using namespace	TU;
    using		std::cin;
    using		std::cout;
    using		std::cerr;
    using		std::endl;

    typedef float	pixel_type;
    typedef RGB		guide_type;

    float		sigma = 5.5;
    size_t		winSize = 5;
    size_t		grainSize = 100;
    extern char*	optarg;
    for (int c; (c = getopt(argc, argv, "s:w:g:")) != -1; )
	switch (c)
	{
	  case 's':
	    sigma = atof(optarg);
	    break;
	  case 'w':
	    winSize = atoi(optarg);
	    break;
	  case 'g':
	    grainSize = atoi(optarg);
	    break;
	}

    try
    {
	Image<pixel_type>	image;
	image.restore(cin);
	Image<guide_type>	guide;
	if (!guide.restore(cin))
	    guide = image;
	else if (image.width()  != guide.width() ||
		 image.height() != guide.height())
	    throw std::runtime_error("Mismatched image sizes!");

	doJob(image, guide, sigma, winSize, grainSize);
    }
    catch (std::exception& err)
    {
	cerr << err.what() << endl;
	return 1;
    }

    return 0;
}
