/*
 *  $Id$
 */
#include <cstdlib>
#include "TU/Image++.h"
#include "TU/GuidedFilter.h"
#include "TU/Profiler.h"

namespace TU
{
/************************************************************************
*  global functions							*
************************************************************************/
template <class T, class G> void
doJob(const Image<T>& in, const Image<G>& guide, float epsilon, size_t winSize)
{
    using	std::cin;
    using	std::cout;
    using	std::cerr;

    GuidedFilter2<float>	gf(winSize, winSize, epsilon);
    Image<T>			out(in.width(), in.height());
    Profiler			profiler(1);
    
    for (size_t n = 0; n < 100; ++n)
    {
	profiler.start(0);
	gf.convolve(in.begin(), in.end(),
		    guide.begin(), guide.end(), out.begin());
	profiler.nextFrame();
    }
    profiler.print(cerr);

    out.save(cout);
}
    
}

int
main(int argc, char* argv[])
{
    using namespace	TU;
    using		std::cin;
    using		std::cout;
    using		std::cerr;
    using		std::endl;

    typedef float	pixel_type;
    typedef float	guide_type;

    float		epsilon = 0.5;
    size_t		winSize = 5;
    extern char*	optarg;
    for (int c; (c = getopt(argc, argv, "e:w:") != -1); )
	switch (c)
	{
	  case 'e':
	    epsilon = atof(optarg);
	    break;
	  case 'w':
	    winSize = atoi(optarg);
	    break;
	}

    try
    {
	Image<pixel_type>	image;
	image.restore(cin);
	Image<guide_type>	guide;
	guide.restore(cin);

	doJob(image, guide, epsilon, winSize);
    }
    catch (std::exception& err)
    {
	cerr << err.what() << endl;
	return 1;
    }

    return 0;
}
