/*
 *  $Id$
 */
#include "TU/Image++.h"
#include "TU/TreeFilter.h"
#include "TU/Profiler.h"

namespace TU
{
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

template <class T, class G, class U> void
doJob(const Image<T>& image, const Image<G>& guide, U sigma, bool normalize)
{
    typedef T					value_type;
    typedef G					guide_type;
    typedef U					weight_type;
    typedef Diff<guide_type, weight_type>	wfunc_type;
    
    boost::TreeFilter<weight_type, wfunc_type, std::chrono::system_clock>
			tf(wfunc_type(), sigma);
    Image<weight_type>	out(image.width(), image.height());
    Profiler<>		profiler(1);
    
    for (size_t n = 0; n < 10; ++n)
    {
	profiler.start(0);
	tf.convolve(image.begin(), image.end(), guide.begin(), guide.end(),
		    out.begin(), normalize);
	profiler.nextFrame();
    }
    tf.print(std::cerr);
    profiler.print(std::cerr);

    out.save(std::cout);
}
    
}

int
main(int argc, char* argv[])
{
    using namespace	TU;
    using		std::cin;
    using		std::cerr;
    using		std::endl;

    typedef u_char	pixel_type;
    typedef u_char	guide_type;
    typedef float	weight_type;
    
    weight_type		sigma = 3;
    bool		normalize = false;
    extern char*	optarg;
    
    for (int c; (c = getopt(argc, argv, "s:n")) != -1; )
	switch (c)
	{
	  case 's':
	    sigma = atof(optarg);
	    break;
	  case 'n':
	    normalize = true;
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

	doJob(image, guide, sigma, normalize);
    }
    catch (std::exception& err)
    {
	cerr << err.what() << endl;
	return 1;
    }
    
    return 0;
}
