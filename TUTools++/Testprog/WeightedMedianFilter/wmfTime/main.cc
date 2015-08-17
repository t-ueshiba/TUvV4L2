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
*  class Exp<S, T>							*
************************************************************************/
template <class S, class T>
class Exp
{
  public:
    typedef S	argument_type;
    typedef T	result_type;

  public:
    Exp(result_type sigma=1)	:_nsigma(-sigma)	{}

    void	setSigma(result_type sigma)		{ _nsigma = -sigma; }
    result_type	operator ()(argument_type x, argument_type y) const
		{
		    return exp(x, y, typename std::is_arithmetic<S>::type());
		}
    
  private:
    result_type	exp(argument_type x, argument_type y, std::true_type) const
		{
		    return std::exp(diff(x, y) / _nsigma);
		}
    result_type	exp(argument_type x, argument_type y, std::false_type) const
		{
		    Vector<T, FixedSizedBuf<T, 3> >	v;
		    v[0] = T(x.r) - T(y.r);
		    v[1] = T(x.g) - T(y.g);
		    v[2] = T(x.b) - T(y.b);
		    return std::exp(v.length() / _nsigma);
		}

  private:
    result_type	_nsigma;
};
    
/************************************************************************
*  global functions							*
************************************************************************/
template <class T, class G> void
doJob(const Image<T>& in, const Image<G>& guide,
      float sigma, size_t winSize, size_t grainSize)
{
    typedef Exp<G, float>	wfunc_type;

    wfunc_type					wfunc(sigma);
    WeightedMedianFilter2<T, wfunc_type, true>	wmf(wfunc, winSize, 256, 16);
    Image<T>					out(in.width(), in.height());
    Profiler<>					profiler(1);

    wmf.setGrainSize(grainSize);
    
    for (size_t n = 0; n < 100; ++n)
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
