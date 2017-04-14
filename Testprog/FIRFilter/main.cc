/*
 *  $Id: main.cc,v 1.17 2012-07-28 09:10:17 ueshiba Exp $
 */
#include <stdlib.h>
#include "TU/Image++.h"
#include "TU/FIRGaussianConvolver.h"
#include "TU/Profiler.h"

namespace TU
{
/************************************************************************
*  static fucntions							*
************************************************************************/
template <class T, class COEFF> void
doJob(COEFF alpha, size_t grainSize, int niter)
{
    using namespace	std;

    typedef T		pixel_type;
    typedef COEFF	value_type;
    
    Image<pixel_type>		in;
    in.restore(cin);
    
    Image<value_type>		out(in.width(), in.height());
    Profiler<>			profiler(1);
    FIRGaussianConvolver2<>	convolver(alpha);
    convolver.setGrainSize(grainSize);
    
    for (int i = 0; i < 5; ++i)
    {
	for (int j = 0; j < niter; ++j)
	{
	    profiler.start(0);
	    convolver.smooth(in.begin(), in.end(), out.begin());
	  //convolver.diffVV(in.begin(), in.end(), out.begin());
	    profiler.nextFrame();
	}
	cerr << "---------------------------------------------" << endl;
	profiler.print(cerr);
    }

    out.save(cout);
}
 
template <class T, class COEFF> void
doJob1(COEFF alpha, int niter)
{
    using namespace	std;

    typedef T		pixel_type;
    typedef COEFF	value_type;
    
    Image<pixel_type>		in;
    in.restore(cin);
    
    Array<value_type>		out(in.width());
    Profiler<>			profiler(1);
    FIRGaussianConvolver<>	convolver(alpha);

    for (int i = 0; i < 5; ++i)
    {
	for (int j = 0; j < niter; ++j)
	{
	    profiler.start(0);
	    convolver.smooth(in[0].begin(), in[0].end(), out.begin());
	    profiler.nextFrame();
	}
	cerr << "---------------------------------------------" << endl;
	profiler.print(cerr);
    }
}
 
}

/************************************************************************
*  global fucntions							*
************************************************************************/
int
main(int argc, char* argv[])
{
    using namespace	std;
    using namespace	TU;

    typedef u_char	pixel_type;
    typedef float	coeff_type;
    
    float		alpha = 1.0;
    size_t		grainSize = 1;
    int			niter = 100;
    extern char*	optarg;
    for (int c; (c = getopt(argc, argv, "a:g:n:")) != -1; )
	switch (c)
	{
	  case 'a':
	    alpha = atof(optarg);
	    break;
	  case 'g':
	    grainSize = atoi(optarg);
	    break;
	  case 'n':
	    niter = atoi(optarg);
	    break;
	}

    try
    {
      //doJob1<pixel_type>(alpha, niter);
	cerr << endl;
	doJob< pixel_type>(alpha, grainSize, niter);
    }
    catch (exception& err)
    {
	cerr << err.what() << endl;
	return 1;
    }
    
    return 0;
}

