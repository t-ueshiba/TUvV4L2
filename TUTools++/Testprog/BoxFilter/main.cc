/*
 *  $Id: main.cc,v 1.17 2012-07-28 09:10:17 ueshiba Exp $
 */
#include <stdlib.h>
#include "TU/simd/Array++.h"
#include "TU/Image++.h"
#include "TU/BoxFilter.h"
#include "TU/Profiler.h"

namespace TU
{
#if defined(SIMD)
template <class T>
using allocator = simd::allocator<T>;
#else
template <class T>
using allocator = std::allocator<T>;
#endif
    
/************************************************************************
*  static fucntions							*
************************************************************************/
template <class S, class T> void
doJob(size_t winSize, size_t grainSize, int niter)
{
    using namespace	std;

    Image<S, allocator<S> >	in;
    in.restore(cin);
    
    Image<T, allocator<T> >	out(in.width(), in.height());
    Profiler<>			profiler(1);
    BoxFilter2<T>		box(winSize, winSize);
    box.setGrainSize(grainSize);
    
    for (int i = 0; i < 5; ++i)
    {
	for (int j = 0; j < niter; ++j)
	{
	    profiler.start(0);
	    box.convolve(in.begin(), in.end(), out.begin());
	    profiler.nextFrame();
	}
	cerr << "---------------------------------------------" << endl;
	profiler.print(cerr);
    }

    out *= T(1)/(T(winSize)*T(winSize));
    out.save(cout);
}
 
template <class S, class T> void
doJob1(size_t winSize, int niter)
{
    using namespace	std;

    Image<S, allocator<S> >	in;
    in.restore(cin);
    
    Array<T, 0, allocator<T> >	out(in.width());
    Profiler<>			profiler(1);
    BoxFilter<T>		box(winSize);

    for (int i = 0; i < 5; ++i)
    {
	for (int j = 0; j < niter; ++j)
	{
	    profiler.start(0);
	    box.convolve(in[0].begin(), in[0].end(), out.begin());
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

    using pixel_type	= short;
    using value_type	= float;
    
    size_t		winSize = 3;
    size_t		grainSize = 100;
    int			niter = 100;
    extern char*	optarg;
    for (int c; (c = getopt(argc, argv, "w:g:n:")) != -1; )
	switch (c)
	{
	  case 'w':
	    winSize = atoi(optarg);
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
      //doJob1<pixel_type, value_type>(winSize, niter);
	cerr << endl;
	doJob<pixel_type, value_type>(winSize, grainSize, niter);
    }
    catch (exception& err)
    {
	cerr << err.what() << endl;
	return 1;
    }
    
    return 0;
}

