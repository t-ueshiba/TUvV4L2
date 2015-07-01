/*
 *  $Id: main.cc,v 1.17 2012-07-28 09:10:17 ueshiba Exp $
 */
#include <stdlib.h>
#include "TU/Image++.h"
#include "TU/BoxFilter.h"
#include "TU/Profiler.h"

namespace TU
{
/************************************************************************
*  static fucntions							*
************************************************************************/
template <class T> void
doJob(size_t winSize, size_t grainSize, int niter)
{
    using namespace	std;

    typedef T		pixel_type;
    typedef float	value_type;
    
    Image<pixel_type>	in;
    in.restore(cin);
    
    Image<value_type>	out(in.width(), in.height());
    Profiler		profiler(1);
    BoxFilter2		box(winSize, winSize);
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

    out *= value_type(1)/(value_type(winSize)*value_type(winSize));
    out.save(cout);
}
 
template <class T> void
doJob1(size_t winSize, int niter)
{
    using namespace	std;

    typedef T		pixel_type;
    typedef float	value_type;
    
    Image<pixel_type>		in;
    in.restore(cin);
    
    ImageLine<value_type>	out(in.width());
    Profiler			profiler(1);
    BoxFilter			box(winSize);

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

    typedef int		pixel_type;
    
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
      //doJob1<pixel_type>(winSize, niter);
	cerr << endl;
	doJob<pixel_type>(winSize, grainSize, niter);
    }
    catch (exception& err)
    {
	cerr << err.what() << endl;
	return 1;
    }
    
    return 0;
}

