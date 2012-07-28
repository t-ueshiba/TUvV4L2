/*
 *  $Id: main.cc,v 1.17 2012-07-28 09:10:17 ueshiba Exp $
 */
#include <stdlib.h>
#include "TU/Image++.h"
#include "TU/DericheConvolver.h"
#include "TU/GaussianConvolver.h"
#include "TU/Profiler.h"

namespace TU
{
/************************************************************************
*  static fucntions							*
************************************************************************/
template <class S, class T> void
doJob(float alpha, bool gaussian)
{
    using namespace	std;
	
    Image<S>	in;
    in.restore(cin);
    
    Image<T>	out(in.width(), in.height());
    Profiler	profiler(1);

    if (gaussian)
    {
	GaussianConvolver2<T>	convolver(alpha);

	for (int i = 0; i < 10; ++i)
	{
	    for (int j = 0; j < 100; ++j)
	    {
		profiler.start(0);
		convolver.smooth(in.begin(), in.end(), out.begin());
		profiler.stop().nextFrame();
	    }
	    cerr << "---------------------------------------------" << endl;
	    profiler.print(cerr);
	}
    }
    else
    {
	DericheConvolver2<T>	convolver(alpha);

	for (int i = 0; i < 10; ++i)
	{
	    for (int j = 0; j < 100; ++j)
	    {
		profiler.start(0);
		convolver.smooth(in.begin(), in.end(), out.begin());
		profiler.stop().nextFrame();
	    }
	    cerr << "---------------------------------------------" << endl;
	    profiler.print(cerr);
	}
    }
    out.save(cout, ImageBase::FLOAT);
}
 
template <class S, class T> void
doJob1(float alpha, bool gaussian)
{
    using namespace	std;
	
    Image<S>	in;
    in.restore(cin);
    
    ImageLine<T>	out(in.width());
    Profiler		profiler(1);

    if (gaussian)
    {
	GaussianConvolver<T>	convolver(alpha);

	for (int i = 0; i < 10; ++i)
	{
	    for (int j = 0; j < 100; ++j)
	    {
		profiler.start(0);
		convolver.smooth(in[0].begin(), in[0].end(), out.begin());
		profiler.stop().nextFrame();
	    }
	    cerr << "---------------------------------------------" << endl;
	    profiler.print(cerr);
	}
    }
    else
    {
	DericheConvolver<T>	convolver(alpha);

	for (int i = 0; i < 10; ++i)
	{
	    for (int j = 0; j < 100; ++j)
	    {
		profiler.start(0);
		convolver.smooth(in[0].begin(), in[0].end(), out.begin());
		profiler.stop().nextFrame();
	    }
	    cerr << "---------------------------------------------" << endl;
	    profiler.print(cerr);
	}
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

    float		alpha = 1.0;
    bool		gaussian = false;
    extern char*	optarg;
    for (int c; (c = getopt(argc, argv, "a:G")) != -1; )
	switch (c)
	{
	  case 'a':
	    alpha = atof(optarg);
	    break;
	  case 'G':
	    gaussian = true;
	    break;
	}

    try
    {
	doJob1<u_char, float>(alpha, gaussian);
	cerr << endl;
	doJob<u_char, float>(alpha, gaussian);
    }
    catch (exception& err)
    {
	cerr << err.what() << endl;
	return 1;
    }
    
    return 0;
}

