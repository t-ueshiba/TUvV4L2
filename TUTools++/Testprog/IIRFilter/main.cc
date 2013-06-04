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
template <class T, class CONVOLVER> void
doJob(typename CONVOLVER::coeff_type alpha, size_t grainSize)
{
    using namespace	std;
	
    typedef typename CONVOLVER::coeff_type	value_type;
    
    Image<T>	in;
    in.restore(cin);
    
    Image<value_type>	out(in.width(), in.height());
    Profiler		profiler(1);
    CONVOLVER		convolver(alpha);
    convolver.setGrainSize(grainSize);
    
    for (int i = 0; i < 5; ++i)
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

    out.save(cout);
}
 
template <class T, class CONVOLVER> void
doJob1(typename CONVOLVER::coeff_type alpha)
{
    using namespace	std;

    typedef typename CONVOLVER::coeff_type	value_type;
    
    Image<T>	in;
    in.restore(cin);
    
    ImageLine<value_type>	out(in.width());
    Profiler			profiler(1);

    CONVOLVER			convolver(alpha);

    for (int i = 0; i < 5; ++i)
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
    bool		gaussian = false;
    size_t		grainSize = 1;
    extern char*	optarg;
    for (int c; (c = getopt(argc, argv, "a:Gg:")) != -1; )
	switch (c)
	{
	  case 'a':
	    alpha = atof(optarg);
	    break;
	  case 'G':
	    gaussian = true;
	    break;
	  case 'g':
	    grainSize = atoi(optarg);
	    break;
	}

    try
    {
	if (gaussian)
	{
	  /*doJob1<pixel_type, GaussianConvolver< coeff_type> >(alpha);
	    cerr << endl;*/
	    doJob< pixel_type, GaussianConvolver2<coeff_type> >(alpha,
								grainSize);
	}
	else
	{
	  /*doJob1<pixel_type, DericheConvolver< coeff_type> >(alpha);
	    cerr << endl;*/
	    doJob< pixel_type, DericheConvolver2<coeff_type> >(alpha,
							       grainSize);
	}
    }
    catch (exception& err)
    {
	cerr << err.what() << endl;
	return 1;
    }
    
    return 0;
}

