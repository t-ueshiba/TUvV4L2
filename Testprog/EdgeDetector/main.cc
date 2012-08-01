/*
 *  $Id: main.cc,v 1.3 2012-08-01 20:48:32 ueshiba Exp $
 */
#include <stdlib.h>
#ifdef WIN32
#  include <io.h>
#  include <fcntl.h>
#endif
#include "TU/Image++.h"
#include "TU/GaussianConvolver.h"
#include "TU/DericheConvolver.h"
#include "TU/EdgeDetector.h"

namespace TU
{
static void
superImpose(const Image<u_char>& in, const Image<u_char>& edge,
	    Image<RGB>& out)
{
    out.resize(in.height(), in.width());
    for (u_int v = 0; v < out.height(); ++v)
	for (u_int u = 0; u < out.width(); ++u)
	    if (edge[v][u])
		out[v][u] = RGB(255, 255, 0);
	    else
		out[v][u] = in[v][u];
}

}
    
int
main(int argc, char* argv[])
{
    using namespace	std;
    using namespace	TU;

    float		alpha = 1.0, th_low = 2.0, th_high = 5.0;
    bool		gaussian = false, laplacian = false;
    extern char*	optarg;
    for (int c; (c = getopt(argc, argv, "a:l:h:GL")) != -1; )
	switch (c)
	{
	  case 'a':
	    alpha = atof(optarg);
	    break;
	  case 'l':
	    th_low = atof(optarg);
	    break;
	  case 'h':
	    th_high = atof(optarg);
	    break;
	  case 'G':
	    gaussian = true;
	    break;
	  case 'L':
	    laplacian = true;
	    break;
	}

    try
    {
#ifdef WIN32
	if (_setmode(_fileno(stdin), _O_BINARY) == -1)
	    throw runtime_error("Cannot set stdin to binary mode!!"); 
	if (_setmode(_fileno(stdout), _O_BINARY) == -1)
	    throw runtime_error("Cannot set stdout to binary mode!!"); 
#endif
	Image<u_char>	in, edge;
	in.restore(cin);

	if (laplacian)
	{
	    Image<float>	edgeH(in.width(), in.height()),
				edgeV(in.width(), in.height()),
				lap  (in.width(), in.height());

	    if (gaussian)
	    {
		Image<float>	edgeHH(in.width(), in.height()),
				edgeVV(in.width(), in.height());
		GaussianConvolver2<float>	convolver(alpha);
		convolver.diffHH(in.begin(), in.end(), lap.begin());
		convolver.diffVV(in.begin(), in.end(), edgeVV.begin());
		convolver.diffH (in.begin(), in.end(), edgeH.begin());
		convolver.diffV (in.begin(), in.end(), edgeV.begin());
		lap += edgeVV;
	    }
	    else
	    {
		Image<float>	edgeHH(in.width(), in.height()),
				edgeVV(in.width(), in.height());
		DericheConvolver2<float>	convolver(alpha);
		convolver.diffHH(in.begin(), in.end(), lap.begin());
		convolver.diffVV(in.begin(), in.end(), edgeVV.begin());
		convolver.diffH (in.begin(), in.end(), edgeH.begin());
		convolver.diffV (in.begin(), in.end(), edgeV.begin());
		lap += edgeVV;
	    }
	    Image<float>	str(in.width(), in.height());
	    EdgeDetector(th_low, th_high).strength(edgeH, edgeV, str)
					 .zeroCrossing(lap, str, edge)
					 .hysteresisThresholding(edge);
	  //EdgeDetector(th_low, th_high).zeroCrossing(lap, edge);
	}
	else
	{
	    Image<float>	edgeH(in.width(), in.height()),
				edgeV(in.width(), in.height());
	    if (gaussian)
	    {
		GaussianConvolver2<float>	convolver(alpha);
		convolver.diffH(in.begin(), in.end(), edgeH.begin());
		convolver.diffV(in.begin(), in.end(), edgeV.begin());
	    }
	    else
	    {
		DericheConvolver2<float>	convolver(alpha);
		convolver.diffH(in.begin(), in.end(), edgeH.begin());
		convolver.diffV(in.begin(), in.end(), edgeV.begin());
	    }
	    Image<float>    str;
	    Image<u_char>   dir;
	    EdgeDetector(th_low, th_high).strength(edgeH, edgeV, str)
					 .direction4(edgeH, edgeV, dir)
					 .suppressNonmaxima(str, dir, edge)
					 .suppressNonmaxima(str, dir, edge)
					 .hysteresisThresholding(edge);
	}

	Image<RGB>		out;
	superImpose(in, edge, out);
	out.save(cout, ImageBase::RGB_24);
    
      //convolver.laplacian(in, out);
      //convolver.smooth(in, out);
      //edgeH.save(cout, ImageBase::FLOAT);
      //edgeV.save(cout, ImageBase::FLOAT);
      //out.save(cout, ImageBase::U_CHAR);
    }
    catch (exception& err)
    {
	cerr << err.what() << endl;
	return 1;
    }
    
    return 0;
}
