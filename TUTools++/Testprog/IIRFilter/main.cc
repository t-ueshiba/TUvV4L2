/*
 *  $Id: main.cc,v 1.8 2008-08-07 07:27:05 ueshiba Exp $
 */
#include <stdlib.h>
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
    for (int v = 0; v < out.height(); ++v)
	for (int u = 0; u < out.width(); ++u)
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
    for (int c; (c = getopt(argc, argv, "a:l:h:GL")) != EOF; )
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

    Image<u_char>	in, edge;
    in.restore(cin);

    if (laplacian)
    {
	Image<float>	lap, edgeH, edgeV;
	if (gaussian)
	    GaussianConvolver(alpha).laplacian(in, lap)
				    .diffH(in, edgeH).diffV(in, edgeV);
	else
	    DericheConvolver(alpha).laplacian(in, lap)
				   .diffH(in, edgeH).diffV(in, edgeV);
	Image<float>	str;
      	EdgeDetector(th_low, th_high).strength(edgeH, edgeV, str)
				     .zeroCrossing(lap, str, edge)
				     .hysteresisThresholding(edge);
      //EdgeDetector(th_low, th_high).zeroCrossing(lap, edge);
    }
    else
    {
	Image<float>	edgeH, edgeV;
	if (gaussian)
	    GaussianConvolver(alpha).diffH(in, edgeH).diffV(in, edgeV);
	else
	    DericheConvolver(alpha).diffH(in, edgeH).diffV(in, edgeV);
	Image<float>	str;
	Image<u_char>	dir;
	EdgeDetector(th_low, th_high).strength(edgeH, edgeV, str)
				     .direction4(edgeH, edgeV, dir)
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
    
    return 0;
}
