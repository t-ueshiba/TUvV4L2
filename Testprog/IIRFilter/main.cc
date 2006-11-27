/*
 *  $Id: main.cc,v 1.4 2006-11-27 00:26:03 ueshiba Exp $
 */
#include <unistd.h>
#include "TU/Image++.h"

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
	Image<float>	lap;
	if (gaussian)
	    GaussianConvolver(alpha).laplacian(in, lap);
	else
	    DericheConvolver(alpha).laplacian(in, lap);
	EdgeDetector(th_low, th_high).zeroCrossing(lap, edge);
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
				     .suppressNonmaxima(str, dir, edge);
    }
    
    Image<RGB>		out;
    superImpose(in, edge, out);
    out.save(cout, ImageBase::RGB_24);
    
  //convolver.laplacian(in, out);
  //convolver.smooth(in, out);
  //edgeH.save(cout, ImageBase::FLOAT);
  //edgeV.save(cout, ImageBase::FLOAT);
  //out.save(cout, ImageBase::U_CHAR);
    
  /*    BilateralIIRFilter<2u>	bi;
    const float	e  = expf(-alpha), beta = sinhf(alpha);
    float	c0[4], c1[4], c2[4];
    c0[0] =  (alpha - 1.0) * e;			// i(n-1)
    c0[1] =  1.0;				// i(n)
    c0[2] = -e * e;				// oF(n-2)
    c0[3] =  2.0 * e;				// oF(n-1)

    c1[0] = -1.0;				// i(n-1)
    c1[1] =  0.0;				// i(n)
    c1[2] = -e * e;				// oF(n-2)
    c1[3] =  2.0 * e;				// oF(n-1)

    c2[0] =  (1.0 + beta) * e;			// i(n-1)
    c2[1] = -1.0;				// i(n)
    c2[2] = -e * e;				// oF(n-2)
    c2[3] =  2.0 * e;				// oF(n-1)
    bi.initialize(c0, BilateralIIRFilter<2u>::Zeroth).convolve(in[64]);
    for (int i = 0; i < bi.dim(); ++i)
	cerr << ' ' << bi[i];
	cerr << endl;*/
    
    return 0;
}

#if defined(__GNUG__) || defined(__INTEL_COMPILER)
#  include "TU/Array++.cc"
#  include "TU/Image++.cc"
#endif
