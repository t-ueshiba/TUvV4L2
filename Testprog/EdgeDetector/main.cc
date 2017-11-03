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
template <class T> static void
superImpose(const Image<T>& in, const Image<u_char>& edge,
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

template <class CONVOLVER, class T> static void
computeEdgeHV(CONVOLVER& convolver,
	      const Image<T>& in, Image<float>& edgeH, Image<float>& edgeV)
{
    edgeH.resize(in.height(), in.width());
    convolver.diffH(in.begin(), in.end(), edgeH.begin());
    edgeV.resize(in.height(), in.width());
    convolver.diffV(in.begin(), in.end(), edgeV.begin());
}
    
template <class CONVOLVER, class T> static void
computeLaplacian(CONVOLVER& convolver,
		 const Image<T>& in, Image<float>& lap)
{
    lap.resize(in.height(), in.width());
    convolver.diffHH(in.begin(), in.end(), lap.begin());
    Image<float>	edgeVV(in.width(), in.height());
    convolver.diffVV(in.begin(), in.end(), edgeVV.begin());
    lap += edgeVV;
}

template <class CONVOLVER, class T> static void
computeEdge(const Image<T>& in, Image<u_char>& edge,
	    bool laplacian, float alpha, float th_low, float th_high)
{
    CONVOLVER		convolver(alpha);
    Image<float>	edgeH, edgeV;
    computeEdgeHV(convolver, in, edgeH, edgeV);

    EdgeDetector	edgeDetector(th_low, th_high);
    Image<float>	str;
    if (laplacian)
    {
	Image<float>	lap;
	computeLaplacian(convolver, in, lap);
	edgeDetector.strength(edgeH, edgeV, str)
		    .zeroCrossing(lap, str, edge)
		    .hysteresisThresholding(edge);
    }
    else
    {
	Image<u_char>	dir;
	edgeDetector.strength(edgeH, edgeV, str)
		    .direction4(edgeH, edgeV, dir)
		    .suppressNonmaxima(str, dir, edge)
		    .hysteresisThresholding(edge);
    }
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
	Image<u_char>	in;
	in.restore(cin);

	Image<u_char>	edge;
	if (gaussian)
	    computeEdge<GaussianConvolver2<float> >(in, edge, laplacian,
						    alpha, th_low, th_high);
	else
	    computeEdge<DericheConvolver2<float> >(in, edge, laplacian,
						   alpha, th_low, th_high);

	Image<RGB>	out;
	superImpose(in, edge, out);
	out.save(cout, ImageBase::RGB_24);
    }
    catch (exception& err)
    {
	cerr << err.what() << endl;
	return 1;
    }
    
    return 0;
}
