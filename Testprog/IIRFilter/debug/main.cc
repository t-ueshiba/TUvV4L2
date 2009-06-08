/*
 * $Id: main.cc,v 1.1 2009-06-08 23:46:21 ueshiba Exp $
 */
#include <stdlib.h>
#include <fstream>
#include "TU/Image++.h"
#include "TU/DericheConvolver.h"

int
main(int argc, char* argv[])
{
    using namespace	std;
    using namespace	TU;

    bool	quad = false, backward = false;
    for (int c; (c = getopt(argc, argv, "4b")) != EOF; )
	switch (c)
	{
	  case '4':
	    quad = true;
	    break;
	  case 'b':
	    backward = true;
	    break;
	}
    
    Array<u_char>	src(640);
    fstream		in("data");
    if (!in)
    {
	cerr << "Input data file not found!" << endl;
	return 1;
    }
    src.restore(in);

    Array<float>	dst;
    float		c[] = {0.1, -0.3, 0.2, -0.9,
			       0.4, -0.7, 0.8, -0.6};
    if (quad)
    {
	IIRFilter<4u>	iir;
	iir.initialize(c);
	
	if (backward)
	    iir.backward(src, dst);
	else
	    iir.forward(src, dst);
    }
    else
    {
	IIRFilter<2u>	iir;
	iir.initialize(c);
	
	if (backward)
	    iir.backward(src, dst);
	else
	    iir.forward(src, dst);
    }
    
    for (int i = 0; i < dst.dim(); ++i)
	cout << i << ": (" << int(src[i]) << ")\t[" << dst[i] << "]" << endl;
    
    return 0;
}
