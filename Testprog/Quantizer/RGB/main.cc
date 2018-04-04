/*
 *  $Id$
 */
#include <cstdlib>
#include "TU/Image++.h"
#include "TU/Quantizer.h"
#include "TU/Profiler.h"

int
main(int argc, char* argv[])
{
    using namespace	TU;

    typedef RGB		value_type;
    
    size_t		nbins = 10;
    extern char*	optarg;
    for (int c; (c = getopt(argc, argv, "n:")) != -1; )
	switch (c)
	{
	  case 'n':
	    nbins = atoi(optarg);
	    break;
	}
    
    Image<value_type>	image;
    image.restore(std::cin);

    Profiler<>			profiler(2);
    Quantizer2<value_type>	quantizer;
    Image<value_type>		quantizedImage(image.width(), image.height());
    for (size_t n = 0; n < 10; ++n)
    {
	profiler.start(0);
	const auto&	indices = quantizer(image.cbegin(),
					    image.cend(), nbins);
	profiler.start(1);
	for (size_t v = 0; v < quantizedImage.height(); ++v)
	    for (size_t u = 0; u < quantizedImage.width(); ++u)
		quantizedImage[v][u] = quantizer[indices[v][u]];
	profiler.nextFrame();
    }
    profiler.print(std::cerr);
    
    image.save(std::cout);
    quantizedImage.save(std::cout);
    
    return 0;
}
