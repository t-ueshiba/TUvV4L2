/*
 *  $Id$
 */
#include <cstdlib>
#include "TU/Image++.h"
#include "TU/Quantizer.h"

int
main(int argc, char* argv[])
{
    using namespace	TU;

    size_t		nbins = 10;
    extern char*	optarg;
    for (int c; (c = getopt(argc, argv, "n:")) != -1; )
	switch (c)
	{
	  case 'n':
	    nbins = atoi(optarg);
	    break;
	}
    
    Image<RGB>	image;
    image.restore(std::cin);

    Quantizer2<RGB>	quantizer;
    const auto&		indices = quantizer(image.cbegin(),
					    image.cend(), nbins);

    Image<RGB>	quantizedImage(image.width(), image.height());
    for (size_t v = 0; v < quantizedImage.height(); ++v)
	for (size_t u = 0; u < quantizedImage.width(); ++u)
	    quantizedImage[v][u] = quantizer[indices[v][u]];
    
    image.save(std::cout);
    quantizedImage.save(std::cout);
    
    return 0;
}
