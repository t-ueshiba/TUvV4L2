#include <list>
#include <random>
#include "TU/Quantizer.h"

int
main()
{
    using namespace	TU;
    using		std::cerr;
    using		std::endl;
    
    typedef float			value_type;
  //typedef u_int8_t			value_type;
    typedef std::list<value_type>	data_type;
    
    std::default_random_engine			generator;
    std::uniform_real_distribution<float>	distribution(0.0, 100.0);
    data_type					in(100);
    for (auto& x : in)
	x = distribution(generator);
    cerr << "--- in ---\n";
    std::copy(in.cbegin(), in.cend(),
	      std::ostream_iterator<value_type>(cerr, " "));
    cerr << endl;

    size_t			nbins = 10;
    Quantizer<value_type>	quantizer;
    const auto&			indices = quantizer(in.cbegin(),
						    in.cend(), nbins);

    cerr << "--- out ---" << endl;
    for (auto idx : indices)
	cerr << ' ' << quantizer[idx];
    cerr << endl;

    cerr << "--- bins ---" << endl;
    for (size_t i = 0; i < quantizer.size(); ++i)
	cerr << ' ' << quantizer[i];
    cerr << endl;

    return 0;
}
