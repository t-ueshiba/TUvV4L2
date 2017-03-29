#include <list>
#include <random>
#include "TU/WeightedMedianFilter.h"

namespace TU
{
template <class T>
struct IdentityFunc
{
    typedef T	result_type;
    typedef T	argument_type;
    
    T	operator ()(T, T) const	{ return 1; }
};
    
}

int
main()
{
    using namespace	TU;
    using		std::cerr;
    using		std::endl;
    
    typedef float			value_type;
    typedef Array2<Array<value_type> >	data_type;
    typedef IdentityFunc<value_type>	wfunc_type;
    
    std::default_random_engine			generator;
    std::uniform_real_distribution<value_type>	distribution(0.0, 1.0);
    data_type					in(7, 7);
    for (auto& row : in)
	for (auto& x : row)
	    x = distribution(generator);
    cerr << "--- in ---\n" << in;

    WeightedMedianFilter2<value_type, wfunc_type> wmf(wfunc_type(), 3, 8, 8);
    data_type					  out(in.nrow(), in.ncol());
    wmf.convolve(in.cbegin(), in.cend(), in.cbegin(), in.cend(), out.begin());

    cerr << "--- out ---\n" << out;

    return 0;
}
