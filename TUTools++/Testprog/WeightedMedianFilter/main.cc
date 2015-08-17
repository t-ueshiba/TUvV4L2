#include <list>
#include <random>
#include "TU/WeightedMedianFilter.h"

namespace TU
{
template <class T>
struct UnitFunc
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
    typedef std::list<value_type>	data_type;
    typedef UnitFunc<value_type>	wfunc_type;
    
    std::default_random_engine			generator;
    std::uniform_real_distribution<value_type>	distribution(0.0, 1.0);
    data_type					in(16);
    for (auto& x : in)
	x = distribution(generator);

    cerr << "--- in ---\n";
    std::copy(in.cbegin(), in.cend(),
	      std::ostream_iterator<value_type>(cerr, " "));
    cerr << endl;

    WeightedMedianFilter<value_type, wfunc_type> wmf(wfunc_type(), 5, 8, 8);
    data_type					 out(in.size());
    wmf.convolve(in.cbegin(), in.cend(), in.cbegin(), in.cend(), out.begin());

    cerr << "--- out ---\n ";
    std::copy(out.cbegin(), out.cend(),
	      std::ostream_iterator<value_type>(cerr, " "));
    cerr << endl;
    
    return 0;
}
