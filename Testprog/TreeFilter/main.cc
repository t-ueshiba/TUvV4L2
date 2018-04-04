/*
 *  $Id$
 */
#include "TU/Array++.h"
#include "TU/TreeFilter.h"

namespace TU
{
template <class S, class T>
struct Diff
{
    typedef S	argument_type;
    typedef T	result_type;

    result_type	operator ()(argument_type x, argument_type y) const
		{
		    return std::abs(x - y);
		}
};
    
}

int
main()
{
    using namespace	TU;
    using		std::cin;
    using		std::cout;
    using		std::cerr;
    using		std::endl;

    typedef int					value_type;
    typedef float				weight_type;
    typedef Array2<value_type>			array2_type;
    typedef Diff<value_type, weight_type>	wfunc_type;

    weight_type					sigma = 1.0/std::log(2.0);
    array2_type					a({ {1, 2, 4, 8},
						    {10, 20, 40, 80},
						    {100, 200, 400, 800} });
    array2_type					b(a.nrow(), a.ncol());
    boost::TreeFilter<weight_type, wfunc_type>	tf(wfunc_type(), sigma);
    tf.convolve(a.begin(), a.end(), a.begin(), a.end(), b.begin());

    cerr << "-------- all vertices --------" << endl;
    tf.printVertices(cerr);
    cerr << "-------- all edges --------" << endl;
    tf.printEdges(cerr);
    
    return 0;
}
