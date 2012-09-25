/*
 *  $Id: main.cc,v 1.1 2012-07-23 00:45:48 ueshiba Exp $
 */
#include <cstdlib>
#include "TU/BoxFilter.h"
#include "TU/Array++.h"
#include <boost/iterator/transform_iterator.hpp>
#include <boost/iterator/zip_iterator.hpp>
#include <boost/foreach.hpp>

namespace TU
{
typedef int				value_type;
typedef Array<value_type>		array_type;
typedef array_type::const_iterator	array_iterator;
typedef Array2<array_type>		array2_type;
typedef array2_type::const_iterator	array2_iterator;

template <class T>
struct square
{
    typedef T	result_type;

    template <class ARG>
    result_type	operator ()(ARG x) const
		{
		    return result_type(x) * result_type(x);
		}
};

typedef boost::transform_iterator<square<value_type>,
				  array_iterator>	square_iterator;

typedef boost::transform_iterator<seq_transform<array_type,
						square<value_type> >,
				  array2_iterator>	array_square_iterator;

template <class T>
struct mul
{
    typedef T	result_type;

    template <class ARG0, class ARG1>
    result_type	operator ()(ARG0 x, ARG1 y) const
		{
		    return result_type(x) * result_type(y);
		}
};

typedef boost::transform_iterator<unarize<mul<value_type> >,
				  boost::zip_iterator<
				      boost::tuple<
					  array_iterator,
					  array_iterator> > >
							mul_iterator;

typedef boost::transform_iterator<unarize<seq_transform<array_type,
							mul<value_type> > >,
				  boost::zip_iterator<
				      boost::tuple<
					  array2_iterator,
					  array2_iterator> > >
							array_mul_iterator;

}

int
main(int argc, char* argv[])
{
    using namespace	std;
    using namespace	TU;

    size_t		w = 3;
    extern char*	optarg;
    for (int c; (c = getopt(argc, argv, "w:")) != -1; )
	switch (c)
	{
	  case 'w':
	    w = atoi(optarg);
	    break;
	}

  // box filterを1D arrayに適用する．
    array_type	a;
    cerr << ">> ";
    cin >> a;

    for (box_filter_iterator<square_iterator>
	     iter(square_iterator(a.begin()), w),
	     end (square_iterator(a.end() + 1 - w));
	 iter != end; ++iter)
	cout << ' ' << *iter;
    cout << endl;

  // box filterを2D arrayに適用する．
    array2_type	A;
    cerr << ">> ";
    cin >> A;

    for (box_filter_iterator<array_square_iterator>
	     row (array_square_iterator(A.begin()), w),
	     rend(array_square_iterator(A.end() + 1 - w));
	 row != rend; ++row)
    {
	for (box_filter_iterator<array_iterator>
		 col(row->begin(), w), cend(row->end() + 1 - w);
	     col != cend; ++col)
	    cout << ' ' << *col;
	cout << endl;
    }
    cout << endl;

  // box filterを2つの1D arrayに適用する．
    array_type	b = a;
    BOOST_FOREACH (value_type& val, b)
	val += 1;

    for (box_filter_iterator<mul_iterator>
	     iter(mul_iterator(
		      boost::make_zip_iterator(
			  boost::make_tuple(a.begin(), b.begin()))), w),
	     end (mul_iterator(
		      boost::make_zip_iterator(
			  boost::make_tuple(a.end() + 1 - w,
					    b.end() + 1 - w))));
	 iter != end; ++iter)
	cout << ' ' << *iter;
    cout << endl << endl;

  // box filterを2つの2D arrayに適用する．
    array2_type	B = A;
    BOOST_FOREACH (array_type& row, B)
	BOOST_FOREACH (value_type& val, row)
	    val += 1;

    for (box_filter_iterator<array_mul_iterator>
	     row (array_mul_iterator(
		      boost::make_zip_iterator(
			  boost::make_tuple(A.begin(), B.begin()))), w),
	     rend(array_mul_iterator(
		      boost::make_zip_iterator(
			  boost::make_tuple(A.end() + 1 - w,
					    B.end() + 1 - w))));
	 row != rend; ++row)
    {
	for (box_filter_iterator<array_iterator>
		 col(row->begin(), w), cend(row->end() + 1 - w);
	     col != cend; ++col)
	    cout << ' ' << *col;
	cout << endl;
    }
    cout << endl;

    return 0;
}
