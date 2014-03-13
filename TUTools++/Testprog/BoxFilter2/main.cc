/*
 *  $Id$
 */
#include <stdlib.h>
#include "TU/BoxFilter.h"

int
main(int argc, char* argv[])
{
    using namespace	std;
    using namespace	TU;

    typedef Array<int>						array_type;
    typedef Array2<array_type>					array2_type;
    typedef box_filter_iterator<array_type::const_iterator>	box_iterator;
    typedef box_filter_iterator<array2_type::const_iterator>	box2_iterator;
    
    size_t		winSize = 3;
    extern char*	optarg;
    for (int c; (c = getopt(argc, argv, "w:")) != -1; )
	switch (c)
	{
	  case 'w':
	    winSize = atoi(optarg);
	    break;
	}

    array_type	a;
    cerr << "a> ";
    cin >> a;

    for (box_iterator iter(a.cbegin(), winSize), end(a.cend());
	 iter != end; ++iter)
	cout << ' ' << *iter;
    cout << endl;
    
    array2_type	A;
    cerr << "A> ";
    cin >> A;

    for (box2_iterator row(A.cbegin(), winSize), rowe(A.cend());
	 row != rowe; ++row)
    {
	for (box_iterator col(row->cbegin(), winSize), cole(row->cend());
	     col != cole; ++col)
	    cout << ' ' << *col;
	cout << endl;
    }
    
    
    return 1;
}
