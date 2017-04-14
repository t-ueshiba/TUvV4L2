/*
 *  $Id: slisttest.cc,v 1.3 2009-09-04 05:48:39 ueshiba Exp $
 */
#include <iostream>
#include <algorithm>
#include "TU/List.h"

namespace TU
{
class Int : public List<Int>::Node
{
  public:
    Int(int i)	:_i(i)		{}

    operator int()	const	{return _i;}

  private:
    int		_i;
};
}

int
main()
{
    using namespace	std;
    using namespace	TU;

    List<Int>	list;
    Int		zero(0), one(1), two(2), three(3), four(4), five(5), six(6);
    
    list.push_front(zero);
    list.push_front(one);
    list.push_front(two);
    list.push_front(three);
    list.push_front(four);
    list.push_front(five);
    list.push_front(six);

    List<Int>::ConstIterator	ci;
    for (ci = list.begin(); ci != list.end(); ++ci)
	cout << ' ' << *ci;
    cout << endl;
    
    List<Int>::Iterator	i;
    for (i = list.begin(); i != list.end(); )
	if (*i == 0)
	{
	    Int&	x = list.erase(i);
	    cout << "deleted item: " << x << endl;
	}
	else
	    ++i;

    for (i = list.begin(); i != list.end(); ++i)
	cout << ' ' << *i;
    cout << endl;

    replace(list.begin(), list.end(), 5, 4);
    for (ci = list.begin(); ci != list.end(); ++ci)
	cout << ' ' << *ci;
    cout << endl;

	return 0;
}
