/*
 *  $Id: main.cc,v 1.1 2002-07-25 04:36:08 ueshiba Exp $
 */
#include <iostream>
#include "TU/Allocator++.h"
#ifdef __GNUG__
#  include "TU/Allocator++.cc"
#  include "TU/List++.cc"
#endif

namespace TU
{
class Tmp
{
  public:
    Tmp(int ii=0)	:i(ii)	{}
    ~Tmp()			{}

    void*	operator new(size_t, void* p)	{return p;}
	
    int		i;
};
}

#define NELMS	100

int
main()
{
    using namespace	std;
    using namespace	TU;
    
    Allocator<Tmp>	allocator(4);
    
    cerr << "sizeof(Tmp) = " << sizeof(Tmp) << endl;
    
    Tmp*	tmp[NELMS];
    for (int i = 0; i < NELMS; ++i)
	tmp[i] = new(allocator.alloc()) Tmp(i);
    tmp[2]->~Tmp();
    allocator.free(tmp[2]);
    tmp[9]->~Tmp();
    allocator.free(tmp[9]);
    tmp[2] = new(allocator.alloc()) Tmp(22);

    for (Allocator<Tmp>::Enumerator enumerator(allocator);
	 enumerator; ++enumerator)
	cerr << ' ' << enumerator->i;
    cerr << endl;
    
    return 0;
}
