#include <fstream>
#include "Int.h"
#include "TU/Collection++.h"

main()
{
    using namespace	std;
    using namespace	TU;
    
    Ptr<ObjList<Int> >	list = new ObjList<Int>;
    
    list->add(Int::new_Int(0));
    list->add(Int::new_Int(1));
    list->add(Int::new_Int(2));
    list->add(Int::new_Int(3));
    list->add(Int::new_Int(4));
    list->add(Int::new_Int(6));
    list->add(Int::new_Int(5));
    list->add(Int::new_Int(6));	// (6 5 6 4 3 2 1 0)

    ObjList<Int>::Iterator iter(list);
    for (; iter; ++iter)// (6 5 6 4 3 2 1 0)
	cout << iter;
    cout << endl;

    ofstream out("tmp.dat", ios::out);
    list->save(out);
    out.close();
    
    for (iter.head(); iter; ++iter)
	if (iter->value() == 6)
	    iter.detach();

    for (iter.head(); iter; ++iter)			// (5 4 3 2 1 0)
	cout << iter;
    cout << endl;

    ifstream in("tmp.dat", ios::in);
    list = ObjList<Int>::restore(in);

    ObjList<Int>::Iterator iter2(list);
    for (; iter2; ++iter2)
	cout << iter2;
    cout << endl;

    Ptr<Int>		t;
    for (iter2.head(); iter2; ++iter2)
	if ((t = iter2)->value() == 2)
	    break;
    list->detach(t);
    
    for (iter2.head(); iter2; )
	cout << iter2++;
    cout << endl;

    cout << list->head() << endl;
}


#ifdef __GNUG__
#  include "TU/Collection++.cc"
#endif

namespace TU
{
template const Object::Desc	ObjList<Int>::_desc;
}

#ifdef __GNUG__
#  include "TU/Collection++.cc"
#endif

