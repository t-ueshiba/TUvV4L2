#include <fstream>
#include "Int.h"
#include "TU/Collection++.h"


main()
{
    using namespace	std;
    using namespace	TU;
    
    Ptr<ObjDList<Int> >	list = new ObjDList<Int>;
    Ptr<Int>		t;
    
    list->addAtHead(Int::new_Int(0));	// (0)
    list->addAtTail(Int::new_Int(1));	// (0 1)
    list->addAtHead(Int::new_Int(2));	// (2 0 1)
    list->addAtTail(Int::new_Int(3));	// (2 0 1 3)
    list->addAtHead(Int::new_Int(4));	// (4 2 0 1 3)
    list->addAtTail(Int::new_Int(6));	// (4 2 0 1 3 6)
    list->addAtHead(Int::new_Int(5));	// (5 4 2 0 1 3 6)
    list->addAtTail(Int::new_Int(6));	// (5 4 2 0 1 3 6 6)
    
    ObjDList<Int>::Iterator iter(list);
    for (; iter; ++iter)
	cout << iter;
    cout << endl;

    ofstream out("tmp.dat", ios::out);
    list->save(out);
    out.close();
    
    for (iter.head(); iter; ++iter)
	if (iter->value() == 6)
	    iter.detach();

    for (iter.head(); iter; ++iter)
	cout << iter;
    cout << endl;

    ifstream in("tmp.dat", ios::in);
    list = ObjDList<Int>::restore(in);

    ObjDList<Int>::Iterator iter2(list);
    for (; iter2; ++iter2)
	cout << iter2;
    cout << endl;

    for (iter2.head(); iter2; ++iter2)
	if ((t = iter2)->value() == 2)
	    break;
    list->detach(t);
    
    for (iter2.tail(); iter2; --iter2)
	cout << iter2;
    cout << endl;
}
