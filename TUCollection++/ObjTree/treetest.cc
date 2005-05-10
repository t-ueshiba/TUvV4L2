#include <fstream>
#include "Int.h"
#include "TU/Collection++.h"

namespace TU
{
#ifdef DEBUG
void
ObjTreeBase::Node::print() const
{
    if (this == 0)
	return;

    _left->print();
    cout << (Int*)_p << endl;
    _right->print();
}    
#endif

/*---------------------------------------------------*/

static int
comp(const Int* p, const Int* q)
{
    if (p->value() < q->value())
	return -1;
    else if (p->value() > q->value())
	return 1;
    else
	return 0;
}

}

main()
{
    using namespace	std;
    using namespace	TU;
    
    Ptr<ObjTree<Int> >	tree = new ObjTree<Int>(comp);
    Ptr<Int>		ip;
    int			i;
    char		command[10];
    
    for (;;)
    {
	cerr << ">> ";
	cin >> command;
	if (!cin)
	    break;
	
	switch (command[0])
	{
	  case 'a':
	    cin >> i;
	    ip = Int::new_Int(i);
	    if (tree->add(ip))
		cout << "\tAdded: \"" << ip << '"' << endl;
	    else
		cerr << "\t\"" << ip << "\" already added!" << endl;
	    break;
	  case 'f':
	    cin >> i;
	    ip = Int::new_Int(i);
	    if (tree->find(ip))
		cout << "\tFound: \"" << ip << '"' << endl;
	    else
		cerr << "\t\"" << ip << "\" not found!" << endl;
	    break;
	  case 'd':
	    cin >> i;
	    ip = Int::new_Int(i);
	    if (tree->detach(ip))
		cout << "\tDetached: \"" << ip << '"' << endl;
	    else
		cerr << "\t\"" << ip << "\" not found!" << endl;
	    break;
	  case 's':
	  {
	    ofstream out("tmp.dat", ios::out);
	    tree->save(out);
	    out.close();
	  }
	    break;
	  case 'r':  
	  {
	    ifstream in("tmp.dat", ios::in);
	    tree = ObjTree<Int>::restore(in);
	    // BUG!! ObjTreeBase::compare must be set again here!!
	    in.close();
	  }
	    break;
#ifdef TUCollectionPP_DEBUG
	  case 'c':
	    tree->check_depth();
	    break;
	  case 'p':
	    tree->print(cout);
	    cout << endl;
	    break;
#endif
	  default:
	    cerr << "\tUnknown command: " << command << endl;
	    break;
	}
    }
}
