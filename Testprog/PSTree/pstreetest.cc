/*
 *  $Id: pstreetest.cc,v 1.4 2008-09-02 05:13:17 ueshiba Exp $
 */
#include "TU/Geometry++.h"
#include "TU/PSTree.h"
#include <fstream>

namespace TU
{
typedef Point2<int>	P2;

struct orderedX
{
    int
    operator ()(const P2& p0, const P2& p1) const
    {
	return p0[0] < p1[0];
    }
};

struct orderedY
{
    int
    operator ()(const P2& p0, const P2& p1) const
    {
	return p0[1] < p1[1];
    }
};

typedef PSTree<P2, P2, orderedX, orderedY>	MyPSTree;
}


main()
{
    using namespace	std;
    using namespace	TU;
    
    Array<P2>	a;
    ifstream	in("data");
    in >> a;
    cerr << a;

    MyPSTree	pstree(a, orderedX(), orderedY());
    cerr << "*** Result ***" << endl;
    pstree.print(cout);

    P2	q;
    cerr << "\nBorder-q >> ";
    cin >> q[0] >> q[1];
    cerr << "  q = " << q << endl;

    const P2*	pp = pstree.closestY(q);
    if (pp == 0)
	cerr << "  no solutions!!" << endl;
    else
	cerr << "*** closestY(q) ***\n" << *pp;

    MyPSTree::List	list;
    pstree.inRegion(q, list);
    cerr << "*** inRegion(q) ***" << endl;
    while (!list.empty())
    {
	MyPSTree::ListNode* node = &list.front();
	list.pop_front();
	cerr << node->p;
	delete node;
    }

    P2	p;
    cerr << "\nBorder-p >> ";
    cin >> p[0] >> p[1];
    cerr << "  p = " << p << endl;
    pstree.inRegion(p, q, list);
    cerr << "*** inRegion(p, q) ***" << endl;
    while (!list.empty())
    {
	MyPSTree::ListNode* node = &list.front();
	list.pop_front();
	cerr << node->p;
	delete node;
    }
}
