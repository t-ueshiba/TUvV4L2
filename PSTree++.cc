/*
 *  $Id: PSTree++.cc,v 1.1.1.1 2002-07-25 02:14:16 ueshiba Exp $
 */
#include "TU/PSTree++.h"

namespace TU
{
/************************************************************************
*  class PSTree<S, T, CompareX, CompareY>				*
************************************************************************/
template <class S, class T, class CompareX, class CompareY>
PSTree<S, T, CompareX, CompareY>::PSTree(Array<T>& array,
					 CompareX compareX,
					 CompareY compareY)
    :_array(array), _root(0), _compareX(compareX), _compareY(compareY)
{
    if (array.dim() > 0)
    {
	sort(array, _compareY);
	_root = new Node(array, 0, array.dim(), _compareX);
    }
}

template <class S, class T, class CompareX, class CompareY> const T*
PSTree<S, T, CompareX, CompareY>::closestY(const S& q) const
{
    const Node*	node = _root->closestY(q, _compareX, _compareY);
    return (node != 0 ? &node->p() : 0);
}

template <class S, class T, class CompareX, class CompareY> void
PSTree<S, T, CompareX, CompareY>::inRegion(const S& q, List& list) const
{
    _root->inRegion(q, list, _compareX, _compareY);
}

template <class S, class T, class CompareX, class CompareY> void
PSTree<S, T, CompareX, CompareY>::inRegion(const S& p, const S& q,
					   List& list) const
{
    _root->inRegion(p, q, list, _compareX, _compareY);
}

/************************************************************************
*  class PSTree<S, T, CompareX, CompareY>::Node				*
************************************************************************/
template <class S, class T, class CompareX, class CompareY>
PSTree<S, T, CompareX, CompareY>::Node::Node(Array<T>& array,
					     int il, int ir,
					     CompareX compareX)
    :_p(array[shift(array, il, ir, compareX)]), _b(array[middle(il, ir)]),
     _l(0), _r(0)
{
    int	ib = (il + 1 + ir) / 2;
    if (il + 1 < ib)
	_l = new Node(array, il+1, ib, compareX);
    if (ib < ir)
	_r = new Node(array, ib, ir, compareX);
}

template <class S, class T, class CompareX, class CompareY> int
PSTree<S, T, CompareX, CompareY>::Node::shift(Array<T>& array,
					      int il, int ir,
					      CompareX compareX)
{
    int im = il;
    for (int i = il+1; i < ir; ++i)	// Find the left-most element.
	if (compareX(array[i], array[im]))
	    im = i;

    T	tmp = array[im];
    while (im-- > il)	// Shift all the elements left of im to right.
	array[im+1] = array[im];
    array[il] = tmp;
    
    return il;
}

/*
 *  compareX(_p, q) && compareY(_p, q) を満たすノード _p のうち、
 *  他のどの _p' に対しても !compareY(_p, _p') となるものを返す。
 */
template <class S, class T, class CompareX, class CompareY>
const typename PSTree<S, T, CompareX, CompareY>::Node*
PSTree<S, T, CompareX, CompareY>::Node::closestY(const S& q,
						 CompareX compareX,
						 CompareY compareY) const
{
    if (!compareX(_p, q))
	return 0;

    const Node*	node = (_r != 0 ? _r->closestY(q, compareX, compareY) : 0);
    if (node == 0 && _l != 0)
	node = _l->closestY(q, compareX, compareY);
    if (compareY(_p, q) && (node == 0 || !compareY(_p, node->p())))
	return this;
    else
	return node;
}

/*
 *  compareX(_p, q) && compareY(_p, q) を満たすノード _p 全てを
 *  列挙して list に返す。
 */
template <class S, class T, class CompareX, class CompareY> void
PSTree<S, T, CompareX, CompareY>::Node::inRegion(const S& q, List& list,
						 CompareX compareX,
						 CompareY compareY) const
{
    if (!compareX(_p, q))
	return;

    if (_l != 0)
	_l->inRegion(q, list, compareX, compareY);
    if (_r != 0 && compareY(_b, q))
	_r->inRegion(q, list, compareX, compareY);
    if (compareY(_p, q))
	list.push_front(*new ListNode(_p));
}

/*
 *  !compareX(_p, p) && !compareY(_p, p) &&
 *   compareX(_p, q) &&  compareY(_p, q) を満たすノード _p 全てを
 *  列挙して list に返す。
 */
template <class S, class T, class CompareX, class CompareY> void
PSTree<S, T, CompareX, CompareY>::Node::inRegion(const S& p, const S& q,
						 List& list,
						 CompareX compareX,
						 CompareY compareY) const
{
    if (!compareX(_p, q))
	return;

    if (_l != 0 && !compareY(_b, p))
	_l->inRegion(p, q, list, compareX, compareY);
    if (_r != 0 && compareY(_b, q))
	_r->inRegion(p, q, list, compareX, compareY);
    if (!compareX(_p, p) && !compareY(_p, p) && compareY(_p, q))
	list.push_front(*new ListNode(_p));
}

template <class S, class T, class CompareX, class CompareY> void
PSTree<S, T, CompareX, CompareY>::Node::print(std::ostream& out,
					      char kind) const
{
    static int	indent = 0;
    
    if (this == 0)
	return;
    
    for (int i = 0; i < indent; ++i)
	out << ' ';
    out << kind << ": " << _p;

    ++indent;
    _l->print(out, 'L');
    _r->print(out, 'R');
    --indent;
}
 
}
