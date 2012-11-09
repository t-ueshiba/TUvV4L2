/*
 *  $Id$
 */
#include "TU/Collection++.h"

namespace TU
{
ObjTreeBase::Node*
ObjTreeBase::Node::add(const Object* p, Compare compare, int& err)
{
    if (this == 0)
    {
	err = 0;
	return new Node(p);
    }

    int comp = (*compare)(p, _p);
    if (comp < 0)				// p < _p
	_left = _left->add(p, compare, err);
    else if (comp > 0)				// p > _p
	_right = _right->add(p, compare, err);
    else					// p == _p (duplicate keys!!)
	err = -1;
    return balance();
}

const ObjTreeBase::Node*
ObjTreeBase::Node::find(const Object* p, Compare compare) const
{
    if (this == 0)				// specified p not found
	return 0;

    int comp = (*compare)(p, _p);
    if (comp < 0)				// p < _p
	return _left->find(p, compare);
    else if (comp > 0)				// p > _p
	return _right->find(p, compare);
    else					// p == _p
	return this;
}

ObjTreeBase::Node*
ObjTreeBase::Node::detach(const Object* p, Compare compare, int& err)
{
    if (this == 0)				// specified p not found
    {
	err = -1;
	return this;
    }

    int comp = (*compare)(p, _p);
    if (comp < 0)				// p < _p
	_left = _left->detach(p, compare, err);
    else if (comp > 0)				// p > _p
	_right = _right->detach(p, compare, err);
    else					// p == _p
    {
	err = 0;
	if (_left == 0)
	    return _right;
	Node* ret = _left->detach_max();
	ret->_right = _right;
	return ret->balance();
    }
    return balance();
}

ObjTreeBase::Node*
ObjTreeBase::Node::detach_max()
{
    if (_right == 0)
	return this;
    Node* ret = _right->detach_max();
    _right = ret->_left;
    ret->_left = balance();
    return ret;
}

ObjTreeBase::Node*
ObjTreeBase::Node::balance()
{
    Node* ret = this;
    switch (_right->depth() - _left->depth())
    {
      case -2:
	if (_left->_left->depth() < _left->_right->depth())
	    _left = _left->rotate_left();
	ret = rotate_right();
	break;
      case 2:
	if (_right->_left->depth() > _right->_right->depth())
	    _right = _right->rotate_right();
	ret = rotate_left();
	break;
    }
    return ret->update_depth();
}

ObjTreeBase::Node*
ObjTreeBase::Node::rotate_right()
{
    Node* ret = _left;
    _left = ret->_right;
    ret->_right = update_depth();
    return ret;
}

ObjTreeBase::Node*
ObjTreeBase::Node::rotate_left()
{
    Node* ret = _right;
    _right = ret->_left;
    ret->_left = update_depth();
    return ret;
}

inline int	max(int x, int y)	{return (x > y ? x : y);}

ObjTreeBase::Node*
ObjTreeBase::Node::update_depth()
{
    _d = 1 + max(_left->depth(), _right->depth());
    return this;
}

void
ObjTreeBase::Node::saveGuts(std::ostream& out) const
{
    out.write((const char*)&_d, sizeof(_d));
}

void
ObjTreeBase::Node::restoreGuts(std::istream& in)
{
    in.read((char*)&_d, sizeof(_d));
}

#ifdef TUCollectionPP_DEBUG
void
ObjTreeBase::Node::check_depth() const
{
    using namespace	std;

    if (this == 0)
	return;

    _left->check_depth();
    int d = _right->real_depth() - _left->real_depth();
    switch (d)
    {
      case -1:
      case 0:
      case 1:
	break;
      default:
	cerr << "Unbalanced Node: \"Right depth\" - \"Left depth\" = " << d
	     << endl;
	break;
    }
    _right->check_depth();
}    

int
ObjTreeBase::Node::real_depth() const
{
    return (this ? 1 + max(_left->real_depth(), _right->real_depth()) : 0);
}

#endif // TUCollectionPP_DEBUG
}
