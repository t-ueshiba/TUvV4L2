/*
 *  平成14-19年（独）産業技術総合研究所 著作権所有
 *  
 *  創作者：植芝俊夫
 *
 *  本プログラムは（独）産業技術総合研究所の職員である植芝俊夫が創作し，
 *  （独）産業技術総合研究所が著作権を所有する秘密情報です．著作権所有
 *  者による許可なしに本プログラムを使用，複製，改変，第三者へ開示する
 *  等の行為を禁止します．
 *  
 *  このプログラムによって生じるいかなる損害に対しても，著作権所有者お
 *  よび創作者は責任を負いません。
 *
 *  Copyright 2002-2007.
 *  National Institute of Advanced Industrial Science and Technology (AIST)
 *
 *  Creator: Toshio UESHIBA
 *
 *  [AIST Confidential and all rights reserved.]
 *  This program is confidential. Any using, copying, changing or
 *  giving any information concerning with this program to others
 *  without permission by the copyright holder are strictly prohibited.
 *
 *  [No Warranty.]
 *  The copyright holder or the creator are not responsible for any
 *  damages caused by using this program.
 *  
 *  $Id: PSTree.h,v 1.5 2012-08-29 21:17:08 ueshiba Exp $
 */
/*!
  \file		PSTree.h
  \brief	クラス TU::PSTree の定義と実装
*/
#ifndef __TUPSTree_h
#define __TUPSTree_h

#include "TU/Heap.h"
#include "TU/List.h"

namespace TU
{
/************************************************************************
*  class PSTree<S, T, CompareX, CompareY>				*
************************************************************************/
/*
 *  PSTree は、２つの強全順序 compareX, compareY（任意の p, q に対して
 *  compareX(p, q), compareX(q, p), p == q のどれか１つだけが成り立ち、
 *  かつ推移律を満たす。よって、compareX(p, p) は常に偽。）が与えられ
 *  たとき,
 *	1. 親ノード _p と子ノード _q に対して !compareX(_q, _p)
 *	2. 左部分木 _l と右部分木 _r に対して !compareY(_r, _l)
 *  が満たされるように構成される。親ノードと子ノードの間には compareY
 *  について何ら決まった関係は無い。
 */
template <class S, class T, class CompareX, class CompareY>
class PSTree
{
  public:
    class ListNode : public TU::List<ListNode>::Node
    {
      public:
	ListNode(const T& pp)	:p(pp)	{}

	const T&	p;
    };

    typedef TU::List<ListNode>	List;

    class Node
    {
      public:
	Node(Array<T>& array, int il, int ir, CompareX compareX);
	~Node()						{delete _r; delete _l;}

	const T&	p()				const	{return _p;}
	const Node*	closestY(const S& q,
				 CompareX compareX,
				 CompareY compareY)	const	;
	void		inRegion(const S& q,
				 List& list,
				 CompareX compareX,
				 CompareY compareY)	const	;
	void		inRegion(const S& p,
				 const S& q,
				 List& list,
				 CompareX compareX,
				 CompareY compareY)	const	;
	void		print(std::ostream& out,
			      char kind)		const	;
	
      private:
	static int	shift(Array<T>& array, int il, int ir,
			      CompareX compareX)		;
	static int	middle(int il, int ir)			;
	
	const T		_p;
	const T		_b;
	Node*		_l;
	Node*		_r;
    };

  public:
    PSTree(Array<T>& array, CompareX compareX, CompareY compareY)	;
    ~PSTree()						{delete _root;}

  /*
   *  compareX(_p, q) && compareY(_p, q) を満たすノード _p のうち、
   *  他のどの _p' に対しても !compareY(_p, _p') となるものを返す。
   */
    const T*	closestY(const S& q)		 const	;
  /*
   *  compareX(_p, q) && compareY(_p, q) を満たすノード _p 全てを
   *  列挙して list に返す。
   */
    void	inRegion(const S& q, List& list) const	;
  /*
   *  !compareX(_p, p) && !compareY(_p, p) &&
   *   compareX(_p, q) &&  compareY(_p, q) を満たすノード _p 全てを
   *  列挙して list に返す。
   */
    void	inRegion(const S& p,
			 const S& q,
			 List& list)	 const	;
    void	print(std::ostream& out) const	{_root->print(out, 'C');}
    
  private:
    Array<T>&		_array;
    Node*		_root;
    const CompareX	_compareX;
    const CompareY	_compareY;
};

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

/*
 *  class PSTree<S, T, CompareX, CompareY>::Node
 */
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

template <class S, class T, class CompareX, class CompareY> inline int
PSTree<S, T, CompareX, CompareY>::Node::middle(int il, int ir)
{
    int	ib = (il + 1 + ir) / 2;
    return (ib < ir ? ib : ir - 1);
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
#endif	// !__TUPSTree_h
