/*
 *  $Id: PSTree++.h,v 1.2 2002-07-25 02:38:06 ueshiba Exp $
 */
#ifndef __TUPSTreePP_h
#define __TUPSTreePP_h

#include "TU/Heap++.h"
#include "TU/List++.h"

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

template <class S, class T, class CompareX, class CompareY> inline int
PSTree<S, T, CompareX, CompareY>::Node::middle(int il, int ir)
{
    int	ib = (il + 1 + ir) / 2;
    return (ib < ir ? ib : ir - 1);
}
 
}
#endif	// !__TUPSTreePP_h
