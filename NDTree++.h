/*
 *  $Id: NDTree++.h,v 1.3 2011-08-22 00:06:25 ueshiba Exp $
 */
/*!
  \file		NDTree++.h
  \brief	クラス TU::NDTree の定義と実装
*/
#ifndef __TUNDTreePP_h
#define __TUNDTreePP_h

#include <stack>
#include "TU/Array++.h"

namespace TU
{
/************************************************************************
*  class NDTree<T, D>							*
************************************************************************/
//! D次元空間を表現する2^D分木を表すクラス
/*!
  \param T	要素の型
  \param D	空間の次元，D=2のときquad tree, D=3のときoctreeとなる
 */
template <class T, u_int D>
class NDTree
{
  private:
    enum	{Dim = D, NChildren = (1 << D)};

    class	Node;
    class	Branch;
    class	Leaf;
    
  public:
    typedef T			value_type;	//!< 要素の型
    typedef value_type&		reference;	//!< 要素への参照
    typedef const value_type&	const_reference;//!< 定数要素への参照
    typedef value_type*		pointer;	//!< 要素へのポインタ
    typedef const value_type*	const_pointer;	//!< 定数要素へのポインタ
    typedef Array<int, FixedSizedBuf<int, D> >
				position_type;	//!< 空間中の位置

  //! 2^D分木のための前進反復子
    template <class S>
    class Iterator : public std::iterator<std::forward_iterator_tag, S>
    {
      public:
	typedef S		value_type;	//!< 要素の型
	typedef value_type&	reference;	//!< 要素への参照
	typedef value_type*	pointer;	//!< 要素へのポインタ

      private:
	struct NodeInfo
	{
	    NodeInfo(Node* n, const position_type& p, u_int l)
		:node(n), dp(p), len(l)					{}
	    
	    Node*		node;		//!< ノードへのポインタ
	    position_type	dp;		//!< ノードの相対位置
	    u_int		len;		//!< ノードのセル長
	};
	
      public:
			Iterator()					;
			Iterator(const NDTree& tree)			;
	
	position_type	position()				const	;
	u_int		length()				const	;
	reference	operator *()				const	;
	pointer		operator ->()				const	;
	Iterator&	operator ++()					;
	Iterator	operator ++(int)				;
	bool		operator ==(const Iterator& iter)	const	;
	bool		operator !=(const Iterator& iter)	const	;

      private:
	position_type	new_dp(u_int idx)			const	;
	
      private:
	position_type		_org;		//!< 2^D分木の原点の位置
	Leaf*			_leaf;		//!< この反復子が指している葉
	position_type		_dp;		//!< 葉の相対位置
	u_int			_len;		//!< 葉のセル長
	std::stack<NodeInfo>	_fringe;	//!< 未訪問のノードたち
    };

    typedef Iterator<value_type>	iterator;	//!< 反復子
    typedef Iterator<const value_type>	const_iterator;	//!< 定数反復子

  public:
			NDTree()					;
			NDTree(const NDTree& tree)			;
			~NDTree()					;
    NDTree&		operator =(const NDTree& tree)			;

    const position_type&
			origin()				const	;
    u_int		length0()				const	;
    u_int		size()					const	;
    bool		empty()					const	;
    void		clear()						;
    const_pointer	find(const position_type& pos)		const	;
    void		insert(const position_type& pos,
			       const_reference val)			;
    void		erase(const position_type& pos)			;
    
    iterator		begin()						;
    const_iterator	begin()					const	;
    iterator		end()						;
    const_iterator	end()					const	;
    
    std::ostream&	put(std::ostream& out)			const	;
    std::istream&	get(std::istream& in)				;
    std::ostream&	print(std::ostream& out)		const	;
    
  private:
    bool		out_of_range(const position_type& pos)	const	;

  private:
    class Node
    {
      public:
	virtual		~Node()						;

	virtual Node*	clone()					 const	= 0;
	virtual u_int	size()					 const	= 0;
	virtual const_pointer
			find(const position_type& dp, u_int len) const	= 0;
	virtual void	insert(const position_type& dp,
			       const_reference val, u_int len)		= 0;
	virtual Node*	erase(const position_type& dp, u_int len)	= 0;
	virtual Branch* branch()				  	= 0;
	virtual Leaf*	leaf()					  	= 0;
	virtual void	print(std::ostream& out, u_int nindents) const	= 0;
	static Node*	create(const position_type& dp,
			       const_reference val, u_int len)		;
    };

    class Branch : public Node
    {
      public:
		        Branch()					;
        virtual		~Branch()					;

	virtual Node*	clone()					 const	;
	virtual u_int	size()					 const	;
	virtual const_pointer
			find(const position_type& dp, u_int len) const	;
	virtual void	insert(const position_type& dp,
			       const_reference val, u_int len)		;
	virtual Node*	erase(const position_type& dp, u_int len)	;
	virtual Branch* branch()				  	;
	virtual Leaf*	leaf()					  	;
	virtual void	print(std::ostream& out, u_int nindents) const	;
	static Node*	ascend(Node* node, u_int idx)			;
	Node*		descend(u_int& idx)				;

      private:
		        Branch(const Branch&)				;
        Branch&		operator =(const Branch&)			;

	static u_int	child_idx(const position_type& dp, u_int len)	;

	friend class	Iterator<value_type>;
	friend class	Iterator<const value_type>;

      private:
        Node*		_children[NChildren];
    };

    class Leaf : public Node
    {
      public:
			Leaf(const_reference val)			;
	virtual		~Leaf()						;
	
	virtual Node*	clone()					 const	;
	virtual u_int	size()					 const	;
	virtual const_pointer
			find(const position_type& dp, u_int len) const	;
	virtual void	insert(const position_type& dp,
			       const_reference val, u_int len)		;
	virtual Node*	erase(const position_type& dp, u_int len)	;
	virtual Branch* branch()				  	;
	virtual Leaf*	leaf()					  	;
	virtual void	print(std::ostream& out, u_int nindents) const	;

	friend class	Iterator<value_type>;
	friend class	Iterator<const value_type>;

      private:
        value_type	_val;
    };

  private:
    position_type	_org;
    u_int		_len0;
    Node*		_root;
};

/************************************************************************
*  class NDTree<T, D>							*
************************************************************************/
//! D次元空間を表現する2^D分木を生成する．
template <class T, u_int D> inline
NDTree<T, D>::NDTree()
    :_org(), _len0(0), _root(0)
{
}

//! コピーコンストラクタ
/*!
  \param tree	コピー元の2^D分木
*/
template <class T, u_int D> inline
NDTree<T, D>::NDTree(const NDTree& tree)
    :_org(tree._org), _len0(tree._len0), _root(0)
{
    if (tree._root)
	_root = tree._root->clone();
}

//! デストラクタ
template <class T, u_int D> inline
NDTree<T, D>::~NDTree()
{
    delete _root;
}

//! 代入演算子
/*!
  \param tree	コピー元の2^D分木
  \return	この2^D分木
*/
template <class T, u_int D> inline NDTree<T, D>&
NDTree<T, D>::operator =(const NDTree& tree)
{
    if (this != &tree)
    {
	_org  = tree._org;
	_len0 = tree._len0;
	_root = (tree._root ? tree._root->clone() : 0);
    }
    
    return *this;
}

//! この2^D分木のrootセルの原点位置を返す．
/*!
  \return	rootセルの原点位置
*/
template <class T, u_int D> inline const typename NDTree<T, D>::position_type&
NDTree<T, D>::origin() const
{
    return _org;
}

//! この2^D分木のrootセルの一辺の長さを返す．
/*!
  \return	rootセルの一辺の長さ
*/
template <class T, u_int D> inline u_int
NDTree<T, D>::length0() const
{
    return _len0;
}

//! この2^D分木中の葉の数を返す．
/*!
  \return	葉の数
*/
template <class T, u_int D> inline u_int
NDTree<T, D>::size() const
{
    return (_root ? _root->size() : 0);
}

//! この2^D分木が空であるか調べる．
/*!
  \return	空であればtrue, そうでなければfalse
*/
template <class T, u_int D> inline bool
NDTree<T, D>::empty() const
{
    return !_root;
}

//! この2^D分木を空にする．
template <class T, u_int D> inline void
NDTree<T, D>::clear()
{
    delete _root;
    _root = 0;
    _len0 = 0;
}

//! D次元空間中の指定された位置における値を探す．
/*!
  \param pos	D次元空間中の位置
  \return	posで指定された位置に葉が存在すればその値へのポインタ
		を返す．存在しなければ0を返す．
*/
template <class T, u_int D> inline typename NDTree<T, D>::const_pointer
NDTree<T, D>::find(const position_type& pos) const
{
    if (!_root || out_of_range(pos))
	return 0;
    
    return _root->find(position_type(pos) -= _org, _len0);
}

//! D次元空間中の指定された位置に値を格納する．
/*!
  指定された位置に葉がなければ新たに葉が作られ，そこに値が格納される．
  \param pos	D次元空間中の位置
  \param val	格納する値
*/
template <class T, u_int D> void
NDTree<T, D>::insert(const position_type& pos, const_reference val)
{
    if (_root)
    {
	for (;;)
	{
	    bool	ascend = false;
	    u_int	idx = 0;
	    for (u_int d = 0; d < Dim; ++d)
		if (pos[d] < _org[d])		// 負方向に逸脱なら...
		{
		    ascend = true;		// _rootの昇階が必要
		    _org[d] -= int(_len0);	// 原点を負方向に移動
		    idx |= (1 << d);
		}
		else if (pos[d] >= _org[d] + int(_len0))  // 正方向に逸脱なら...
		    ascend = true;		// _rootの昇階が必要
	    if (!ascend)			// _rootの昇階が不要ならば
		break;				// 直ちに脱出

	    _len0 <<= 1;			// _rootのセル長を2倍にする．

	    _root = Branch::ascend(_root, idx);	// _rootを昇階
	}
	
	_root->insert(position_type(pos) -= _org, val, _len0);
    }
    else
    {
	_len0 = 1;
	_org  = pos;

	_root = Node::create(position_type(pos) -= _org, val, _len0);
    }
}

//! D次元空間中の指定された位置の葉を消去する．
/*!
  \param pos	D次元空間中の位置
*/
template <class T, u_int D> void
NDTree<T, D>::erase(const position_type& pos)
{
    if (!_root || out_of_range(pos))
	return;
    
  // posの位置にある葉を消す．	
    _root = _root->erase(position_type(pos) -= _org, _len0);

    if (!_root)					// 空ならば...
    {
	_len0 = 0;				// _rootのセル長を0にして
	return;					// 直ちにリターン
    }

  // _rootが子を1つだけ持つ枝ならば，子を_rootに付け替えることにより降階する．
    for (Branch* b; b = _root->branch(); )	// _rootが枝ならば...
    {
	u_int	idx;
	Node*	child = b->descend(idx);	// 1段分の降階を試みる．
	if (!child)				// 降階できなければ...
	    break;				// 直ちに脱出

	_root = child;				// 子を_rootに付け替える
		
	_len0 >>= 1;				// _rootのセル長を半分にする．
	position_type	dp;			// 古いrootから見た子の相対位置
	for (u_int d = 0; d < Dim; ++d)			// 子のindexを
	    dp[d] = (idx & (1 << d) ? _len0 : 0);	// 相対位置に変換
	_org += dp;				// 原点を子の位置に移す．
    }
}

//! 2^D分木の先頭要素を指す反復子を返す．
/*!
  \return	先頭要素を指す反復子
*/
template <class T, u_int D> inline typename NDTree<T, D>::iterator
NDTree<T, D>::begin()
{
    return iterator(*this);
}

//! 2^D分木の先頭要素を指す定数反復子を返す．
/*!
  \return	先頭要素を指す定数反復子
*/
template <class T, u_int D> inline typename NDTree<T, D>::const_iterator
NDTree<T, D>::begin() const
{
    return const_iterator(*this);
}

//! 2^D分木の末尾を指す反復子を返す．
/*!
  \return	末尾を指す反復子
*/
template <class T, u_int D> inline typename NDTree<T, D>::iterator
NDTree<T, D>::end()
{
    return iterator();
}

//! 2^D分木の末尾を指す定数反復子を返す．
/*!
  \return	末尾を指す定数反復子
*/
template <class T, u_int D> inline typename NDTree<T, D>::const_iterator
NDTree<T, D>::end() const
{
    return const_iterator();
}

//! 出力ストリームに2^D分木を書き出す(ASCII)．
/*!
  \param out	出力ストリーム
  \return	outで指定した出力ストリーム
*/
template <class T, u_int D> std::ostream&
NDTree<T, D>::put(std::ostream& out) const
{
    using namespace	std;
    
    for (const_iterator iter = begin(); iter != end(); ++iter)
	iter.position().put(out) << '\t' << iter.length() << '\t'
				 << *iter << endl;

    return out;
}
    
//! 入力ストリームから2^D分木を読み込む(ASCII)．
/*!
  \param in	入力ストリーム
  \return	inで指定した入力ストリーム
*/
template <class T, u_int D> std::istream&
NDTree<T, D>::get(std::istream& in)
{
    clear();					// 既存の全セルを廃棄

    for (position_type pos; in >> pos; )	// 葉の位置を読み込み
    {
	u_int		len;
	value_type	val;
	in >> len >> val;			// 葉のセル長と値を読み込む．
	insert(pos, val);			// 指定された位置に値を挿入
    }

    return in;
}

//! 出力ストリームに2^D分木の構造を書き出す(ASCII)．
/*!
  \param out	出力ストリーム
  \return	outで指定した出力ストリーム
*/
template <class T, u_int D> inline std::ostream&
NDTree<T, D>::print(std::ostream& out) const
{
    if (_root)
	_root->print(out, 0);

    return out;
}
    
template <class T, u_int D> bool
NDTree<T, D>::out_of_range(const position_type& pos) const
{
    for (u_int d = 0; d < Dim; ++d)
	if ((pos[d] < _org[d]) || (pos[d] >= _org[d] + int(_len0)))
	    return true;
    return false;
}

/************************************************************************
*  class NDTree<T, D>::Iterator<S>					*
************************************************************************/
//! 何も指さない2^D分木のための反復子を作る．
template <class T, u_int D> template <class S> inline
NDTree<T, D>::Iterator<S>::Iterator()
    :_org(), _leaf(0), _dp(), _len(0)
{
}

//! 2^D分木のための反復子を作る．
/*!
  反復子は2^D分木の先頭要素を指すように初期化される．
  \param tree	2^D分木
*/
template <class T, u_int D> template <class S> inline
NDTree<T, D>::Iterator<S>::Iterator(const NDTree& tree)
    :_org(), _leaf(0), _dp(), _len(0)
{
    if (tree._root)
    {
	_org = tree._org;
	_dp  = 0;
	_len = tree._len0;
	_fringe.push(NodeInfo(tree._root, _dp, _len));
	++*this;			// 最初の葉を指すまで反復子を進める
    }
}

//! この反復子が指す値が収められている葉の位置を返す．
/*!
  \return	葉の位置
*/
template <class T, u_int D> template <class S>
inline typename NDTree<T, D>::position_type
NDTree<T, D>::Iterator<S>::position() const
{
    return position_type(_org) += _dp;
}

//! この反復子が指す値が収められている葉のセル長を返す．
/*!
  \return	葉のセル長
*/
template <class T, u_int D> template <class S> inline u_int
NDTree<T, D>::Iterator<S>::length() const
{
    return _len;
}

//! この反復子が指す値への参照を返す．
/*!
  \return	値への参照
*/
template <class T, u_int D> template <class S> inline S&
NDTree<T, D>::Iterator<S>::operator *() const
{
    return _leaf->_val;
}

//! この反復子が指す値へのポインタを返す．
/*!
  \return	値へのポインタ
*/
template <class T, u_int D> template <class S> inline S*
NDTree<T, D>::Iterator<S>::operator ->() const
{
    return &(operator *());
}

//! この反復子をインクリメントする(前置)．
/*!
  \return	インクリメント後のこの反復子
*/
template <class T, u_int D> template <class S> NDTree<T, D>::Iterator<S>&
NDTree<T, D>::Iterator<S>::operator ++()
{
  // 1. _leaf != 0			反復子が空でない葉を指している状態
  // 2. _leaf == 0 && _fringe.emtpy()	反復子が末尾に到達している状態
  // 3. _leaf == 0 && !_fringe.empty()	下記whileループの途中
    
    _leaf = 0;					// 過去の葉を捨てる．
    
    while (!_fringe.empty())
    {
	Node*	node = _fringe.top().node;	// 先頭ノード，
	_dp  = _fringe.top().dp;		// そのセル位置
	_len = _fringe.top().len;		// およびセル長を
	_fringe.pop();				// popする．

	if (_leaf = node->leaf())		// popしたノードが葉であれば...
	    break;				// インクリメント後の位置に到達
	
	Branch* branch = node->branch();	// popしたノードは枝
	_len >>= 1;				// 子のレベルに1段降下
	    
	for (int i = NChildren; --i >= 0; )
	    if (branch->_children[i])		// 空でない子を逆順にpush
		_fringe.push(NodeInfo(branch->_children[i],
				      new_dp(i), _len));
    }

    return *this;
}

//! この反復子をインクリメントする(後置)．
/*!
  \return	インクリメント前の反復子
*/
template <class T, u_int D> template <class S> inline NDTree<T, D>::Iterator<S>
NDTree<T, D>::Iterator<S>::operator ++(int)
{
    Iterator	tmp = *this;
    ++*this;
    return tmp;
}

//! この反復子と与えられた反復子が同一の要素を指しているか調べる．
/*!
  \param iter	比較対象の反復子
  \return	同一の要素を指していればtrue, そうでなければfalse
*/
template <class T, u_int D> template <class S> inline bool
NDTree<T, D>::Iterator<S>::operator ==(const Iterator& iter) const
{
    return _leaf == iter._leaf;
}

//! この反復子と与えられた反復子が異なる要素を指しているか調べる．
/*!
  \param iter	比較対象の反復子
  \return	異なる要素を指していればtrue, そうでなければfalse
*/
template <class T, u_int D> template <class S> inline bool
NDTree<T, D>::Iterator<S>::operator !=(const Iterator& iter) const
{
    return !operator ==(iter);
}

template <class T, u_int D> template <class S>
typename NDTree<T, D>::position_type
NDTree<T, D>::Iterator<S>::new_dp(u_int idx) const
{
    position_type	dp = _dp;
    for (u_int d = 0; d < Dim; ++d)
	if (idx & (1 << d))
	    dp[d] |= _len;

    return dp;
}

/************************************************************************
*  class NDTree<T, D>::Node						*
************************************************************************/
template <class T, u_int D>
NDTree<T, D>::Node::~Node()
{
}

template <class T, u_int D> inline typename NDTree<T, D>::Node*
NDTree<T, D>::Node::create(const position_type& dp,
			   const_reference val, u_int len)
{
    if (len != 1)
    {
	Branch*	b = new Branch;
	b->insert(dp, val, len);	// 新たに生成したノードに挿入する．
	return b;
    }
    else
	return new Leaf(val);
}

/************************************************************************
*  class NDTree<T, D>::Branch						*
************************************************************************/
template <class T, u_int D>
NDTree<T, D>::Branch::Branch()
{
    for (u_int i = 0; i < NChildren; ++i)
	_children[i] = 0;
}

template <class T, u_int D>
NDTree<T, D>::Branch::~Branch()
{
    for (u_int i = 0; i < NChildren; ++i)
	delete _children[i];
}

template <class T, u_int D> typename NDTree<T, D>::Node*
NDTree<T, D>::Branch::clone() const
{
    Branch*	b = new Branch;
    for (u_int i = 0; i < NChildren; ++i)
	if (_children[i])
	    b->_children[i] = _children[i]->clone();
    return b;
}
    
template <class T, u_int D> u_int
NDTree<T, D>::Branch::size() const
{
    u_int	n = 0;
    for (u_int i = 0; i < NChildren; ++i)
	if (_children[i])
	    n += _children[i]->size();
    return n;
}

template <class T, u_int D> typename NDTree<T, D>::const_pointer
NDTree<T, D>::Branch::find(const position_type& dp, u_int len) const
{
    len >>= 1;						// 1つ下のレベルへ
    const Node*	child = _children[child_idx(dp, len)];	// 子
    
    return (child ? child->find(dp, len) : 0);
}

template <class T, u_int D> void
NDTree<T, D>::Branch::insert(const position_type& dp,
			     const_reference val, u_int len)
{
    len >>= 1;						// 1つ下のレベルへ
    Node*&	child = _children[child_idx(dp, len)];	// 子

    if (child)						// 子があれば...
	child->insert(dp, val, len);			// 既存の子に挿入
    else						// 子がなければ...
	child = create(dp, val, len);			// 新たに子を作って挿入
}

template <class T, u_int D> typename NDTree<T, D>::Node*
NDTree<T, D>::Branch::erase(const position_type& dp, u_int len)
{
    len >>= 1;						// 1つ下のレベルへ
    Node*&	child = _children[child_idx(dp, len)];	// 子

    if (!child)						// 子がなければ...
	return this;					// この枝自身を返す
    child = child->erase(dp, len);			// 子孫から消去

    for (u_int i = 0; i < NChildren; ++i)
	if (_children[i])	// 子が1つでも残っていれば...
	    return this;	// この枝自身を返す

    delete this;		// 全ての子が空なので，この枝自身を解放
    return 0;			// 0を返す
}
    
template <class T, u_int D> typename NDTree<T, D>::Branch*
NDTree<T, D>::Branch::branch()
{
    return this;
}
    
template <class T, u_int D> typename NDTree<T, D>::Leaf*
NDTree<T, D>::Branch::leaf()
{
    return 0;
}
    
template <class T, u_int D> void
NDTree<T, D>::Branch::print(std::ostream& out, u_int nindents) const
{
    using namespace	std;

    out << endl;
    for (u_int i = 0; i < NChildren; ++i)
    {
	for (u_int n = 0; n < nindents; ++n)
	    out << ' ';
	out << '[' << i << "]: ";
	if (_children[i])
	    _children[i]->print(out, nindents + 2);
	else
	    out << "NULL"<< endl;
    }
}
    
template <class T, u_int D> typename NDTree<T, D>::Node*
NDTree<T, D>::Branch::ascend(Node* node, u_int idx)
{
    Branch*	b = new Branch;
    b->_children[idx] = node;

    return b;
}
    
template <class T, u_int D> typename NDTree<T, D>::Node*
NDTree<T, D>::Branch::descend(u_int& idx)
{
    u_int	nchildren = 0;
    for (u_int i = 0; i < NChildren; ++i)
	if (_children[i])
	{
	    ++nchildren;		// 子の数と
	    idx = i;			// そのindexを調べる．
	}
    if (nchildren != 1)			// 子が1つでなければ...
	return 0;			// 階層は減らせないので脱出

    Node*	root = _children[idx];	// この子を新たなrootにして1階層削減
    
    _children[idx] = 0;			// 子へのリンクを消してから
    delete this;			// 自身を解放する．

    return root;			// 新しいrootを返す．
}
    
template <class T, u_int D> u_int
NDTree<T, D>::Branch::child_idx(const position_type& dp, u_int len)
{
    u_int	i = 0;
    for (int d = 0; d < Dim; ++d)
	if (dp[d] & len)
	    i |= (1 << d);

    return i;
}
    
/************************************************************************
*  class NDTree<T, D>::Leaf						*
************************************************************************/
template <class T, u_int D> inline
NDTree<T, D>::Leaf::Leaf(const_reference val)
    :_val(val)
{
}

template <class T, u_int D>
NDTree<T, D>::Leaf::~Leaf()
{
}

template <class T, u_int D> typename NDTree<T, D>::Node*
NDTree<T, D>::Leaf::clone() const
{
    return new Leaf(*this);
}
    
template <class T, u_int D> u_int
NDTree<T, D>::Leaf::size() const
{
    return 1;
}
    
template <class T, u_int D> typename NDTree<T, D>::const_pointer
NDTree<T, D>::Leaf::find(const position_type&, u_int len) const
{
    if (len != 1)
	throw std::logic_error("NDTree<T, D>::Leaf::find: non-zero \'len\'!");
    return &_val;
}

template <class T, u_int D> void
NDTree<T, D>::Leaf::insert(const position_type& pos,
			   const_reference val, u_int len)
{
    if (len != 1)
	throw std::logic_error("NDTree<T, D>::Leaf::insert: non-zero \'len\'!");
    _val = val;
}

template <class T, u_int D> typename NDTree<T, D>::Node*
NDTree<T, D>::Leaf::erase(const position_type&, u_int len)
{
    if (len != 1)
	throw std::logic_error("NDTree<T, D>::Leaf::erase: non-zero \'len\'!");
    delete this;				// この葉自身を解放
    return 0;
}

template <class T, u_int D> typename NDTree<T, D>::Branch*
NDTree<T, D>::Leaf::branch()
{
    return 0;
}
    
template <class T, u_int D> typename NDTree<T, D>::Leaf*
NDTree<T, D>::Leaf::leaf()
{
    return this;
}
    
template <class T, u_int D> void
NDTree<T, D>::Leaf::print(std::ostream& out, u_int nindents) const
{
    out << _val << std::endl;
}

/************************************************************************
*  global fucntions							*
************************************************************************/
//! 入力ストリームから2^D分木を読み込む(ASCII)．
/*!
  \param in	入力ストリーム
  \param tree	2^D分木の読み込み先
  \return	inで指定した入力ストリーム
*/
template <class T, u_int D> inline std::istream&
operator >>(std::istream& in, NDTree<T, D>& tree)
{
    return tree.get(in);
}

//! 出力ストリームへ2^D分木を書き出す(ASCII)．
/*!
  \param out	出力ストリーム
  \param tree	書き出す2^D分木
  \return	outで指定した出力ストリーム
*/
template <class T, u_int D> inline std::ostream&
operator <<(std::ostream& out, const NDTree<T, D>& tree)
{
    return tree.put(out);
}

}
#endif
