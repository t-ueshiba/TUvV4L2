/*
 *  $Id: Page.cc,v 1.2 2002-07-25 02:38:02 ueshiba Exp $
 */
#include "TU/Object++_.h"
#include <stdexcept>

namespace TU
{
/************************************************************************
*  class Page::Cell:		memory cells assigned to the objects	*
************************************************************************/
//! 指定されたblock数以上の大きさを持つcellをfree listから探す
/*!
  \param nblocks	block数．
  \param addition	falseならば，指定されたblock数以上のcellがみつかる
			まで全てのfree listを探索する．trueならば，指定され
			たblock数を格納するのにふさわしいfree listのみを探索
			し，格納位置の直後のcellを返す．
  \return		みつかったcellを返す．みつからなければ0を返す．
*/
Page::Cell*
Page::Cell::find(u_int nblocks, bool addition)
{
  // Find i s.t. 2^i <= nblocks - 1 < 2^(i+1).
    u_int	i = 0;
    for (u_int n = nblocks - 1; n >>= 1; )
	++i;
    
  // Search for the smallest cell of size greater than nblocks.
    for (; i < TBLSIZ; ++i)
    {
	_head[i]._nb = nblocks;			// sentinel.
	Cell*	cell = _head[i]._nxt;
	while (cell->_nb < nblocks)
	    cell = cell->_nxt;
	if (addition || cell != &_head[i])
	    return cell;
    }
    return 0;
}

//! 自身をfree listに格納する
/*!
  各free listの中でcellはその大きさ(block数)の昇順に格納される．this == 0
  も許され，もちろんこの場合は何もしない．
  \return	this != 0の場合は自身のblock数が返される．this == 0の場合
		は0が返される．
*/
u_int
Page::Cell::add()
{
    if (this)
    {
	Cell* cell = find(_nb, true);
	_nxt = cell;
	_prv = cell->_prv;
	_prv->_nxt = _nxt->_prv = this;
	_fr = 1;

	return _nb;
    }
    else
	return 0;
}
    
//! 自身をfree listから取り出す
/*!
  free listに格納されていることを表すフラグ_frが1の時のみ，実際の取り出し
  が起こり，もちろんこの時は_frが0に書き換えられる．
  \return	自分自身が返される．
*/
Page::Cell*
Page::Cell::detach()
{
    if (_fr)
    {
	_nxt->_prv = _prv;
	_prv->_nxt = _nxt;
	_fr = 0;
    }
    return this;
}
    
//! 自身を2つのcellに分割する
/*!
  自身のサイズを指定されたblock数に切り詰め，残りを新たなcellとして返す．
  もしも残りのサイズがcellそのもののサイズより小さい場合は分割は生じない．
  \param blocks	指定block数．
  \return	分割が成功すれば新たなcellが，失敗すれば0が，それぞれ返される．
*/
Page::Cell*
Page::Cell::split(u_int nblocks)
{
    const u_int	rest = _nb - nblocks;
    if (rest < nbytes2nblocks(sizeof(Cell)))
	return 0;
    _nb = nblocks;
    Cell* cell = new(forward()) Cell(rest);
    return cell;
}

//! ユーザ側に返されるメモリをきれいにする
/*!
  Object::newによって得られたメモリを用いてユーザがオブジェクトを構
  築する際に，そのオブジェクトの内部に他のオブジェクトへのポインタがあ
  ると，そのポインタの初期化が済んでいない時点でGCが生じた場合にポイン
  タにゴミの値が入っているためにmarkingが暴走する可能性がある．これを
  防ぐために，cellの中身全体を自身へのポインタで埋めておく．
*/
void*
Page::Cell::clean()
{
#ifdef TUObjectPP_DEBUG
    if (_gc || _fr)	// Must not be marked as in use or in freelist.
	throw std::domain_error("Page::Cell::clean: dirty cell!!");
#endif
    for (Cell **p = &_prv, **q = (Cell**)forward(); p < q; )
	*p++ = 0;
  /*    for (Cell **p = &_prv, **q = (Cell**)forward(); p < q; )
	*p++ = this;*/
    return this;
}

/************************************************************************
*  class Page:		memory page					*
************************************************************************/
//! 新たなメモリページを確保する
/*!
  ページを確保したら，自身をページリストに登録すると共に，中身のブロック
  をcellとしてfree listに格納する．
*/
Page::Page()
    :_nxt(_root)
{
    _root = this;			// Register myself to the page list.
    
    Cell*	cell = new(&_block[0]) Cell(NBLOCKS);
    cell->add();
}

//! 全てのメモリページをsweepして使用されていないcellを回収する
/*!
  \return	回収したblock数を返す．
*/
u_int
Page::sweep()
{    
    u_int	nblocks = 0;
    
    for (Page* page = _root; page; page = page->_nxt)	// for all pages...
    {
#ifdef TUObjectPP_DEBUG
	std::cerr << "\tPage::sweep\tsweeping...." << std::endl;
#endif
	Cell	*garbage = 0;
	for (Cell *cell = (Cell*)(&page->_block[0]),
		  *end  = (Cell*)(&page->_block[NBLOCKS]);
	     cell < end; cell = cell->forward())
	    if (cell->_gc)		// 使用中．
	    {
		cell->_gc = 0;			// マークをはずすだけ．
		nblocks += garbage->add();	// これまでに集めたゴミを格納．
		garbage = 0;
	    }
	    else			// free listにあるか又はdangling状態．
	    {
#ifdef TUObjectPP_DEBUG
		if (cell->_nb == 0)
		    std::cerr << "size 0 cell!!" << std::endl;
#endif
		cell->detach();			// free listにあれば取り出す．
	      // これまでに集めたゴミとマージする．
		garbage = (garbage ? garbage->merge() : cell);
	    }
	nblocks += garbage->add();
    }
    return nblocks;
}
 
}
