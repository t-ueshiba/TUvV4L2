/*
 *  $Id: Object++_.h,v 1.1.1.1 2002-07-25 02:14:15 ueshiba Exp $
 */
#include "TU/Object++.h"

namespace TU
{
/************************************************************************
*  class Page & Page::Cell:	memory page				*
************************************************************************/
/*!
  pageを表すクラス．pageとは，システムがheap領域から確保して自分の
  管理下に置き，ユーザからの要求に応じて貸し出すためのメモリ領域である．
  GCを行ってもユーザからの要求に応えられない場合は，新たなpageが確保
  される．
*/  
class Page
{
  public:
    typedef double		Block;

  /*!
    cellを表すクラス．ユーザからの1回の要求毎に1つのcellがpageから切り取られて
    ユーザに貸し出される．
  */  
    class Cell : private ObjectHeader
    {
	enum			{TBLSIZ = 16};	// TBLSIZ = 4 or 10 or 16.
	
      public:
	Cell(u_int nb=0) :ObjectHeader(nb), _prv(this), _nxt(this)	{}

	static Cell*		find(u_int nblocks, bool addition=false);
	Cell*			forward() const
				{
				    return (Cell*)((Block*)this + _nb);
				}
	u_int			add();
	Cell*			detach();
	Cell*			split(u_int nblocks);
	Cell*			merge()
				{
				    _nb += forward()->_nb; 
				    return this;
				}
	void*			clean();
	
      private:
	static Cell		_head[TBLSIZ];	// doubly-linked list heads.

	Cell*			_prv;		// doubly-linked to other node.
	Cell*			_nxt;		// ibid.

	friend class		Page;		// allow access to header
    };

    class Root
    {
      public:
	Root()	:_p(0)				{}
	~Root()					{while (_p != 0) delete _p;}
	
      			operator Page*() const 	{return _p;}
      	Root&		operator = (Page* page)	{_p = page; return *this;}
      	Page*		operator ->()	 const	{return _p;}
	
      private:
	Page*		_p;
    };
    
  private:
    enum		{NBLOCKS = (1 << Cell::TBLSIZ)};
    
  public:
    Page()						;
    ~Page()						{_root = _root->_nxt;}
    static u_int	sweep()				;
    static u_int	nbytes2nblocks(size_t nbytes)
			{ // must have enough size for a Cell.
			    size_t	nb = (nbytes > sizeof(Cell) ?
					      nbytes : sizeof(Cell));
			    u_int nblocks = (nb-1) / sizeof(Block) + 1;
			    return (nblocks <= NBLOCKS ? nblocks : 0);
			}

  private:
    static Root		_root;			// root of memory page list.

    Block		_block[NBLOCKS];	// used as cells.
    Page* const		_nxt;

};

/************************************************************************
*  class SaveMap:	a map for registering objects already saved	*
************************************************************************/
const u_long	Eoc	 = ~0;			// End of Context
const u_long	NotFound = Eoc;
const u_long	Nil	 = NotFound - 1;	// pointer value of 0

class SaveMap
{
  public:
    typedef std::map<const Object*, u_long>	Map;

    static u_long	find(const Object* obj)
			{    
			    return (obj == 0 ? Nil :
				    obj->_sv ? _map[obj] : NotFound);
			}
    static u_long	insert(const Object* obj)
			{
			    const_cast<Object*>(obj)->_sv = 1;
			    _map[obj] = _maxID;
			    return _maxID++;
			}
    static void		reset()
			{
			    for (Map::iterator i  = _map.begin();
					       i != _map.end(); )
			    {
				const_cast<Object*>((*i).first)->_sv = 0;
				_map.erase(i++);
			    }
			    _maxID = 0;
			}

  private:
    static Map		_map;
    static u_long	_maxID;
};

/************************************************************************
*  class RestoreMap:	a map for registering objects already restored	*
************************************************************************/
class RestoreMap
{
  public:
    typedef std::map<u_long, Object*>	Map;

    static Object*	find(u_long id)
			{
			    return (id == Nil   ? 0 :
				    id < _maxID ? _map[id] : (Object*)NotFound);
			}
    static void		insert(Object* obj)
			{
			    _map[_maxID++] = obj;
			}
    static void		reset()
			{
			    _map.erase(_map.begin(), _map.end());
			    _maxID = 0;
			}

  private:
    static Map		_map;
    static u_long	_maxID;
};

/************************************************************************
*  class CopyMap:	a map for registering objects already copied	*
************************************************************************/
class CopyMap
{
  public:
    typedef std::map<const Object*, Object*>			Map;
    
    static Object*	find(const Object* obj)
			{    
			    return (obj == 0 ? 0 :
				    obj->_cp ? _map[obj] : (Object*)NotFound);
			}
    static void		insert(const Object* obj, Object* dst)
			{
			    const_cast<Object*>(obj)->_cp = 1;
			    _map[obj] = dst;
			}
    static void		reset()
			{
			    for (Map::iterator i  = _map.begin();
					       i != _map.end(); )
			    {
				const_cast<Object*>((*i).first)->_cp = 0;
				_map.erase(i++);
			    }
			}

  private:
    static Map		_map;
};
 
}
