/*
 *  $Id: Object.cc,v 1.4 2012-08-29 21:17:03 ueshiba Exp $
 */
#include "Object++_.h"
#include <stdexcept>

namespace TU
{
/*
 *  PtrBase::mark()
 */
void
PtrBase::mark()
{
#ifdef TUObjectPP_DEBUG
    std::cerr << "\tPtrBase::mark\tmarking....\n";
#endif
    for (PtrBase* objp = _root; objp; objp = objp->_nxt)
	objp->_p->mark();
}

/*
 *  Object::mark(), new(), save(), eoc(), restore(), copy(), cpy()
 */
void
Object::mark() const
{
    if (null() || _gc)
	return;
    const_cast<Object*>(this)->_gc = 1;
    for (const Mbrp* p = desc().mbrp(); *p != 0; )
	(this->*(*p++))->mark();
}

void*
Object::operator new(size_t size)
{
    using namespace	std;
  /* 要求サイズをbyte単位からblock単位に変更する．nblocks * sizeof(Block)
     は size 以上であることはもちろん，メモリブロックをCellとして管理する
     ことから，sizeof(Cell) 以上でなければならない．*/
    const u_int	nblocks = Page::nbytes2nblocks(size);
    if (nblocks == 0)
	throw std::domain_error("TU::Object::operator new\tToo large memory requirement!!");
    Page::Cell*	cell;
    if ((cell = Page::Cell::find(nblocks)) == 0)
    {
#ifdef TUObjectPP_DEBUG
	cerr << "TU::Object::operator new\tGarbage collection!!" << endl;
#endif
        PtrBase::mark();
	u_int	garbage = Page::sweep();	  
#ifdef TUObjectPP_DEBUG
	cerr << "TU::Object::operator new\t" << garbage << " blocks collected."
	     << endl;
#endif
	if ((cell = Page::Cell::find(nblocks)) == 0)
	{
#ifdef TUObjectPP_DEBUG
	    cerr << "TU::Object::operator new\tGet new Page!!" << endl;
#endif
	    new Page;
	    if ((cell = Page::Cell::find(nblocks)) == 0)
		throw std::bad_alloc();
	}
#ifdef TUObjectPP_DEBUG
	cerr << endl;
#endif
    }
    cell->detach()->split(nblocks)->add();
    return cell->clean();
}

std::ostream&
Object::save(std::ostream& out) const
{ 
    u_long	objID;

    if ((objID = SaveMap::find(this)) != NotFound)	// already saved ?
	out.write((char*)&objID, sizeof(objID));
    else
    {
	objID = SaveMap::insert(this);			// get new objID for me
      	out.write((char*)&objID, sizeof(objID));
	u_short classID = desc().id();			// get my classID
      	out.write((char*)&classID, sizeof(classID));
	saveGuts(out);					// save data members
	for (const Mbrp* p = desc().mbrp(); *p != 0; )
	    (this->*(*p++))->save(out);			// save recursively
    }
    return out;
}

std::ostream&
eoc(std::ostream& out)					// end of context
{
    u_long	objID = Eoc;

    out.write((char*)&objID, sizeof(objID));
    SaveMap::reset();
    return out;
}

Object*
Object::restoreObject(std::istream& in)
{
    Ptr<Object>	obj;
    u_long		objID;

    if (in.read((char*)&objID, sizeof(objID)) == 0 || objID == Eoc)
	RestoreMap::reset();
    else if ((obj = RestoreMap::find(objID)) == (Object*)NotFound)
    {							// not read yet
	obj = 0;					// for GC.
	u_short	classID;
	in.read((char*)&classID, sizeof(classID));
	obj = Desc::newObject(classID);
	RestoreMap::insert(obj);
	obj->restoreGuts(in);				// restore data members
	for (const Mbrp* p = obj->desc().mbrp(); *p != 0; )
	    obj->*(*p++) = restoreObject(in);		// restore recursively
    }
    return obj;
}

Object*
Object::copyObject(u_int depth) const
{
    Ptr<Object>	obj = CopyMap::find(this);
    
    if (obj == (Object*)NotFound)
    {
	obj = 0;					// for GC
	obj = clone();
	CopyMap::insert(this, obj);
	for (const Mbrp* p = desc().mbrp(); *p != 0; ++p)
	    obj->*(*p) = (this->*(*p))->copyObject(depth + 1);
    }
    if (depth == 0)
	CopyMap::reset();
    return obj;
}
 
}
