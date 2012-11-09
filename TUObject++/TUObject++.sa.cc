/*
 *  $Id$
 */

#include "Object++_.h"

namespace TU
{
PtrBase*		PtrBase::_root = 0;	// root of the all objects

Page::Root		Page::_root;		// root of page list
Page::Cell		Page::Cell::_head[];

u_int			Object::Desc::_ndescs = 0;
Object::Desc::Map*	Object::Desc::_map = 0;
SaveMap::Map		SaveMap::_map;
u_long			SaveMap::_maxID = 0;	// maxID of save table
RestoreMap::Map		RestoreMap::_map;
u_long			RestoreMap::_maxID = 0;	// maxID of restore table
CopyMap::Map		CopyMap::_map;
}
