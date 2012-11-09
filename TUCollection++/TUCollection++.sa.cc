/*
 *  $Id$
 */
#include "TU/Collection++.h"

namespace TU
{
const Object::Desc	ObjTreeBase::Node::_desc(id_TreeNode, 0,
						 ObjTreeBase::Node::newObject,
						 &ObjTreeBase::Node::_p,
						 &ObjTreeBase::Node::_left,
						 &ObjTreeBase::Node::_right,
						 MbrpEnd);
const Object::Desc	ObjTreeBase      ::_desc(id_TreeBase, 0,
						 ObjTreeBase::newObject,
						 &ObjTreeBase::_root,
						 MbrpEnd);
}
