/*
 *  $Id: TUCollection++.sa.cc,v 1.2 2002-07-25 02:38:01 ueshiba Exp $
 */
#include "TU/Collection++.h"

namespace TU
{
const Object::Desc	ObjTreeBase::Node::_desc(id_TreeNode, 0,
						ObjTreeBase::Node::newObject,
						&ObjTreeBase::Node::_p,
						&ObjTreeBase::Node::_left,
						&ObjTreeBase::Node::_right,
						0);
const Object::Desc	ObjTreeBase      ::_desc(id_TreeBase, 0,
						ObjTreeBase::newObject,
						&ObjTreeBase::_root,
						0);
}
