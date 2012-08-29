/*
 *  $Id: TUBrep++.sa.cc,v 1.5 2012-08-29 21:16:44 ueshiba Exp $
 */
#include "TU/Brep/Brep++.h"

namespace TU
{
namespace Brep
{
const Object::Desc	Loop	::_desc(id_Loop, 0, 0,
					&Loop::_parent,
					&Loop::_brother,
					&Loop::_children,
					&Loop::_head,
					&Loop::_prop,
					MbrpEnd);
const Object::Desc	Face	::_desc(id_Face, id_Loop, Face::newObject,
					MbrpEnd);
const Object::Desc	Ring	::_desc(id_Ring, id_Loop, Ring::newObject,
					MbrpEnd);
const Object::Desc	Root	::_desc(id_Root, id_Face, Root::newObject,
					MbrpEnd);
const Object::Desc	HalfEdge::_desc(id_HalfEdge, 0,
					HalfEdge::newObject, 
					&HalfEdge::_parent,
					&HalfEdge::_prev,
					&HalfEdge::_next,
					&HalfEdge::_conj,
					&HalfEdge::_geo,
					&HalfEdge::_prop,
					MbrpEnd);
const Object::Desc	PointB::_desc(id_Point, 0, PointB::newObject, MbrpEnd);
#ifdef TUBrepPP_DEBUG
u_int			Loop::nLoops		= 0;
u_int			HalfEdge::nHalfEdges	= 0;
#endif
}
template <>
const Object::Desc	Cons<Brep::PointB>::_desc(id_Geometry, 0,
					    Cons<TU::Brep::PointB>::newObject,
					    &Cons<TU::Brep::PointB>::_ca,
					    &Cons<TU::Brep::PointB>::_cd,
					    MbrpEnd);
}

