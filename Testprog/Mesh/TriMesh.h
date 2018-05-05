/*
 *  $Id: edit.h,v 1.3 2011-01-30 23:39:46 ueshiba Exp $
 */
#include "TU/Mesh++.h"
#include "TU/Manip.h"
#include "TU/v/OglDC.h"

namespace TU
{
typedef Vector3f	TriVertex;
    
/************************************************************************
*  class TriFace							*
************************************************************************/
class TriFace : public Mesh<TriVertex, TriFace>::Face
{
  private:
    typedef Mesh<TriVertex, TriFace>	mesh_t;
    typedef mesh_t::Face		super;

  public:
    typedef mesh_t::viterator		viterator;
    typedef mesh_t::fiterator		fiterator;
    typedef mesh_t::Edge		Edge;
    
  public:
#ifndef TU_MESH_DEBUG
    TriFace(viterator v[])		:super(v), mark(0)		{}
#else
    TriFace(viterator v[], int fn)	:super(v, fn), mark(0)		{}
#endif
    Vector3f		normal()				const	;
    
  public:
    mutable u_int	mark;
};

inline Vector3f
TriFace::normal() const
{
    Vector3f	normal = (v(1) - v(0)) ^ (v(2) - v(0));
    return normalize(normal);
}

/************************************************************************
*  typedef TriMesh							*
************************************************************************/
typedef Mesh<TriVertex, TriFace>	TriMesh;
 
namespace v
{
/************************************************************************
*  drawing functions							*
************************************************************************/
OglDC&	operator <<(OglDC& dc, const TriMesh& mesh)			;
OglDC&	operator <<(OglDC& dc, const TriFace& face)			;
OglDC&	operator <<(OglDC& dc, const TriMesh::Edge& edge)		;
OglDC&	operator <<(OglDC& dc, const TriVertex& v)			;

OglDC&	drawColoredMeshFaces(OglDC& dc, const TriMesh& mesh)		;

OglDC&	draw(OglDC& dc, const TriMesh::Edge& edge)			;
OglDC&	erace(OglDC& dc, const TriMesh::Edge& edge)			;
}
}

