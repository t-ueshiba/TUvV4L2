/*
 *  $Id: Mesh++.cc,v 1.1.1.1 2002-07-25 02:14:16 ueshiba Exp $
 */
#include <float.h>
#include <map>
#include <set>
#include "TU/Mesh++.h"
#include <stdexcept>

namespace TU
{
/************************************************************************
*  class Mesh<V, E, F, M>						*
************************************************************************/
/*
 *  "kill" procedure for "Triangular" meshes.
 */
template <class V, class E, class F, u_int M> E
Mesh<V, E, F, M>::kill(Edge& edge)
{
    using namespace	std;
    
  // Check whether valence of neighboring vertices is enough.
    Edge	edgeNC(edge.next().conj()),
		edgeCPC(edge.conj().prev().conj()),
		edgeCNC(edge.conj().next().conj());
    if (edge.valence() + edgeCPC.valence() <= 6 ||
	edgeNC.valence() <= 3 || edgeCNC.valence() <= 3)
    {
	cerr << "TU::Mesh<V, E, F, 3u>::kill(): Too small valence!" << endl;
	return edge;
    }

    Edge	edgeCN(edge.conj().next());
    V		*vn = &edge.next().v();
    for (Edge tmp(edge.prev().conj()); ~(--tmp) != edgeCN; )
	for (Edge tmp1(tmp); --(~tmp1) != tmp; )
	    if (&tmp1.v() == vn)
	    {
		cerr << "TU::Mesh<V, E, F, 3u>::kill(): "
		     << "Pre-requisits for topology are not met!"
		     << endl;
		return edge;
	    }
    
  // Replace vertex pointers of the faces around edge.v().
    V	*v = &edge.v();
    edge.replaceVertex(vn, edge);
    v->~V();
    _v.free(v);

  // Collapse edgePC and edgeCPC.    
    F	*f = &edge.f(), *fC = &edge.conj().f();
    ~(--edge);
    edge   .pair(edgeNC);
    edgeCPC.pair(edgeCNC);
    f->~F();
    _f.free(f);
    fC->~F();
    _f.free(fC);

    return edgeCPC;
}

/*
 *  "make" procedure for "Triangular" meshes.
 */
template <class V, class E, class F, u_int M> E
Mesh<V, E, F, M>::make(const Edge& edge0, const Edge& edge1, const V& v)
{
    if (!edge0.commonVertex(edge1))
	throw std::domain_error("TU::Mesh<V, E, F, 3u>::make(): Given two edges have no common vertex!");
    if (edge0 == edge1)
	throw std::domain_error("TU::Mesh<V, E, F, 3u>::make(): Given two edges are identical!");

  // Make a new vertex.
    V*	vnew = new(_v.alloc()) V(v);

  // Make new faces.
    V*	vp[3];
    vp[0] = vnew;
    vp[1] = &edge0.v();
    vp[2] = &edge0.next().v();
    F*	f = new(_f.alloc()) F(vp);
    vp[0] = &edge1.v();
    vp[1] = vnew;
    vp[2] = &edge1.next().v();
    F*	fC = new(_f.alloc()) F(vp);

  // Keep edge0.conj() before replacing the vertex of edge0.    
    Edge	edge0C(edge0.conj()), edge1C(edge1.conj());

  // Replace vertices of edges in [edge0, edge1).
    edge0.replaceVertex(vnew, edge1);

  // Build winged-edge structure.
    Edge	edge(*f), edgeC(*fC);
    edge.pair(edgeC);
    (--edge ).pair(edge0);
    (--edge ).pair(edge0C);
    (--edgeC).pair(edge1);
    (--edgeC).pair(edge1C);
    
    return --edge;
}

/*
 *  "swap" procedure for "Triangular" meshes.
 */
template <class V, class E, class F, u_int M> E
Mesh<V, E, F, M>::swap(const Edge& edge)
{
    using namespace	std;
    
  // Check whether valence of neighboring vertices is enough.
    Edge	edgePC(edge.prev().conj()),
		edgeCPC(edge.conj().prev().conj());
    if (edgePC.valence() <= 3 || edgeCPC.valence() <= 3)
    {
	cerr << "TU::Mesh<V, E, F, 3u>::swap(): Too small valence!" << endl;
	return edge;
    }
    
  // Replace vertex pointers of the faces around edge.v() and edge.conj().v().
    Edge	edgeC(edge.conj()),
		edgeNC(edge.next().conj()),
		edgeCNC(edge.conj().next().conj());
    edge.replaceVertex(&edgeCNC.v());
    edge.next().replaceVertex(&edgeNC.v());
    edge.prev().replaceVertex(&edgePC.v());
    edgeC.replaceVertex(&edgeNC.v());
    edgeC.next().replaceVertex(&edgeCNC.v());
    edgeC.prev().replaceVertex(&edgeCPC.v());

  // Swap edges.
    edge.next().pair(edgePC);
    edge.prev().pair(edgeCNC);
    edgeC.next().pair(edgeCPC);
    edgeC.prev().pair(edgeNC);

    return edge;
}

template <class V, class E, class F, u_int M> std::istream&
Mesh<V, E, F, M>::get(std::istream& in)
{
    Topology::setAllocator(_v);
    Array<Topology>	topology;
    in >> topology;			// Read vertices.
    in.clear();				// Clear EOF flag.

    char	c;
    while (in >> c && c == 'F')		// Read faces.
    {
	char	token[64];
	int	fnum;
	in.width(sizeof(token));
	in >> token >> fnum;		// Skip face number.

	V*	v[M];
	int	e, vnum[M];
	for (e = 0; e < M; ++e)
	{
	    in >> vnum[e];		// Read vertex numbers of this face.
	    v[e] = topology[--vnum[e]].v();
	}
#ifndef TUMeshPP_DEBUG
	F*	f = new(_f.alloc()) F(v);	// Make a new face.
#else
	F*	f = new(_f.alloc()) F(v, fnum);	// Make a new face.
#endif
	for (e = 0; e < M; ++e)		// Associate each vertex with the face.
	    topology[vnum[e]].push_front(*new FaceNode(f));
    }
    
    for (int n = 0; n < topology.dim(); ++n)
    {
	topology[n].pair();
	while (!topology[n].empty())
	    delete &topology[n].pop_front();
    }
    
    in.putback(c);

    return in;
}

template <class V, class E, class F, u_int M> std::ostream&
Mesh<V, E, F, M>::put(std::ostream& out) const
{
    std::map<V*, int>	dict;
    int					vnum = 1;
    for (Allocator<V>::Enumerator vertices(_v); vertices; ++vertices)
    {
	dict[vertices] = vnum;
	out << "Vertex " << vnum++ << ' ' << *vertices;
    }
    int		fnum = 1;
    for (Allocator<F>::Enumerator faces(_f); faces; ++faces)
    {
	out << "Face " << fnum++;
	for (int e = 0; e < M; ++e)
	    out << ' ' << dict[&(faces->v(e))];
	out << std::endl;
    }
    
    return out;
}

template <class V, class E, class F, u_int M> 
typename Mesh<V, E, F, M>::BoundingBox
Mesh<V, E, F, M>::boundingBox() const
{
    BoundingBox	bbox;
    bbox.xmin = bbox.ymin = bbox.zmin = FLT_MAX;
    bbox.xmax = bbox.ymax = bbox.zmax = FLT_MIN;
    
    for (Allocator<F>::Enumerator faces(_f); faces; ++faces)
	for (int e = 0; e < M; ++e)
	{
	    if (faces->v(e)[0] <= bbox.xmin)
		bbox.xmin = faces->v(e)[0];
	    if (faces->v(e)[0] >= bbox.xmax)
		bbox.xmax = faces->v(e)[0];
	    if (faces->v(e)[1] <= bbox.ymin)
		bbox.ymin = faces->v(e)[1];
	    if (faces->v(e)[1] >= bbox.ymax)
		bbox.ymax = faces->v(e)[1];
	    if (faces->v(e)[2] <= bbox.zmin)
		bbox.zmin = faces->v(e)[2];
	    if (faces->v(e)[2] >= bbox.zmax)
		bbox.zmax = faces->v(e)[2];
	}
    
    return bbox;
}

template <class V, class E, class F, u_int M> void
Mesh<V, E, F, M>::clean()
{
    using namespace	std;
    
    set<V*>	verticesUsed;
    for (Allocator<F>::Enumerator faces(_f); faces; ++faces)
	for (int e = 0; e < M; ++e)
	    verticesUsed.insert(&(faces->v(e)));
    for (Allocator<V>::Enumerator vertices(_v); vertices; ++vertices)
    {
	set<V*, less<V*> >::iterator	p = verticesUsed.find(vertices);
	if (p == verticesUsed.end())
	{
	    vertices->~V();
	    _v.free(vertices);
	}
    }
}

/************************************************************************
*  class Mesh<V, E, F, M>::BoundingBox					*
************************************************************************/
template <class V, class E, class F, u_int M> float
Mesh<V, E, F, M>::BoundingBox::maxlength() const
{
    if (xlength() > ylength())
	if (xlength() > zlength())
	    return xlength();
	else
	    return zlength();
    else
	if (ylength() > zlength())
	    return ylength();
	else
	    return zlength();
}

template <class V, class E, class F, u_int M> float
Mesh<V, E, F, M>::BoundingBox::minlength() const
{
    if (xlength() < ylength())
	if (xlength() < zlength())
	    return xlength();
	else
	    return zlength();
    else
	if (ylength() < zlength())
	    return ylength();
	else
	    return zlength();
}

/************************************************************************
*  class Mesh<V, E, F, M>::Face						*
************************************************************************/
template <class V, class E, class F, u_int M>
#ifndef TUMeshPP_DEBUG
Mesh<V, E, F, M>::Face::Face(V* v[])
#else
Mesh<V, E, F, M>::Face::Face(V* v[], int fn)
    :fnum(fn)
#endif
{
    for (u_int e = 0; e < M; ++e)
    {
	_f[e] = 0;
	_v[e] = v[e];
    }
}

template <class V, class E, class F, u_int M> Coordinate<float, 3u>
Mesh<V, E, F, M>::Face::centroid() const
{
    Coordinate<float, 3u>	c;
    for (u_int i = 0; i < M; ++i)
	c += *_v[i];
    c /= M;
    return c;
}

/************************************************************************
*  class Mesh<V, E, F, M>::Edge						*
************************************************************************/
template <class V, class E, class F, u_int M> u_int
Mesh<V, E, F, M>::Edge::valence() const
{
    u_int	n = 0;
    Edge	edge(*this);
    do
    {
	++n;
    } while (~(--edge) != *this);

    return n;
}

template <class V, class E, class F, u_int M> int
Mesh<V, E, F, M>::Edge::commonVertex(const Edge& edge) const
{
    Edge	tmp(*this);
    do
    {
	if (tmp == edge)
	    return 1;
    } while (~(--tmp) != *this);
    
    return 0;
}

template <class V, class E, class F, u_int M> E
Mesh<V, E, F, M>::Edge::conj() const
{
    V*	vn = &next().v();
    E	edge(*(_f->_f[_e]));
    while (&edge.v() != vn)
	++edge;
    return edge;
}

template <class V, class E, class F, u_int M> void
Mesh<V, E, F, M>::Edge::replaceVertex(V* v, const Edge& edgeE) const
{
    Edge	edgePC(prev().conj());
    if (edgePC != edgeE)
	edgePC.replaceVertex(v, edgeE);
    _f->_v[_e] = v;
}

/************************************************************************
*  class Mesh<V, E, F, M>::Topology					*
************************************************************************/
template <class V, class E, class F, u_int M> Allocator<V>*
Mesh<V, E, F, M>::Topology::_a = 0;

template <class V, class E, class F, u_int M> void
Mesh<V, E, F, M>::Topology::pair() const
{
    for (List<FaceNode>::ConstIterator iter = begin(); iter != end(); ++iter)
    {
	Edge	edge(iter->f());
	while (&edge.v() != _v)
	    ++edge;
	V*	vn = &edge.next().v();
	for (List<FaceNode>::ConstIterator iter1(iter); ++iter1 != end(); )
	{
	    Edge	edgeF0(iter1->f()), edgeF(edgeF0);
	    do
	    {
		if (&edgeF.v() == vn)
		{
		    edge.pair(edgeF);
		    goto done;
		}
	    } while (++edgeF != edgeF0);
	}
      done:
	continue;
    }
}

template <class V, class E, class F, u_int M> std::istream&
Mesh<V, E, F, M>::Topology::get(std::istream& in)
{
    char	c;
    if (in >> c)
	if (c == 'V')
	{
	    char	token[64];
	    _v = new(_a->alloc()) V();
	    in.width(sizeof(token));
	    in >> token >> token >> *_v;
	}
	else
	{
	    in.putback(c);
	    in.clear(std::ios::failbit|in.rdstate());
	}

    return in;
}
 
}
