/*
 *  •½¬14-19”Ni“ÆjY‹Æ‹Zp‘‡Œ¤‹†Š ’˜ìŒ Š—L
 *  
 *  ‘nìÒFAÅr•v
 *
 *  –{ƒvƒƒOƒ‰ƒ€‚Íi“ÆjY‹Æ‹Zp‘‡Œ¤‹†Š‚ÌEˆõ‚Å‚ ‚éAÅr•v‚ª‘nì‚µC
 *  i“ÆjY‹Æ‹Zp‘‡Œ¤‹†Š‚ª’˜ìŒ ‚ğŠ—L‚·‚é”é–§î•ñ‚Å‚·D’˜ìŒ Š—L
 *  Ò‚É‚æ‚é‹–‰Â‚È‚µ‚É–{ƒvƒƒOƒ‰ƒ€‚ğg—pC•¡»C‰ü•ÏC‘æOÒ‚ÖŠJ¦‚·‚é
 *  “™‚Ìsˆ×‚ğ‹Ö~‚µ‚Ü‚·D
 *  
 *  ‚±‚ÌƒvƒƒOƒ‰ƒ€‚É‚æ‚Á‚Ä¶‚¶‚é‚¢‚©‚È‚é‘¹ŠQ‚É‘Î‚µ‚Ä‚àC’˜ìŒ Š—LÒ‚¨
 *  ‚æ‚Ñ‘nìÒ‚ÍÓ”C‚ğ•‰‚¢‚Ü‚¹‚ñB
 *
 *  Copyright 2002-2007.
 *  National Institute of Advanced Industrial Science and Technology (AIST)
 *
 *  Creator: Toshio UESHIBA
 *
 *  [AIST Confidential and all rights reserved.]
 *  This program is confidential. Any using, copying, changing or
 *  giving any information concerning with this program to others
 *  without permission by the copyright holder are strictly prohibited.
 *
 *  [No Warranty.]
 *  The copyright holder or the creator are not responsible for any
 *  damages caused by using this program.
 *  
 *  $Id: Mesh++.h,v 1.19 2010-01-31 23:35:07 ueshiba Exp $
 */
#ifndef __TUMeshPP_h
#define __TUMeshPP_h

#include <map>
#include <set>
#include "TU/Geometry++.h"
#include "TU/Allocator.h"

namespace TU
{
/************************************************************************
*  class Mesh<V, E, F, M>						*
************************************************************************/
template <class V, class E, class F, u_int M>
class Mesh		// Mesh with M-sided faces of type F, edges of type E
{			// and vertices of type V.
  public:
    class FaceNode : public List<FaceNode>::Node
    {
      public:
	FaceNode(F* f)	:_f(f)			{}
	
	F&		f()			const	{return *_f;}
	    
      private:
	F*		_f;
    };

  public:
    class Topology : public List<FaceNode>
    {
      public:
	Topology()		    :List<FaceNode>(), _v(0)	{}
	Topology(const Topology& t) :List<FaceNode>(), _v(t._v)	{}

	Topology&	operator =(const Topology& t)	{_v=t._v;return *this;}

	using		List<FaceNode>::begin;
	using		List<FaceNode>::end;
	
	static void	setAllocator(Allocator<V>& a)	{_a = &a;}
	V*		v()			const	{return _v;}
	void		pair()			const	;

      private:
	std::istream&	get(std::istream& in)		;
	friend std::istream&
	    operator >>(std::istream& in, Topology& t)	{return t.get(in);}

	V*			_v;			// Vertex.

	static Allocator<V>*	_a;
    };

    class Vertex
    {
      public:
	void*	operator new(size_t, void* p)		{return p;}
    };
    
    class Edge
    {
      public:
	Edge(F& face)	:_f(&face), _e(0)		{}

	F&	f()				const	{return *_f;}
	V&	v()				const	{return _f->v(_e);}
	bool	operator ==(const Edge& edge)	const	;
	bool	operator !=(const Edge& edge)	const	;
	bool	commonVertex(const Edge& edge)	const	;
	u_int	valence()			const	;
	Edge&	operator ++()				;
	Edge&	operator --()				;
	Edge&	operator ~()				;
	E	next()				const	;
	E	prev()				const	;
	E	conj()				const	;

      protected:
	u_int	e()				const	{return _e;}
	
      private:
	friend class 	Topology;	// Allow access to pair().
	friend class	Mesh;		// Allow access to relaceVertex().
	
	void	pair(const Edge& edge)			const	;
	void	replaceVertex(V* v, const Edge& edgeE)	const	;
	void	replaceVertex(V* v)			const	;
	
	F*	_f;		// parent face
	u_int	_e;		// my edge number
    };

    class Face
    {
      public:
#ifndef TUMeshPP_DEBUG
	Face(V* v[])					;
#else
	Face(V* v[], u_int fn)				;
#endif
	~Face()						{}

	F&		f(u_int e)		const	{return *_f[e];}
	V&		v(u_int e)		const	{return *_v[e];}
	Vector3f	centroid()		const	;
	
	void*		operator new(size_t, void* p)	{return p;}
	
      private:
	friend class	Edge;
    
	F*		_f[M];		// _f[e] : neighboring face of e
	V*		_v[M];		// _v[e] : starting vertex of e
#ifdef TUMeshPP_DEBUG
      public:
	const u_int	fnum;
#endif
    };

    class FaceEnumerator : public Allocator<F>::Enumerator
    {
      private:
	typedef typename Allocator<F>::Enumerator	Enumerator;

      public:
	FaceEnumerator(const Mesh& mesh)
	    :Enumerator(mesh._f)			{}
    };
    
    class VertexEnumerator : public Allocator<V>::Enumerator
    {
      private:
	typedef typename Allocator<V>::Enumerator	Enumerator;

      public:
	VertexEnumerator(const Mesh& mesh)
	    :Enumerator(mesh._v)			{}
    };
    
  public:
    Mesh(u_int nfacesPerPage)	:_f(nfacesPerPage)	{}

    E			kill(Edge& edge)		;
    E			make(const Edge& edge0,
			     const Edge& edge1,
			     const V& v)		;
    E			swap(const Edge& edge)		;
    BoundingBox<V>	boundingBox()		const	;
    void		clean()				;
    
  private:
    friend class	FaceEnumerator;
    friend class	VertexEnumerator;
    
    std::istream&	get(std::istream& in)		;
    std::ostream&	put(std::ostream& out)	const	;
    
    friend std::istream&
	operator >>(std::istream& in, Mesh& mesh)	{return mesh.get(in);}
    friend std::ostream&
	operator <<(std::ostream& out, const Mesh& mesh){return mesh.put(out);}
    
    Allocator<F>	_f;
    Allocator<V>	_v;
};

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
	u_int	fnum;
	in.width(sizeof(token));
	in >> token >> fnum;		// Skip face number.

	V*	v[M];
	u_int	e, vnum[M];
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
    
    for (u_int n = 0; n < topology.dim(); ++n)
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
    std::map<V*, u_int>	dict;
    u_int		vnum = 1;
    for (typename Allocator<V>::Enumerator vertices(_v); vertices; ++vertices)
    {
	dict[vertices] = vnum;
	out << "Vertex " << vnum++ << ' ' << *vertices;
    }
    u_int		fnum = 1;
    for (typename Allocator<F>::Enumerator faces(_f); faces; ++faces)
    {
	out << "Face " << fnum++;
	for (u_int e = 0; e < M; ++e)
	    out << ' ' << dict[&(faces->v(e))];
	out << std::endl;
    }
    
    return out;
}

template <class V, class E, class F, u_int M> BoundingBox<V>
Mesh<V, E, F, M>::boundingBox() const
{
    BoundingBox<V>	bbox;
    
    for (typename Allocator<F>::Enumerator faces(_f); faces; ++faces)
	for (u_int e = 0; e < M; ++e)
	    bbox.expand(faces->v(e));
    
    return bbox;
}

template <class V, class E, class F, u_int M> void
Mesh<V, E, F, M>::clean()
{
    using namespace	std;
    
    set<V*>	verticesUsed;
    for (typename Allocator<F>::Enumerator faces(_f); faces; ++faces)
	for (u_int e = 0; e < M; ++e)
	    verticesUsed.insert(&(faces->v(e)));
    for (typename Allocator<V>::Enumerator vertices(_v); vertices; ++vertices)
    {
	typename set<V*, less<V*> >::iterator	p = verticesUsed.find(vertices);
	if (p == verticesUsed.end())
	{
	    vertices->~V();
	    _v.free(vertices);
	}
    }
}

/************************************************************************
*  class Mesh<V, E, F, M>::Topology					*
************************************************************************/
template <class V, class E, class F, u_int M> Allocator<V>*
Mesh<V, E, F, M>::Topology::_a = 0;

template <class V, class E, class F, u_int M> void
Mesh<V, E, F, M>::Topology::pair() const
{
    for (typename List<FaceNode>::ConstIterator iter = begin(); iter != end(); ++iter)
    {
	Edge	edge(iter->f());
	while (&edge.v() != _v)
	    ++edge;
	V*	vn = &edge.next().v();
	for (typename List<FaceNode>::ConstIterator iter1(iter); ++iter1 != end(); )
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

/************************************************************************
*  class Mesh<V, E, F, M>::Edge						*
************************************************************************/
template <class V, class E, class F, u_int M> inline bool
Mesh<V, E, F, M>::Edge::operator ==(const Edge& edge) const
{
    return (_e == edge._e) && (_f == edge._f);
}

template <class V, class E, class F, u_int M> inline bool
Mesh<V, E, F, M>::Edge::operator !=(const Edge& edge) const
{
    return !(*this == edge);
}

template <class V, class E, class F, u_int M> bool
Mesh<V, E, F, M>::Edge::commonVertex(const Edge& edge) const
{
    Edge	tmp(*this);
    do
    {
	if (tmp == edge)
	    return true;
    } while (~(--tmp) != *this);
    
    return false;
}

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

template <class V, class E, class F, u_int M>
inline typename Mesh<V, E, F, M>::Edge&
Mesh<V, E, F, M>::Edge::operator ++()
{
    if (_e == M - 1)
	_e = 0;
    else
	++_e;
    return *this;
}

template <class V, class E, class F, u_int M>
inline typename Mesh<V, E, F, M>::Edge&
Mesh<V, E, F, M>::Edge::operator --()
{
    if (_e == 0)
	_e = M - 1;
    else
	--_e;
    return *this;
}

template <class V, class E, class F, u_int M>
inline typename Mesh<V, E, F, M>::Edge&
Mesh<V, E, F, M>::Edge::operator ~()
{
    return *this = conj();
}

template <class V, class E, class F, u_int M> inline E
Mesh<V, E, F, M>::Edge::next() const
{
    E	edge(*this);
    return ++edge;
}

template <class V, class E, class F, u_int M> inline E
Mesh<V, E, F, M>::Edge::prev() const
{
    E	edge(*this);
    return --edge;
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

template <class V, class E, class F, u_int M> inline void
Mesh<V, E, F, M>::Edge::pair(const Edge& edge) const
{
    _f->_f[_e] = edge._f;
    edge._f->_f[edge._e] = _f;
}

template <class V, class E, class F, u_int M> void
Mesh<V, E, F, M>::Edge::replaceVertex(V* v, const Edge& edgeE) const
{
    Edge	edgePC(prev().conj());
    if (edgePC != edgeE)
	edgePC.replaceVertex(v, edgeE);
    _f->_v[_e] = v;
}

template <class V, class E, class F, u_int M> inline void
Mesh<V, E, F, M>::Edge::replaceVertex(V* v) const
{
    _f->_v[_e] = v;
}

/************************************************************************
*  class Mesh<V, E, F, M>::Face						*
************************************************************************/
template <class V, class E, class F, u_int M>
#ifndef TUMeshPP_DEBUG
Mesh<V, E, F, M>::Face::Face(V* v[])
#else
Mesh<V, E, F, M>::Face::Face(V* v[], u_int fn)
    :fnum(fn)
#endif
{
    for (u_int e = 0; e < M; ++e)
    {
	_f[e] = 0;
	_v[e] = v[e];
    }
}

template <class V, class E, class F, u_int M> Vector3f
Mesh<V, E, F, M>::Face::centroid() const
{
    Vector3f	c;
    for (u_int i = 0; i < M; ++i)
	c += *_v[i];
    c /= M;
    return c;
}

}
#endif	// !__TUMeshPP_h
