/*
 *  $Id: Mesh++.h,v 1.3 2002-07-26 08:59:31 ueshiba Exp $
 */
#ifndef __TUMeshPP_h
#define __TUMeshPP_h

#include "TU/Geometry++.h"
#include "TU/Allocator++.h"

namespace TU
{
typedef Coordinate<float, 3u>	Coord3f;

inline Coord3f
operator +(const Coord3f& a, const Coord3f& b)
    {Coord3f r(a); r += b; return r;}

inline Coord3f
operator -(const Coord3f& a, const Coord3f& b)
    {Coord3f r(a); r -= b; return r;}

inline Coord3f
operator *(double c, const Coord3f& a)
    {Coord3f r(a); r *= c; return r;}

inline Coord3f
operator *(const Coord3f& a, double c)
    {Coord3f r(a); r *= c; return r;}

inline Coord3f
operator /(const Coord3f& a, double c)
    {Coord3f r(a); r /= c; return r;}

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
	Topology()		    :List<FaceNode>(), _v(0)		{}
	Topology(const Topology& t) :List<FaceNode>(), _v(t._v)	{}

	Topology&	operator =(const Topology& t)	{_v=t._v;return *this;}
	
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
	int	operator ==(const Edge& edge)	const	;
	int	operator !=(const Edge& edge)	const	;
	int	commonVertex(const Edge& edge)	const	;
	u_int	valence()			const	;
	Edge&	operator ++()				;
	Edge&	operator --()				;
	Edge&	operator ~()				;
	E	next()				const	;
	E	prev()				const	;
	E	conj()				const	;

      protected:
	int	e()				const	{return _e;}
	
      private:
	friend class 	Topology;	// Allow access to pair().
	friend class	Mesh;		// Allow access to relaceVertex().
	
	void	pair(const Edge& edge)			const	;
	void	replaceVertex(V* v, const Edge& edgeE)	const	;
	void	replaceVertex(V* v)			const	;
	
	F*	_f;		// parent face
	int	_e;		// my edge number
    };

    class Face
    {
      public:
#ifndef TUMeshPP_DEBUG
	Face(V* v[])					;
#else
	Face(V* v[], int fn)				;
#endif
	~Face()						{}

	F&			f(int e)	const	{return *_f[e];}
	V&			v(int e)	const	{return *_v[e];}
	Coordinate<float, 3u>	centroid()	const	;
	
	void*	operator new(size_t, void* p)		{return p;}
	
      private:
	friend class	Edge;
    
	F*		_f[M];		// _f[e] : neighboring face of e
	V*		_v[M];		// _v[e] : starting vertex of e
#ifdef TUMeshPP_DEBUG
      public:
	const int	fnum;
#endif
    };

    struct BoundingBox
    {
	float	xlength()		const	{return xmax - xmin;}
	float	ylength()		const	{return ymax - ymin;}
	float	zlength()		const	{return zmax - zmin;}
	float	maxlength()		const	;
	float	minlength()		const	;
	
	float	xmin, xmax, ymin, ymax, zmin, zmax;
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
    BoundingBox		boundingBox()		const	;
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
 *  Implementations
 */
template <class V, class E, class F, u_int M> inline int
Mesh<V, E, F, M>::Edge::operator ==(const Edge& edge) const
{
    return (_e == edge._e) && (_f == edge._f);
}

template <class V, class E, class F, u_int M> inline int
Mesh<V, E, F, M>::Edge::operator !=(const Edge& edge) const
{
    return !(*this == edge);
}

template <class V, class E, class F, u_int M>
inline typename Mesh<V, E, F, M>::Edge&
Mesh<V, E, F, M>::Edge::operator ++()
{
    if (++_e >= M)
	_e -= M;
    return *this;
}

template <class V, class E, class F, u_int M>
inline typename Mesh<V, E, F, M>::Edge&
Mesh<V, E, F, M>::Edge::operator --()
{
    if (--_e < 0)
	_e += M;
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

template <class V, class E, class F, u_int M> inline void
Mesh<V, E, F, M>::Edge::pair(const Edge& edge) const
{
    _f->_f[_e] = edge._f;
    edge._f->_f[edge._e] = _f;
}

template <class V, class E, class F, u_int M> inline void
Mesh<V, E, F, M>::Edge::replaceVertex(V* v) const
{
    _f->_v[_e] = v;
}

}
#endif	// !__TUMeshPP_h
