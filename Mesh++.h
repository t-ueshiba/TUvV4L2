/*
 *  平成14-19年（独）産業技術総合研究所 著作権所有
 *  
 *  創作者：植芝俊夫
 *
 *  本プログラムは（独）産業技術総合研究所の職員である植芝俊夫が創作し，
 *  （独）産業技術総合研究所が著作権を所有する秘密情報です．著作権所有
 *  者による許可なしに本プログラムを使用，複製，改変，第三者へ開示する
 *  等の行為を禁止します．
 *  
 *  このプログラムによって生じるいかなる損害に対しても，著作権所有者お
 *  よび創作者は責任を負いません。
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
 *  $Id: Mesh++.h,v 1.23 2012-08-29 21:17:08 ueshiba Exp $
 */
/*!
  \file		Mesh++.h
  \brief	クラス TU::Mesh の定義と実装
*/
#ifndef __TUMeshPP_h
#define __TUMeshPP_h

#include <list>
#include <vector>
#include <map>
#include "TU/Geometry++.h"

namespace TU
{
/************************************************************************
*  class Mesh<V, F, M>							*
************************************************************************/
//! 多角形メッシュを表すクラス
/*!
  \param V	頂点の型
  \param F	面の型．Mesh<V, F, M>::Faceの派生クラスでなければならない．
  \param M	1つの面が持つ辺の数
*/
template <class V, class F, u_int M=3u>
class Mesh
{
  public:
    typedef V						vertex_type;
    typedef F						face_type;
    typedef typename std::list<V>::iterator		viterator;
    typedef typename std::list<V>::const_iterator	const_viterator;
    typedef typename std::list<F>::iterator		fiterator;
    typedef typename std::list<F>::const_iterator	const_fiterator;

    enum
    {
	NSides = M			//!< 1つの面が持つ辺の数
    };
    
  //! 多角形メッシュの面の基底となるクラス
    class Face
    {
      public:
	V&		v(u_int e)				const	;
	F&		f(u_int e)				const	;
	
	friend class	Edge;

      protected:
#ifndef TUMeshPP_DEBUG
	Face(viterator v[])						;
#else
	Face(viterator v[], u_int fn)					;
#endif

      private:
	fiterator	self()					const	;

      private:
	viterator	_v[NSides];	//!< この面の頂点を指す反復子
	fiterator	_f[NSides];	//!< この面に隣接する面を指す反復子
#ifdef TUMeshPP_DEBUG
      public:
	const u_int	fnum;
#endif
    };

  //! 多角形メッシュの辺を表すクラス
  /*!
    面を左に見るように向き付けされている．
  */
    class Edge
    {
      public:
	Edge(const Face& f)						;
	
	V&		v()					const	;
	F&		f()					const	;
	u_int		e()					const	;
	bool		operator ==(const Edge& edge)		const	;
	bool		operator !=(const Edge& edge)		const	;
	bool		commonVertex(const Edge& edge)		const	;
	u_int		valence()				const	;
	Edge&		operator ++()					;
	Edge&		operator --()					;
	Edge&		operator ~()					;
	Edge		next()					const	;
	Edge		prev()					const	;
	Edge		conj()					const	;

	friend class	Mesh;

      private:
	Edge(fiterator f)						;

	viterator	viter()					const	;
	fiterator	fiter()					const	;
	void		pair(const Edge& edge)			const	;
	void		replaceVertex(viterator v,
				      const Edge& edgeE)	const	;
	void		replaceVertex(viterator v)		const	;

      private:
	fiterator	_f;		//!< 親の面を指す反復子
	u_int		_e;		//!< 辺の番号
    };

  public:
    Edge		initialize(const V vertex[])			;
    void		clear()						;
    
    Edge		kill(Edge& edge)				;
    Edge		make(const Edge& edge0,
			     const Edge& edge1, const V& v)		;
    Edge		swap(const Edge& edge)				;
    BoundingBox<V>	boundingBox()				const	;

    viterator		vbegin()					;
    const_viterator	vbegin()				const	;
    viterator		vend()						;
    const_viterator	vend()					const	;
    fiterator		fbegin()					;
    const_fiterator	fbegin()				const	;
    fiterator		fend()						;
    const_fiterator	fend()					const	;
#ifdef TUMeshPP_DEBUG
    std::ostream&	showTopology(std::ostream& out)		const	;
#endif
    
  private:
    std::istream&	get(std::istream& in)				;
    std::ostream&	put(std::ostream& out)			const	;

    viterator		newVertex(const V& v)				;
    fiterator		newFace(const F& f)				;
    void		deleteVertex(viterator v)			;
    void		deleteFace(fiterator f)				;
    
  //! 入力ストリームからメッシュを読み込む．
  /*!
    \param in	入力ストリーム
    \param mesh	読み込み先のメッシュ
    \return	inで指定した入力ストリーム
  */
    friend std::istream&
	operator >>(std::istream& in, Mesh& mesh)	{return mesh.get(in);}

  //! 出力ストリームにメッシュを書き出す．
  /*!
    \param out	出力ストリーム
    \param mesh	書き出し元のメッシュ
    \return	outで指定した出力ストリーム
  */
    friend std::ostream&
	operator <<(std::ostream& out, const Mesh& mesh){return mesh.put(out);}

  private:
    std::list<V>	_vertices;			//!< 頂点のリスト
    std::list<F>	_faces;				//!< 面のリスト
};

//! 指定された頂点から背中合わせの2つの面を生成してメッシュを初期化する．
/*!
  \param vertex	M個の頂点
  \return	v[0] を始点とする辺
*/
template <class V, class F, u_int M> typename Mesh<V, F, M>::Edge
Mesh<V, F, M>::initialize(const V vertex[])
{
  // 表の面を生成する．
    viterator	v[NSides];
    for (u_int e = 0; e < NSides; ++e)
	v[e] = newVertex(vertex[e]);
    fiterator	f = newFace(F(v));

  // 裏の面を生成する．
    viterator	vC[NSides];
    for (u_int e = 0; e < NSides; ++e)
	vC[e] = v[NSides-1-e];
    fiterator	fC = newFace(F(vC));

  // 表と裏を貼り合わせる．
    Edge	edge0(f), edge(edge0), edgeC(fC);
    --edgeC;
    do
    {
	edge.pair(--edgeC);
    } while (++edge != edge0);

    return edge0;
}

//! メッシュの全ての頂点と面を消去して空にする．
template <class V, class F, u_int M> inline void
Mesh<V, F, M>::clear()
{
    _vertices.clear();
    _faces.clear();
}
    
//! 3角形メッシュについて，指定された辺を消去する．
/*!
  指定された辺の両側の面および辺の始点も消去される．
  \param edge	消去する辺．
		リターン後はedgeの手前の裏の辺を指すように更新される．
  \return	edgeの裏の手前の裏の辺を指す反復子
*/
template <class V, class F, u_int M> typename Mesh<V, F, M>::Edge
Mesh<V, F, M>::kill(Edge& edge)
{
    using namespace	std;
    
  // edgeを含む面とその裏の辺を含む面の頂点(計4つ)について，
  // その価数が充分か調べる．
    Edge	edgeNC(edge.next().conj()),
		edgeCPC(edge.conj().prev().conj()),
		edgeCNC(edge.conj().next().conj());
    if (edge.valence() + edgeCPC.valence() <= 6 ||
	edgeNC.valence() <= 3 || edgeCNC.valence() <= 3)
    {
	cerr << "TU::Mesh<V, F, 3u>::kill(): Too small valence!" << endl;
	return edge;
    }

    Edge	edgeCN(edge.conj().next());
    viterator	vn = edge.next().viter();		// edgeの終点
    for (Edge tmp(edge.prev().conj()); ~(--tmp) != edgeCN; )
	for (Edge tmp1(tmp); --(~tmp1) != tmp; )
	    if (tmp1.viter() == vn)
	    {
		cerr << "TU::Mesh<V, F, 3u>::kill(): "
		     << "Pre-requisits for topology are not met!"
		     << endl;
		return edge;
	    }
    
  // edgeの始点を頂点に持つ全ての面について，その頂点をedgeの終点に置き換える．
    viterator	v = edge.viter();
    edge.replaceVertex(vn, edge);
    deleteVertex(v);					// 頂点vを消去

  // edgePC, edgeCPCをそれぞれedgeNC, edgeCNCと背中合わせにする．
    fiterator	f = edge.fiter(), fC = edge.conj().fiter();
    ~(--edge);						// edgePC
    edge   .pair(edgeNC);
    edgeCPC.pair(edgeCNC);
    deleteFace(f);					// 面fを消去
    deleteFace(fC);					// 面fCを消去

    return edgeCPC;
}

//! 3角形メッシュについて，指定された2つの辺の始点間に新たに辺を作る．
/*!
  1つの頂点と2つの面が生成される．
  \param edge0	新たな辺と始点を共有する辺
  \param edge1	新たな辺と終点を共有する辺．
		edge0と始点を共有していなければならない．
  \param v	新たな辺およびedge0の始点となる頂点
  \return	新たな辺
*/
template <class V, class F, u_int M> typename Mesh<V, F, M>::Edge
Mesh<V, F, M>::make(const Edge& edge0, const Edge& edge1, const V& v)
{
    using namespace	std;

  // edge0とedge1が始点を共有しているかチェック．
    if (!edge0.commonVertex(edge1))
	throw domain_error("TU::Mesh<V, F, 3u>::make(): Given two edges have no common vertex!");

  // edge0とedge1が同一でないかチェック．
    if (edge0 == edge1)
	throw domain_error("TU::Mesh<V, F, 3u>::make(): Given two edges are identical!");

  // 新しい頂点を作る．
    viterator	vnew = newVertex(v);

  // 新しい面を2つ作る．
    viterator	vp[3];
    vp[0] = vnew;
    vp[1] = edge0.viter();
    vp[2] = edge0.next().viter();
    fiterator	f = newFace(F(vp));
    vp[0] = edge1.viter();
    vp[1] = vnew;
    vp[2] = edge1.next().viter();
    fiterator	fC = newFace(F(vp));

  // edge0の始点を置き換える前にedge0とedge1の裏を保持しておく．
    Edge	edge0C(edge0.conj()), edge1C(edge1.conj());

  // [edge0, edge1)の範囲の辺の始点を新しい頂点に置き換える.
    edge0.replaceVertex(vnew, edge1);

  // winged-edge構造を作る．
    Edge	edge(f), edgeC(fC);
    edge.pair(edgeC);
    (--edge ).pair(edge0);
    (--edge ).pair(edge0C);
    (--edgeC).pair(edge1);
    (--edgeC).pair(edge1C);
    
    return --edge;
}

//! 3角形メッシュの指定された辺を消去し，これによってできる四角形のもう一方の対角線を新たな辺として生成する．
/*!
  \param edge	消去する辺
  \return	生成された辺
*/
template <class V, class F, u_int M> typename Mesh<V, F, M>::Edge
Mesh<V, F, M>::swap(const Edge& edge)
{
    using namespace	std;
    
  // edgeの始点と終点の価数が1つ減っても3以上になるか調べる．
    Edge	edgePC(edge.prev().conj()),
		edgeCPC(edge.conj().prev().conj());
    if (edgePC.valence() <= 3 || edgeCPC.valence() <= 3)
    {
	cerr << "TU::Mesh<V, F, 3u>::swap(): Too small valence!" << endl;
	return edge;
    }
    
  // edgeの始点と終点を置き換える．
    Edge	edgeC(edge.conj()),
		edgeNC(edge.next().conj()),
		edgeCNC(edge.conj().next().conj());
    edge.replaceVertex(edgeCNC.viter());
    edge.next().replaceVertex(edgeNC.viter());
    edge.prev().replaceVertex(edgePC.viter());
    edgeC.replaceVertex(edgeNC.viter());
    edgeC.next().replaceVertex(edgeCNC.viter());
    edgeC.prev().replaceVertex(edgeCPC.viter());

  // 辺を入れ替えてwinged-edge構造を作る．
    edge.next().pair(edgePC);
    edge.prev().pair(edgeCNC);
    edgeC.next().pair(edgeCPC);
    edgeC.prev().pair(edgeNC);

    return edge;
}

//! メッシュのbounding boxを計算する．
/*!
  \return	bounding box
*/
template <class V, class F, u_int M> BoundingBox<V>
Mesh<V, F, M>::boundingBox() const
{
    BoundingBox<V>	bbox;
    
    for (const_fiterator f = _faces.begin(); f != _faces.end(); ++f)
	for (u_int e = 0; e < NSides; ++e)
	    bbox.expand(f->v(e));
    
    return bbox;
}

//! このメッシュの最初の頂点を指す反復子を返す．
/*!
  \return	最初の頂点を指す反復子
*/
template <class V, class F, u_int M> inline typename Mesh<V, F, M>::viterator
Mesh<V, F, M>::vbegin()
{
    return _vertices.begin();
}
    
//! このメッシュの最初の頂点を指す定数反復子を返す．
/*!
  \return	最初の頂点を指す定数反復子
*/
template <class V, class F, u_int M>
inline typename Mesh<V, F, M>::const_viterator
Mesh<V, F, M>::vbegin() const
{
    return _vertices.begin();
}
    
//! このメッシュ最後の頂点の次を指す反復子を返す．
/*!
  \return	最後の頂点の次を指す反復子
*/
template <class V, class F, u_int M> inline typename Mesh<V, F, M>::viterator
Mesh<V, F, M>::vend()
{
    return _vertices.end();
}
    
//! このメッシュ最後の頂点の次を指す定数反復子を返す．
/*!
  \return	最後の頂点の次を指す定数反復子
*/
template <class V, class F, u_int M>
inline typename Mesh<V, F, M>::const_viterator
Mesh<V, F, M>::vend() const
{
    return _vertices.end();
}

//! このメッシュの最初の面を指す反復子を返す．
/*!
  \return	最初の面を指す反復子
*/
template <class V, class F, u_int M> inline typename Mesh<V, F, M>::fiterator
Mesh<V, F, M>::fbegin()
{
    return _faces.begin();
}
    
//! このメッシュの最初の面を指す定数反復子を返す．
/*!
  \return	最初の面を指す定数反復子
*/
template <class V, class F, u_int M>
inline typename Mesh<V, F, M>::const_fiterator
Mesh<V, F, M>::fbegin() const
{
    return _faces.begin();
}
    
//! このメッシュ最後の面の次を指す反復子を返す．
/*!
  \return	最後の面の次を指す反復子
*/
template <class V, class F, u_int M> inline typename Mesh<V, F, M>::fiterator
Mesh<V, F, M>::fend()
{
    return _faces.end();
}
    
//! このメッシュ最後の面の次を指す定数反復子を返す．
/*!
  \return	最後の面の次を指す定数反復子
*/
template <class V, class F, u_int M>
inline typename Mesh<V, F, M>::const_fiterator
Mesh<V, F, M>::fend() const
{
    return _faces.end();
}

#ifdef TUMeshPP_DEBUG
//! 出力ストリームにこのメッシュを構成する面の接続関係を書き出す．
/*!
  \param out	出力ストリーム
  \return	outで指定した出力ストリーム
*/
template <class V, class F, u_int M> std::ostream&
Mesh<V, F, M>::showTopology(std::ostream& out) const
{
    for (const_fiterator f = _faces.begin(); f != _faces.end(); ++f)
    {
	out << "Face[" << f->fnum << "]:";
	for (u_int e = 0; e < NSides; ++e)
	    out << ' ' << f->f(e).fnum;
	out << std::endl;
    }

    return out;
}
#endif
    
//! 入力ストリームからメッシュを読み込む．
/*!
  \param in	入力ストリーム
  \return	inで指定した入力ストリーム
*/
template <class V, class F, u_int M> std::istream&
Mesh<V, F, M>::get(std::istream& in)
{
    using namespace	std;
    
    typedef vector<fiterator>			Faces;
    typedef typename Faces::const_iterator	FacesIterator;
    typedef vector<pair<viterator, Faces> >	VerticesWithFaces;
    typedef typename VerticesWithFaces::const_iterator
						VerticesWithFacesIterator;

  // 頂点と面のリストを空にする．
    clear();
    
  // 全ての頂点を読み込む．
    VerticesWithFaces	verticesWithFaces;
    char		c;
    while (in >> c && c == 'V')
    {
	char		dummy[64];
	u_int		vnum;
	V		vertex;
	in >> dummy >> vnum >> vertex;		// 頂点を読み込む．
	viterator	v = newVertex(vertex);	// 新しい頂点を生成

      // 追加した頂点への反復子を，この頂点を共有する面の空リストと一緒に登録．
	verticesWithFaces.push_back(make_pair(v, Faces()));
    }
    in.putback(c);

  // 全ての面を読み込む．
    while (in >> c && c == 'F')
    {
	char	dummy[64];
	u_int	fnum;
	in >> dummy >> fnum;		// 面番号をスキップ．

	viterator	v[NSides];	// この面の頂点を指す全反復子
	u_int		vnum[NSides];	// この面の頂点の全番号
	for (u_int e = 0; e < NSides; ++e)
	{
	    in >> vnum[e];		// この面の頂点の番号を読み込む．
	    --vnum[e];			// 頂点番号は1から始まるのでデクリメント
	    v[e] = verticesWithFaces[vnum[e]].first;	// 反復子を取り出す
	}
#ifndef TUMeshPP_DEBUG
	fiterator	f = newFace(F(v));		// 新しい面を生成
#else
	fiterator	f = newFace(F(v, fnum));	// 新しい面を生成
#endif
      // 個々の頂点に自身の親としてこの面を登録する．
	for (u_int e = 0; e < NSides; ++e)
	    verticesWithFaces[vnum[e]].second.push_back(f);
    }
    if (in)
	in.putback(c);

  // 個々の頂点について，それを囲む2つの隣接面間にwinged-edge構造を作る．
    for (VerticesWithFacesIterator vertex  = verticesWithFaces.begin();
	 vertex != verticesWithFaces.end(); ++vertex)
    {
	viterator	v     = vertex->first;
	const Faces&	faces = vertex->second;		// vを囲む面のリスト

      // vを囲む個々の面fについて．．．
	for (FacesIterator f = faces.begin(); f != faces.end(); ++f)
	{
	  // fについて，vを始点とする辺edgeを探す．
	    Edge	edge(*f);
	    while (edge.viter() != v)
		++edge;

	    viterator	vn = edge.next().viter();	// edgeの終点
	    
	  // vertexを囲む別の面f1について．．．
	    for (FacesIterator f1 = faces.begin(); f1 != f; ++f1)
	    {
	      // f1のまわりを一周してedgeの終点vnを始点とする辺を探す．
		Edge	edgeF0(*f1), edgeF(edgeF0);
		do
		{
		    if (edgeF.viter() == vn)		// 始点がvnであれば
		    {
			edge.pair(edgeF);		// この辺がedgeの裏辺
			goto done;
		    }
		} while (++edgeF != edgeF0);
	    }
	  done:
	    continue;
	}
    }
#ifdef TUMeshPP_DEBUG
    showTopology(cerr);
#endif
    
    return in;
}

//! 出力ストリームにメッシュを書き出す．
/*!
  \param out	出力ストリーム
  \return	outで指定した出力ストリーム
*/
template <class V, class F, u_int M> std::ostream&
Mesh<V, F, M>::put(std::ostream& out) const
{
    using namespace	std;
    
    map<const V*, u_int>	dict;
    u_int			vnum = 1;
    for (const_viterator v = vbegin(); v != vend(); ++v)
    {
	dict[&(*v)] = vnum;
	out << "Vertex " << vnum++ << ' ' << *v;
    }
    u_int	fnum = 1;
    for (const_fiterator f = fbegin(); f != fend(); ++f)
    {
	out << "Face " << fnum++;
	for (u_int e = 0; e < NSides; ++e)
	    out << ' ' << dict[&(f->v(e))];
	out << std::endl;
    }
    
    return out;
}

//! 新しい頂点を生成して頂点リストに登録する．
/*!
 \param f	生成する頂点のプロトタイプ
 \return	生成された頂点を指す反復子
*/
template <class V, class F, u_int M> inline typename Mesh<V, F, M>::viterator
Mesh<V, F, M>::newVertex(const V& v)
{
    _vertices.push_front(v);
    return _vertices.begin();
}
    
//! 新しい面を生成して面リストに登録する．
/*!
 \param f	生成する面のプロトタイプ
 \return	生成された面を指す反復子
*/
template <class V, class F, u_int M> inline typename Mesh<V, F, M>::fiterator
Mesh<V, F, M>::newFace(const F& f)
{
    _faces.push_front(f);
    return _faces.begin();
}
    
//! 指定された頂点を破壊して頂点リストから取り除く．
/*!
 \param f	破壊する頂点を指す反復子
*/
template <class V, class F, u_int M> inline void
Mesh<V, F, M>::deleteVertex(viterator v)
{
    _vertices.erase(v);
}
    
//! 指定された面を破壊して面リストから取り除く．
/*!
 \param f	破壊する面を指す反復子
*/
template <class V, class F, u_int M> inline void
Mesh<V, F, M>::deleteFace(fiterator f)
{
    _faces.erase(f);
}
    
/************************************************************************
*  class Mesh<V, F, M>::Face						*
************************************************************************/
//! 頂点を指定して面を生成する．
/*!
  \param v	M個の頂点への反復子
*/
template <class V, class F, u_int M>
#ifndef TUMeshPP_DEBUG
Mesh<V, F, M>::Face::Face(viterator v[])
#else
Mesh<V, F, M>::Face::Face(viterator v[], u_int fn)
    :fnum(fn)
#endif
{
    for (u_int e = 0; e < NSides; ++e)
    {
      //_f[e] = 0;
	_v[e] = v[e];
    }
}

//! 指定された辺の始点を返す．
/*!
    \param e	辺のindex, 0 <= e < M
    \return	e番目の辺の始点すなわちこの面のe番目の頂点
*/
template <class V, class F, u_int M> inline V&
Mesh<V, F, M>::Face::v(u_int e) const
{
    return *_v[e];
}

//! 指定された辺を介してこの面に隣接する面を返す．
/*!
    \param e	辺のindex, 0 <= e < M
    \return	e番目の辺を介して隣接する面
*/
template <class V, class F, u_int M> inline F&
Mesh<V, F, M>::Face::f(u_int e) const
{
    return *_f[e];
}

//! この面を指す反復子を返す．
/*!
    \return	この面を指す反復子
*/
template <class V, class F, u_int M> typename Mesh<V, F, M>::fiterator
Mesh<V, F, M>::Face::self() const
{
    using namespace	std;
    
    fiterator	fC = _f[0];		// 0番目の辺を介して隣接する面
    for (u_int e = 0; e < NSides; ++e)
    {					// fCのe番目の辺を
	fiterator	f = fC->_f[e];	// 介して隣接する面への反復子fが
	if (&(*f) == this)		// この面を指していたら
	    return f;			// fがこの面への反復子である．
    }

    throw runtime_error("TU::Mesh<V, F, M>::Face::self(): Internal error!");

    return fC;
}

/************************************************************************
*  class Mesh<V, F, M>::Edge						*
************************************************************************/
//! 指定された面の最初の辺を指すように初期化する．
/*!
  \param face	面
*/
template <class V, class F, u_int M> inline
Mesh<V, F, M>::Edge::Edge(const Face& face)
    :_f(face.self()), _e(0)
{
}

//! この辺の始点を返す．
/*!
  \return	この辺の始点
*/
template <class V, class F, u_int M> inline V&
Mesh<V, F, M>::Edge::v() const
{
    return *viter();
}
    
//! この辺を所有する面を返す．
/*!
  \return	この辺を所有する面
*/
template <class V, class F, u_int M> inline F&
Mesh<V, F, M>::Edge::f() const
{
    return *fiter();
}
    
//! この辺の番号を返す．
/*!
  \return	辺の番号
*/
template <class V, class F, u_int M> inline u_int
Mesh<V, F, M>::Edge::e() const
{
    return _e;
}

//! 2つの辺が同一であるか調べる．
/*!
  \param edge	比較対象の辺
  \return	同一ならtrue, そうでなければfalse
*/
template <class V, class F, u_int M> inline bool
Mesh<V, F, M>::Edge::operator ==(const Edge& edge) const
{
    return (_e == edge._e) && (_f == edge._f);
}

//! 2つの辺が異なるか調べる．
/*!
  \param edge	比較対象の辺
  \return	異なればtrue, そうでなければfalse
*/
template <class V, class F, u_int M> inline bool
Mesh<V, F, M>::Edge::operator !=(const Edge& edge) const
{
    return !(*this == edge);
}

//! 2つの辺が始点を共有しているか調べる．
/*!
  \param edge	比較対象の辺
  \return	共有していればtrue, そうでなければfalse
*/
template <class V, class F, u_int M> bool
Mesh<V, F, M>::Edge::commonVertex(const Edge& edge) const
{
    Edge	tmp(*this);
    do
    {
	if (tmp == edge)
	    return true;
    } while (~(--tmp) != *this);
    
    return false;
}

//! 辺の始点の価数，すなわちその点を共有する面(辺)の数を返す．
/*!
  \return	辺の始点の価数
*/
template <class V, class F, u_int M> u_int
Mesh<V, F, M>::Edge::valence() const
{
    u_int	n = 0;
    Edge	edge(*this);
    do
    {
	++n;
    } while (~(--edge) != *this);

    return n;
}

//! 次の辺に前進する．
/*!
  \return	前進後の辺
*/
template <class V, class F, u_int M> inline typename Mesh<V, F, M>::Edge&
Mesh<V, F, M>::Edge::operator ++()
{
    if (_e == NSides - 1)
	_e = 0;
    else
	++_e;
    return *this;
}

//! 手前の辺に後退する．
/*!
  \return	後退後の辺
*/
template <class V, class F, u_int M> inline typename Mesh<V, F, M>::Edge&
Mesh<V, F, M>::Edge::operator --()
{
    if (_e == 0)
	_e = NSides - 1;
    else
	--_e;
    return *this;
}

//! 裏側の辺に移動する．
/*!
  \return	移動後の辺
*/
template <class V, class F, u_int M> inline typename Mesh<V, F, M>::Edge&
Mesh<V, F, M>::Edge::operator ~()
{
    return *this = conj();
}

//! この辺の次の辺を返す．
/*!
  \return	次の辺
*/
    template <class V, class F, u_int M> inline typename Mesh<V, F, M>::Edge
Mesh<V, F, M>::Edge::next() const
{
    Edge	edge(*this);
    return ++edge;
}

//! この辺の手前の辺を返す．
/*!
  \return	手前の辺
*/
template <class V, class F, u_int M> inline typename Mesh<V, F, M>::Edge
Mesh<V, F, M>::Edge::prev() const
{
    Edge	edge(*this);
    return --edge;
}

//! この辺の裏側の辺を返す．
/*!
  \return	裏側の辺
*/
template <class V, class F, u_int M> typename Mesh<V, F, M>::Edge
Mesh<V, F, M>::Edge::conj() const
{
    viterator	vn = next().viter();	// この辺の終点
    Edge	edge(_f->_f[_e]);	// この辺を介して隣接する面の最初の辺
    while (edge.viter() != vn)		// この辺の終点を始点とする辺を探す
	++edge;
    return edge;
}

//! この辺を所有する面を指す反復子を返す．
/*!
  \return	この辺を所有する面を指す反復子
*/
template <class V, class F, u_int M> inline typename Mesh<V, F, M>::fiterator
Mesh<V, F, M>::Edge::fiter() const
{
    return _f;
}
    
//! この辺の始点を指す反復子を返す．
/*!
  \return	この辺の始点を指す反復子
*/
template <class V, class F, u_int M> inline typename Mesh<V, F, M>::viterator
Mesh<V, F, M>::Edge::viter() const
{
    return _f->_v[_e];
}
    
//! 指定された面の最初の辺を指すように初期化する．
/*!
  \param f	面を指す反復子
*/
template <class V, class F, u_int M> inline
Mesh<V, F, M>::Edge::Edge(fiterator f)
    :_f(f), _e(0)
{
}

//! この辺と指定された辺を背中合わせにする．
/*!
  \param edge	背中合わせの対象となる辺
*/
template <class V, class F, u_int M> inline void
Mesh<V, F, M>::Edge::pair(const Edge& edge) const
{
    _f->_f[_e] = edge._f;	// この辺の裏面をedgeの親面に
    edge._f->_f[edge._e] = _f;	// edgeの裏面をこの辺の親面に
}

//! この辺から指定された辺の手前までの辺の始点を指定された頂点に置き換える．
/*!
  この辺の始点を共有する辺を反時計回りに走査しながら始点を置き換えてゆく．
  \param v	頂点を指す反復子
  \param edgeE	走査の終点となる辺 (この辺の始点は置き換えられない)
*/
template <class V, class F, u_int M> void
Mesh<V, F, M>::Edge::replaceVertex(viterator v, const Edge& edgeE) const
{
  // 先に始点を書き換えてしまうと裏に移れなくなってしまうので，
  // 再帰的に処理することによってまずedgeEの1つ手前の辺まで移動し，
  // 戻りながら順次書き換える．
    Edge	edgePC(prev().conj());
    if (edgePC != edgeE)
	edgePC.replaceVertex(v, edgeE);		// 再帰する．
    _f->_v[_e] = v;
}

//! この辺の始点を指定された頂点に置き換える．
/*!
  \param v	頂点を指す反復子
*/
template <class V, class F, u_int M> inline void
Mesh<V, F, M>::Edge::replaceVertex(viterator v) const
{
    _f->_v[_e] = v;
}

}
#endif	// !__TUMeshPP_h
