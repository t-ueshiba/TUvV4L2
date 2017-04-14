/*
 *  平成14-24年（独）産業技術総合研究所 著作権所有
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
 *  Copyright 2002-2012.
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
 *  $Id: GraphCuts.h 1829 2015-06-15 00:30:03Z ueshiba $
 */
/*!
  \file		GraphCuts.h
  \brief	graph cutに関するクラスの定義と実装
*/
#ifndef __TU_GRAPHCUTS_H
#define __TU_GRAPHCUTS_H

#include <boost/version.hpp>
#include <boost/graph/graph_traits.hpp>
#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/read_dimacs.hpp>
#include <boost/graph/edmonds_karp_max_flow.hpp>
#include <boost/graph/push_relabel_max_flow.hpp>
#if BOOST_VERSION<104601
#  include <boost/graph/kolmogorov_max_flow.hpp>
#  define boykov_kolmogorov_max_flow	kolmogorov_max_flow
#else
#  include <boost/graph/boykov_kolmogorov_max_flow.hpp>
#endif
#include <boost/foreach.hpp>
#include <iostream>
#include <algorithm>
#include <stack>
#include <cassert>

/*!
  \namespace	boost
  \brief	boostライブラリを利用したクラスや関数を名前空間boostに追加
*/
namespace boost
{
enum vertex_id_t	{vertex_id};		// サイトのID
enum vertex_label_t	{vertex_label};		// サイトのラベル
enum vertex_sedge_t	{vertex_sedge};		// 開始点からの辺
enum vertex_tedge_t	{vertex_tedge};		// 終端点への辺
enum edge_smooth_t	{edge_smooth};		// 平滑化項を表す辺
BOOST_INSTALL_PROPERTY(vertex, id);
BOOST_INSTALL_PROPERTY(vertex, label);
BOOST_INSTALL_PROPERTY(vertex, sedge);
BOOST_INSTALL_PROPERTY(vertex, tedge);
BOOST_INSTALL_PROPERTY(edge, smooth);
    
/************************************************************************
*  class GraphCuts<T, ID, L, EL>					*
************************************************************************/
//! グラフカットによってエネルギー最小化を行うクラス
/*!
  \param T	エネルギー値の型(負数を表現できなければならない)
  \param ID	サイトを特定するIDの型
  \param L	サイトのラベルの型
  \param EL	グラフの辺の実装(boost::vecS, boost::listS のいずれか)
*/
template <class T, class ID=int, class L=bool, class EL=vecS>
class GraphCuts
{
  public:
    typedef T	value_type;		//!< エネルギー値の型
    typedef ID	id_type;		//!< サイトを特定するIDの型
    typedef L	label_type;		//!< サイトのラベルの型
    enum Algorithm			//<! 最大フローアルゴリズム
    {
	BoykovKolmogorov		//!< Boykov-Kolmogorovアルゴリズム
#ifdef WITH_PARALLEL_EDGES
	, EdmondsKarp,			//!< Edmonds-Karpアルゴリズム
	PushRelabel			//!< Push-Relabelアルゴリズム
#endif
    };
    
  private:
    typedef adjacency_list_traits<EL, vecS, directedS>	traits_t;
    typedef typename traits_t::vertex_descriptor	vertex_t;
    typedef typename traits_t::edge_descriptor		edge_t;
    typedef default_color_type			color_t;
    typedef color_traits<color_t>		color_traits_t;
    typedef property<vertex_id_t,		id_type,
	    property<vertex_label_t,		label_type,
	    property<vertex_color_t,		color_t,
	    property<vertex_sedge_t,		edge_t,
	    property<vertex_tedge_t,		edge_t,
	    property<vertex_distance_t,		long,
	    property<vertex_predecessor_t,	edge_t
		     > > > > > > >			VertexProps;
    typedef property<edge_index_t,		int,
	    property<edge_capacity_t,		value_type,
	    property<edge_residual_capacity_t,	value_type,
	    property<edge_reverse_t,		edge_t,
	    property<edge_smooth_t,		bool
		     > > > > >				EdgeProps;
    typedef adjacency_list<EL, vecS, directedS,
			   VertexProps, EdgeProps>	graph_t;
    typedef typename graph_traits<graph_t>::vertex_iterator
							vertex_iterator;
    typedef typename graph_traits<graph_t>::out_edge_iterator
							out_edge_iterator;

  public:
    typedef vertex_t			site_type;	//!< サイトの型
    typedef std::pair<vertex_iterator,
		      vertex_iterator>	site_range;	//!< サイトの範囲
    
  public:
  // 構造の生成と破壊
    GraphCuts()								;
    site_type		createDataTerm(const id_type& id)		;
    void		createSmoothingTerm(site_type u, site_type v)	;
    void		clear()						;

  // サイト情報の取得と設定
    static site_type	nullSite()					;
    site_range		sites()					const	;
    size_t		nsites()				const	;
    id_type		id(site_type v)				const	;
    label_type		operator ()(site_type v)		const	;
    label_type&		operator ()(site_type v)			;
    bool		haveSmoothingTerm(site_type u,
					  site_type v)		const	;
    value_type		dataEnergy(site_type v, bool source)	const	;
    value_type&		dataEnergy(site_type v, bool source)		;
    value_type		smoothingEnergy(site_type u, site_type v) const	;
    value_type&		smoothingEnergy(site_type u, site_type v)	;
    
  // 最適化計算
    template <class F>
    value_type		value(F energyTerm)			const	;
    value_type		maxFlow(label_type alpha, Algorithm alg)	;
    template <class F>
    value_type		alphaExpansion(label_type alpha, F energyTerm,
				       Algorithm alg)			;
    void		check()					const	;
    
  // 入出力
    std::istream&	getDimacsMaxFlow(std::istream& in)		;
    std::ostream&	putDimacsMaxFlow(std::ostream& out)	const	;
    std::ostream&	putCapacities(std::ostream& out)	const	;
    std::ostream&	putMaxFlow(std::ostream& out)		const	;
    std::ostream&	putMinCut(std::ostream& out)		const	;
    
  private:
    edge_t		createEdgePair(vertex_t u, vertex_t v)		;
    void		computeMinCut()					;
    value_type		flow(edge_t e)				const	;
    
  private:
    graph_t		_g;	//!< グラフ
    vertex_t		_s;	//!< 開始点
    vertex_t		_t;	//!< 終端点
};

/*
 * ----------------------- 構造の生成と破壊 ----------------------------
 */
//! グラフカット実行器を生成する．
template <class T, class ID, class L, class EL> inline
GraphCuts<T, ID, L, EL>::GraphCuts()
    :_g(),
     _s(add_vertex(_g)),
     _t(add_vertex(_g))
{
}

//! データ項を生成する．
/*!
  \param id	サイトのID
  \return	サイトを表す頂点
*/ 
template <class T, class ID, class L, class EL>
inline typename GraphCuts<T, ID, L, EL>::site_type
GraphCuts<T, ID, L, EL>::createDataTerm(const id_type& id)
{
    const vertex_t	v = add_vertex(_g);	      // 頂点を生成
    put(vertex_id,    _g, v, id);		      // そのIDを登録
    put(vertex_sedge, _g, v, createEdgePair(_s, v));  // 開始点への辺対を生成
    put(vertex_tedge, _g, v, createEdgePair(v, _t));  // 終端点への辺対を生成

    return v;
}

//! 平滑化項をセットする．
/*!
  \param u	平滑化項が対象とする一方のサイト
  \param v	平滑化項が対象とするもう一方のサイト
*/ 
template <class T, class ID, class L, class EL> inline void
GraphCuts<T, ID, L, EL>::createSmoothingTerm(site_type u, site_type v)
{
#ifdef WITH_PARALLEL_EDGES
    edge_t	e = createEdgePair(u, v);
#else
    edge_t	e;
    bool	exists;
    tie(e, exists) = edge(u, v, _g);	// 既に(u, v)があれば，それをeにセットし，
    if (!exists)			// なければ...
	e = createEdgePair(u, v);	// 新たに作る．
#endif
    put(edge_smooth, _g, e, true);	// 平滑化項フラグをセット
}
    
//! グラフの全ての辺および開始点と終端点を除く全ての頂点を除去する．
template <class T, class ID, class L, class EL> inline void
GraphCuts<T, ID, L, EL>::clear()
{
    _g.clear();
    _s = add_vertex(_g);
    _t = add_vertex(_g);
}

/*
 * ----------------------- サイト情報の取得と設定 ----------------------
 */
//! 実際には存在していないダミーサイトを返す．
/*!
  \return	ダミーサイト
*/
template <class T, class ID, class L, class EL>
inline typename GraphCuts<T, ID, L, EL>::site_type
GraphCuts<T, ID, L, EL>::nullSite()
{
    return graph_traits<graph_t>::null_vertex();
}
    
//! 全サイト(開始点と終端点を除く全ての頂点)の範囲を示す反復子ペアを返す．
/*!
  \return	全サイトの範囲
*/
template <class T, class ID, class L, class EL>
inline typename GraphCuts<T, ID, L, EL>::site_range
GraphCuts<T, ID, L, EL>::sites() const
{
    site_range	range = vertices(_g);
    assert(*range.first++ == _s);
    assert(*range.first++ == _t);

    return range;
}
    
//! サイト数を返す．
/*!
  \return	サイト数
*/
template <class T, class ID, class L, class EL> inline size_t
GraphCuts<T, ID, L, EL>::nsites() const
{
    return num_vertices(_g) - 2;
}
    
//! 指定されたサイトのIDを返す．
/*!
  \param v	サイトを表す頂点
  \return	サイトのID
*/
template <class T, class ID, class L, class EL>
inline typename GraphCuts<T, ID, L, EL>::id_type
GraphCuts<T, ID, L, EL>::id(site_type v) const
{
    return get(vertex_id, _g, v);
}
    
//! 指定されたサイトのラベルを返す．
/*!
  \param v	サイトを表す頂点
  \return	サイトのラベル
*/
template <class T, class ID, class L, class EL>
inline typename GraphCuts<T, ID, L, EL>::label_type
GraphCuts<T, ID, L, EL>::operator ()(site_type v) const
{
    return get(vertex_label, _g, v);
}

//! 指定されたサイトのラベルを返す．
/*!
  \param v	サイトを表す頂点
  \return	サイトのラベル
*/
template <class T, class ID, class L, class EL>
inline typename GraphCuts<T, ID, L, EL>::label_type&
GraphCuts<T, ID, L, EL>::operator ()(site_type v)
{
    return get(vertex_label, _g)[v];
}

//! 指定された2つのサイト間に平滑化エネルギー項があるか調べる．
/*!
  \param u	サイトを表す頂点
  \param v	もう一つのサイトを表す頂点
  \return	平滑化エネルギー項があればtrue, なければfalse
*/
template <class T, class ID, class L, class EL> inline bool
GraphCuts<T, ID, L, EL>::haveSmoothingTerm(site_type u, site_type v) const
{
    BOOST_FOREACH (edge_t e, out_edges(u, _g))
	if (target(e, _g) == v && get(edge_smooth, _g, e))
	    return true;
    return false;
}
    
//! 指定されたサイトのデータエネルギー値を返す．
/*!
  \param v	サイトを表す頂点
  \param source	trueならば開始点側を，falseならば終端点側をそれぞれ切断
  \return	データエネルギー値
*/
template <class T, class ID, class L, class EL>
inline typename GraphCuts<T, ID, L, EL>::value_type
GraphCuts<T, ID, L, EL>::dataEnergy(site_type v, bool source) const
{
    edge_t	e = (source ? get(vertex_sedge, _g, v)
			    : get(vertex_tedge, _g, v));
    return get(edge_capacity, _g, e);
}

//! 指定されたサイトのデータエネルギー値への参照を返す．
/*!
  \param v	サイトを表す頂点
  \param source	trueならば開始点側を，falseならば終端点側をそれぞれ切断
  \return	データエネルギー値への参照
*/
template <class T, class ID, class L, class EL>
inline typename GraphCuts<T, ID, L, EL>::value_type&
GraphCuts<T, ID, L, EL>::dataEnergy(site_type v, bool source)
{
    edge_t	e = (source ? get(vertex_sedge, _g, v)
			    : get(vertex_tedge, _g, v));
    return get(edge_capacity, _g, e);
}

//! 指定された2つのサイト間の平滑化エネルギー値を返す．
/*!
  これらのサイト間に平滑化エネルギー項がなければ0が返される．
  \param u	サイトを表す頂点
  \param v	もう一つのサイトを表す頂点
  \return	平滑化エネルギー項があればそのエネルギー値，なければ0
*/
template <class T, class ID, class L, class EL>
typename GraphCuts<T, ID, L, EL>::value_type
GraphCuts<T, ID, L, EL>::smoothingEnergy(site_type u, site_type v) const
{
#ifdef WITH_PARALLEL_EDGES
    BOOST_FOREACH (edge_t e, out_edges(u, _g))
	if (target(e, _g) == v && get(edge_smooth, _g, e))
	    return get(edge_capacity, _g, e);
    return 0;
#else
    edge_t	e;
    bool	exists;
    tie(e, exists) = edge(u, v, _g);
    return (exists && get(edge_smooth, _g, e) ? get(edge_capacity, _g, e) : 0);
#endif
}

//! 指定された2つのサイト間の平滑化エネルギー値への参照を返す．
/*!
  これらのサイト間に平滑化エネルギー項がなければ，例外が送出される．
  \param u	サイトを表す頂点
  \param v	もう一つのサイトを表す頂点
  \return	平滑化エネルギー値への参照
*/
template <class T, class ID, class L, class EL>
typename GraphCuts<T, ID, L, EL>::value_type&
GraphCuts<T, ID, L, EL>::smoothingEnergy(site_type u, site_type v)
{
#ifdef WITH_PARALLEL_EDGES
    BOOST_FOREACH (edge_t e, out_edges(u, _g))
	if (target(e, _g) == v && !get(edge_smooth, _g, e))
	    return get(edge_capacity, _g, e);
    throw std::runtime_error("GraphCuts<T, ID, L, EL>::energy(): non-existing smoothing term!");
    return get(edge_capacity, _g, get(vertex_sedge, _g, u));  // ここには到達せず
#else
    edge_t	e;
    bool	exists;
    tie(e, exists) = edge(u, v, _g);
    if (!exists || !get(edge_smooth, _g, e))
	throw std::runtime_error("GraphCuts<T, ID, L, EL>::energy(): non-existing smoothing term!");
    return get(edge_capacity, _g, e);
#endif
}

/*
 * ----------------------------- 最適化計算 -----------------------------
 */
//! 現在のラベル配置のもとでのエネルギー値を求める．
/*!
  \param energyTerm	サイトのIDとそのラベルを与えるとデータエネルギー値を
			返すメンバおよび隣接する2つのサイトのIDとそれらの
			ラベルを与えると平滑化エネルギー値を返すメンバの2つ
			を持つ関数オブジェクト
  \return		エネルギー値
*/
template <class T, class ID, class L, class EL> template <class F>
typename GraphCuts<T, ID, L, EL>::value_type
GraphCuts<T, ID, L, EL>::value(F energyTerm) const
{
    value_type	val = 0;				// エネルギー値
    BOOST_FOREACH (vertex_t u, sites())
    {
	const id_type		uid = id(u);
	const label_type	Xu  = (*this)(u);

	val += energyTerm(uid, Xu);				// データ項
	
	BOOST_FOREACH (edge_t e, out_edges(u, _g))
	{
	    if (get(edge_smooth, _g, e))
	    {
		const vertex_t	v = target(e, _g);

		val += energyTerm(uid, id(v), Xu, (*this)(v));	// 平滑化項
	    }
	}
    }
    
    return val;
}
    
//! 指定された最大フローアルゴリズムによってフローの最大値を求める．
/*!
  \param alpha	最小カットにより終端点側に分類されたサイトに与えるラベル
  \param alg	使用する最大フローアルゴリズム
  \return	フローの最大値
*/
template <class T, class ID, class L, class EL>
typename GraphCuts<T, ID, L, EL>::value_type
GraphCuts<T, ID, L, EL>::maxFlow(label_type alpha, Algorithm alg)
{
    value_type	f;
    switch (alg)
    {
#ifdef WITH_PARALLEL_EDGES
      case EdmondsKarp:
	f = edmonds_karp_max_flow(_g, _s, _t,
				  color_map(get(vertex_color, _g)).
				  predecessor_map(get(vertex_predecessor, _g)));
	break;
      case PushRelabel:
	f = push_relabel_max_flow(_g, _s, _t);
	computeMinCut();
	break;
#endif
      default:
	f = boykov_kolmogorov_max_flow(_g, _s, _t);
	computeMinCut();
	break;
    }

  // 終端点側の頂点にラベル値alphaを与える．
    BOOST_FOREACH (vertex_t v, sites())
	if (get(vertex_color, _g, v) == color_traits_t::white())
	    (*this)(v) = alpha;

    return f;
}

//! アルファ拡張を1回行う．
/*!
  alphaでないラベルを持つサイトについて，"ラベルを変えない"/"alphaに変える"の
  2値で最小カットを求め，その結果に応じてラベルを付け替える．
  \param alpha		拡張先のラベル
  \param energyTerm	サイトのIDとそのラベルを与えるとデータエネルギー値を
			返すメンバおよび隣接する2つのサイトのIDとそれらの
			ラベルを与えると平滑化エネルギー値を返すメンバの2つ
			を持つ関数オブジェクト
  \param alg		最大フローアルゴリズム
  \return		アルファ拡張で達成された最小エネルギー値
*/
template <class T, class ID, class L, class EL> template <class F>
typename GraphCuts<T, ID, L, EL>::value_type
GraphCuts<T, ID, L, EL>::alphaExpansion(label_type alpha, F energyTerm,
					Algorithm alg)
{
    using namespace	std;
    
  // すべての辺の容量を0にする．
    BOOST_FOREACH (edge_t e, edges(_g))
	put(edge_capacity, _g, e, 0);

  // alphaでないラベルを持つ頂点に接続する辺にエネルギー値を付与する．
    value_type	bias = 0;		// バイアス
    BOOST_FOREACH (vertex_t u, sites())
    {
	const id_type		uid = id(u);
	const label_type	Xu  = (*this)(u);

	if (Xu != alpha)		// u のラベルがalphaでなければ...
	{
	    const edge_t	es = get(vertex_sedge, _g, u),
				et = get(vertex_tedge, _g, u);

	    get(edge_capacity, _g)[es] += energyTerm(uid, alpha);
	    get(edge_capacity, _g)[et] += energyTerm(uid, Xu);
	    
	    BOOST_FOREACH (edge_t e, out_edges(u, _g))
	    {
		if (get(edge_smooth, _g, e))	// e が平滑化項を表すなら...
		{
		    const vertex_t	v   = target(e, _g);
		    const id_type	vid = id(v);
		    const label_type	Xv  = (*this)(v);
		    
		    if (Xv == alpha)	// e = (u, v), Xu != alpha, Xv = alpha
		    {
			get(edge_capacity, _g)[es] +=
			    energyTerm(uid, vid, alpha, alpha);
			get(edge_capacity, _g)[et] +=
			    energyTerm(uid, vid, Xu, alpha);
		    }
		    else		// e = (u, v), Xu != alpha, Xv != alpha
		    {
			const value_type
			    h00 = energyTerm(uid, vid,    Xu,    Xv),
			    h01 = energyTerm(uid, vid,    Xu, alpha),
			    h10 = energyTerm(uid, vid, alpha,    Xv),
			    h11 = energyTerm(uid, vid, alpha, alpha),
			    h   = h01 + h10 - h00 - h11;

			if (h < 0)
			    throw std::runtime_error("boost::GraphCuts<T, ID, L, EL>::alphaExpansion(): submodularity constraint is violated!");

			const edge_t	ev = get(vertex_tedge, _g, v);

			get(edge_capacity, _g)[es] += (h10 - h00);
			get(edge_capacity, _g)[ev] += (h10 - h11);
			get(edge_capacity, _g)[e ] += h;
		    
			bias -= (h - h01);
		    }
		}
	    }
	}
	else				// u のラベルがalphaならば...
	{
	    bias += energyTerm(uid, alpha);
	    
	    BOOST_FOREACH (edge_t e, out_edges(u, _g))
	    {
		if (get(edge_smooth, _g, e))
		{
		    const vertex_t	v   = target(e, _g);
		    const id_type	vid = id(v);
		    const label_type	Xv  = (*this)(v);

		    if (Xv == alpha)	// e = (u, v), Xu = Xv = alpha
			bias += energyTerm(uid, vid, alpha, alpha);
		    else		// e = (u, v), Xu = alpha, Xv != alpha
		    {
			const edge_t	es = get(vertex_sedge, _g, v),
					et = get(vertex_tedge, _g, v);
			
			get(edge_capacity, _g)[es] +=
			    energyTerm(uid, vid, alpha, alpha);
			get(edge_capacity, _g)[et] +=
			    energyTerm(uid, vid, alpha, Xv);
		    }
		}
	    }
	}
    }

  // _s から流れ出す辺と _t に流れ込む辺の容量が非負になるように調整する．
    BOOST_FOREACH (vertex_t v, sites())
    {
	const edge_t	es = get(vertex_sedge, _g, v),
			et = get(vertex_tedge, _g, v);
	value_type&	cs = get(edge_capacity, _g)[es];
	value_type&	ct = get(edge_capacity, _g)[et];

	if (cs < ct)
	{
	    bias += cs;
	    ct -= cs;
	    cs  = 0;
	}
	else
	{
	    bias += ct;
	    cs -= ct;
	    ct  = 0;
	}
    }
    
  // 最大フローと最小カットを求める．
    return bias + maxFlow(alpha, alg);
}

template <class T, class ID, class L, class EL> void
GraphCuts<T, ID, L, EL>::check() const
{
    using namespace	std;
    
    BOOST_FOREACH (edge_t e, edges(_g))
	if (get(edge_smooth, _g, e))
	    if (flow(e) + flow(get(edge_reverse, _g, e)) != 0)
		cerr << "Inconsistent flow!" << endl;
    
    value_type	f = 0;
    BOOST_FOREACH (edge_t e, out_edges(_s, _g))
	f += flow(e);
    cerr << " s => " << f << ", ";
    
  // xxx_max_flow() が返す最大フロー値が正しいかチェックする．
    f = 0;
    BOOST_FOREACH (edge_t e, edges(_g))
    {
	vertex_t	u = source(e, _g), v = target(e, _g);

#if 1
	if (get(vertex_color, _g, u) == color_traits_t::black() &&
	    get(vertex_color, _g, v) != color_traits_t::black())
#else	
	if (get(vertex_color, _g, u) != color_traits_t::white() &&
	    get(vertex_color, _g, v) == color_traits_t::white())
#endif
	{
	    f += get(edge_capacity, _g, e);

	    if (get(edge_residual_capacity, _g, e) != 0)
	    {
		edge_t	er = get(edge_reverse, _g, e);
		cerr << "\tc(" << u << ',' << v << ") = "
		     << get(edge_capacity, _g, e)
		     << ", r(" << u << ',' << v << ") = "
		     << get(edge_residual_capacity, _g, e)
		     << ", c(" << v << ',' << u << ") = "
		     << get(edge_capacity, _g, er)
		     << ", r(" << v << ',' << u << ") = "
		     << get(edge_residual_capacity, _g, er)
		     << endl;
	    }
	}
    }
    cerr << "cut = " << f << ", ";

    f = 0;
    BOOST_FOREACH (edge_t e, out_edges(_t, _g))
	f += flow(get(edge_reverse, _g, e));
    cerr << f << " => t" << endl;
}
    
/*
 * ------------------------------- 入出力 -------------------------------
 */
//! 入力ストリームから最大フロー問題を読み込む．
template <class T, class ID, class L, class EL> std::istream&
GraphCuts<T, ID, L, EL>::getDimacsMaxFlow(std::istream& in)
{
  // 全ての辺と開始点と終端点を含む全ての頂点を除去する．
    _g.clear();

    read_dimacs_max_flow(_g, get(edge_capacity, _g), get(edge_reverse, _g),
			 _s, _t, in);
    return in;
}

//! 出力ストリームに最大フロー問題を書き込む．
template <class T, class ID, class L, class EL> std::ostream&
GraphCuts<T, ID, L, EL>::putDimacsMaxFlow(std::ostream& out) const
{
    using namespace	std;

#if 1
    out << "p max\t" << num_vertices(_g) << '\t' << num_edges(_g)/2 << endl;

    out << "n\t" << _s + 1 << " s" << endl;	// 開始点
    out << "n\t" << _t + 1 << " t" << endl;	// 終端点

    int	nedges = 0;
    BOOST_FOREACH (vertex_t v, sites())
    {
	edge_t	es = get(vertex_sedge, _g, v),
		et = get(vertex_tedge, _g, v);
	
	out << "a\t" << _s + 1 << '\t' << v + 1
	    << '\t' << get(edge_capacity, _g, es) << endl;
	out << "a\t" << v + 1 << '\t' << _t + 1
	    << '\t' << get(edge_capacity, _g, et) << endl;
	nedges += 2;
	
	BOOST_FOREACH (edge_t e, out_edges(v, _g))
	{
	    if (get(edge_smooth, _g, e))
	    {
		out << "a\t" << v + 1 << '\t' << target(e, _g) + 1
		    << '\t' << get(edge_capacity, _g, e) << endl;
		++nedges;
	    }
	}
    }
    cerr << "#v = " << num_vertices(_g)
	 << ", #e(!zero)/#e = " << nedges << '/' << num_edges(_g)/2
	 << endl;
#else
    int	nNonzeroEdges = 0;
    BOOST_FOREACH (edge_t e, edges(_g))
    {
	if (get(edge_capacity, _g, e) > 0)
	    ++nNonzeroEdges;
    }

    out << "p max\t" << num_vertices(_g) << '\t' << nNonzeroEdges << endl;

    out << "n\t" << _s + 1 << " s" << endl;	// 開始点
    out << "n\t" << _t + 1 << " t" << endl;	// 終端点

    BOOST_FOREACH (edge_t e, edges(_g))
    {
	const value_type	c = get(edge_capacity, _g, e);
	if (c > 0)
	    out << "a\t" << source(e, _g) + 1 << '\t' << target(e, _g) + 1
		<< '\t' << c << endl;
    }
    cerr << "#v = " << num_vertices(_g)
	 << ", #e(!zero)/#e = " << nNonzeroEdges << '/' << num_edges(_g)
	 << endl;
#endif    
    return out;
}

template <class T, class ID, class L, class EL> std::ostream&
GraphCuts<T, ID, L, EL>::putCapacities(std::ostream& out) const
{
    using namespace	std;

    BOOST_FOREACH (vertex_t v, vertices(_g))
    {
	BOOST_FOREACH (edge_t e, out_edges(v, _g))
	{
	    out << "c(" << v << ", " << target(e, _g) << ") = "
		<< get(edge_capacity, _g, e)
		<< endl;
	}
    }

    return out;
}
    
template <class T, class ID, class L, class EL> std::ostream&
GraphCuts<T, ID, L, EL>::putMaxFlow(std::ostream& out) const
{
    using namespace	std;

    BOOST_FOREACH (vertex_t v, vertices(_g))
    {
	BOOST_FOREACH (edge_t e, out_edges(v, _g))
	{
	    value_type	c = get(edge_capacity, _g, e);
	    
	    if (c > 0)
	    {
		value_type	r = get(edge_residual_capacity, _g, e);

		out << "f(" << v << ", " << target(e, _g) << ") = "
		    << c - r;
		if (r == 0)
		    out << " *";
		out << endl;
	    }
	}
    }

    return out;
}
    
template <class T, class ID, class L, class EL> std::ostream&
GraphCuts<T, ID, L, EL>::putMinCut(std::ostream& out) const
{
    using namespace	std;

    BOOST_FOREACH (vertex_t v, vertices(_g))
    {
	out << "p[" << v << "] = "
	    << (get(vertex_color, _g, v) == color_traits_t::white() ? 1 : 0)
	    << endl;
    }

    return out;
}

/*
 * ----------------------- private members -----------------------------
 */
template <class T, class ID, class L, class EL>
inline typename GraphCuts<T, ID, L, EL>::edge_t
GraphCuts<T, ID, L, EL>::createEdgePair(vertex_t u, vertex_t v)
{
    edge_t	e0 = add_edge(u, v, _g).first;
    edge_t	e1 = add_edge(v, u, _g).first;
    put(edge_capacity, _g, e0, 0);
    put(edge_capacity, _g, e1, 0);
    put(edge_residual_capacity, _g, e0, 0);
    put(edge_residual_capacity, _g, e1, 0);
    put(edge_reverse,  _g, e0, e1);
    put(edge_reverse,  _g, e1, e0);
    put(edge_smooth,   _g, e0, false);
    put(edge_smooth,   _g, e1, false);

    return e0;
}
    
template <class T, class ID, class L, class EL> inline void
GraphCuts<T, ID, L, EL>::computeMinCut()
{
  // 全頂点を白に塗る．
    BOOST_FOREACH (vertex_t v, vertices(_g))
	put(vertex_color, _g, v, color_traits_t::white());

  // 開始点から飽和していない辺を通って到達できる頂点を黒に塗る．
    std::stack<vertex_t>	stack;	// 深さ優先探索のためのスタック
    stack.push(_s);
    
    while (!stack.empty())		// 深さ優先探索
    {
	vertex_t	u = stack.top();
	stack.pop();

	if (get(vertex_color, _g, u) != color_traits_t::black())
	{
	    put(vertex_color, _g, u, color_traits_t::black());

	    BOOST_FOREACH (edge_t e, out_edges(u, _g))
	    {
		vertex_t	v = target(e, _g);
		
		if ((get(vertex_color, _g, v) != color_traits_t::black()) &&
		    (get(edge_residual_capacity, _g, e) > 0))
		    stack.push(v);
	    }
	}
    }
}

template <class T, class ID, class L, class EL>
inline typename GraphCuts<T, ID, L, EL>::value_type
GraphCuts<T, ID, L, EL>::flow(edge_t e) const
{
    return get(edge_capacity, _g, e) - get(edge_residual_capacity, _g, e);
}

}
#endif	// !__TU_GRAPHCUTS_H
