/*
 *  $Id$
 */
#ifndef __TU_TREEFILTER_H
#define __TU_TREEFILTER_H

#include <iostream>
#include <cmath>
#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/filtered_graph.hpp>
#include <boost/graph/prim_minimum_spanning_tree.hpp>
#include <boost/graph/depth_first_search.hpp>
#include <boost/graph/graphviz.hpp>
#include <boost/foreach.hpp>
#include "TU/Profiler.h"

namespace boost
{
enum vertex_weight_t		{vertex_weight};
enum vertex_tree_val_t		{vertex_tree_val};
enum vertex_aggr_val_t		{vertex_aggr_val};
enum vertex_tree_weight_t	{vertex_tree_weight};
enum vertex_aggr_weight_t	{vertex_aggr_weight};
enum vertex_upward_edge_t	{vertex_upward_edge};
enum edge_tree_t		{edge_tree};
BOOST_INSTALL_PROPERTY(vertex, weight);
BOOST_INSTALL_PROPERTY(vertex, tree_val);
BOOST_INSTALL_PROPERTY(vertex, aggr_val);
BOOST_INSTALL_PROPERTY(vertex, tree_weight);
BOOST_INSTALL_PROPERTY(vertex, aggr_weight);
BOOST_INSTALL_PROPERTY(vertex, upward_edge);
BOOST_INSTALL_PROPERTY(edge,   tree);
    
/************************************************************************
*  class TreeFilter<T, W, CLOCK>					*
************************************************************************/
template <class T, class W, class CLOCK=void>
class TreeFilter : public TU::Profiler<CLOCK>
{
  public:
    typedef T							value_type;
    typedef typename W::argument_type				guide_type;
    typedef typename W::result_type				weight_type;
    
  private:
    typedef TU::Profiler<CLOCK>					pf_type;
    typedef adjacency_list_traits<vecS, vecS, undirectedS>	traits_t;
    typedef traits_t::vertex_descriptor				vertex_t;
    typedef traits_t::edge_descriptor				edge_t;
    typedef property<vertex_predecessor_t, vertex_t,
	    property<vertex_weight_t,	   weight_type,
	    property<vertex_tree_val_t,	   value_type,
	    property<vertex_aggr_val_t,	   value_type,
	    property<vertex_tree_weight_t, weight_type,
	    property<vertex_aggr_weight_t, weight_type> > > > > >
								VertexProps;
    typedef property<edge_weight_t,	   weight_type,
	    property<edge_tree_t,	   bool> >		EdgeProps;
    typedef adjacency_list<vecS, vecS, undirectedS,
			   VertexProps, EdgeProps>		grid_t;

    struct IsTreeEdge
    {
	typedef typename property_map<grid_t, edge_tree_t>::type TreeEdgeMap;

		IsTreeEdge()					{}
		IsTreeEdge(TreeEdgeMap treeEdgeMap)
		    :_treeEdgeMap(treeEdgeMap)			{}
	
	bool	operator ()(edge_t e)	const	{return get(_treeEdgeMap, e);}

      private:
	TreeEdgeMap	_treeEdgeMap;
    };

    typedef filtered_graph<grid_t, IsTreeEdge>			tree_t;

    struct TreeExtractor : public default_dijkstra_visitor
    {
		TreeExtractor(weight_type sigma) :_nrsigma(-1/sigma)	{}
#  ifdef PRINT
	void	examine_vertex(vertex_t v, const grid_t& g) const
		{
		    std::cerr << "TreeExtractor::examine_vertex:\t"
			      << v << std::endl;
		}
	void	edge_relaxed(edge_t e, const grid_t& g) const
		{
		    std::cerr << "TreeExtractor::edge_relaxed:\t";
		    printEdge(std::cerr, g, e) << std::endl;
		}
#  endif
	void	finish_vertex(vertex_t v, const grid_t& g) const
		{
		    auto	u = get(vertex_predecessor, g, v);

		    if (u != v)
		    {
			auto	e = edge(u, v, g).first;
			put(edge_tree,	   const_cast<grid_t&>(g), e, true);
			put(vertex_weight, const_cast<grid_t&>(g), v,
			    std::exp(get(edge_weight, g, e) * _nrsigma));
		    }
#  ifdef PRINT
		    std::cerr << "TreeExtractor::finish_vertex:\t"
			      << v << std::endl;
#  endif
		}

      private:
	const weight_type	_nrsigma;
    };

    struct TreeInitializer : public default_dfs_visitor
    {
#  ifdef PRINT
	void	discover_vertex(vertex_t v, const tree_t& g) const
		{
		    std::cerr << "TreeInitializer::discover_vertex: ";
		    printVertex(std::cerr , g, v) << std::endl;
		}
	void	tree_edge(edge_t e, const tree_t& g) const
		{
		    std::cerr << "TreeInitializer::tree_edge:\t  ";
		    printEdge(std::cerr, g, e) << std::endl;
		}
#  endif
	void	finish_vertex(vertex_t v, const tree_t& g) const
		{
		    const auto	u = get(vertex_predecessor, g, v);

		    if (u != v)
		    {
			auto	w = get(vertex_weight, g, v);
			auto	tree_vals = get(vertex_tree_val,
						const_cast<tree_t&>(g));
			tree_vals[u] += w * tree_vals[v];
		    }
#  ifdef PRINT
		    std::cerr << "TreeInitializer::finish_vertex:\t  ";
		    printVertex(std::cerr, g, v) << "\t=> ";
		    if (u != v)
			printVertex(std::cerr, g, u) << std::endl;
		    else
			std::cerr << "root" << std::endl;
#  endif
		}
    };

    struct TreeInitializerWithNormalization : public default_dfs_visitor
    {
	void	finish_vertex(vertex_t v, const tree_t& g) const
		{
		    const auto	u = get(vertex_predecessor, g, v);

		    if (u != v)
		    {
			auto	w = get(vertex_weight, g, v);
			auto	tree_vals = get(vertex_tree_val,
						const_cast<tree_t&>(g));
			auto	tree_weights = get(vertex_tree_weight,
						   const_cast<tree_t&>(g));
			tree_vals[u] += w * tree_vals[v];
			tree_weights[u] += w * tree_weights[v];
		    }
		}
    };
    
    struct TreeAggregator : public default_dfs_visitor
    {
	void	start_vertex(vertex_t v, const tree_t& g) const
		{
		    put(vertex_aggr_val, g, v, get(vertex_tree_val, g, v));
#  ifdef PRINT
		    std::cerr << "TreeAggregator::start_vertex:\t";
		    printVertex(std::cerr, g, v) << std::endl;
#  endif
		}
	void	tree_edge(edge_t e, const tree_t& g) const
		{
		    auto	u = source(e, g);
		    auto	v = target(e, g);
		    if (get(vertex_predecessor, g, v) != u)
			std::swap(u, v);

		    auto	w = get(vertex_weight, g, v);
		    auto	tree_vals = get(vertex_tree_val,
						const_cast<tree_t&>(g));
		    auto	aggr_vals = get(vertex_aggr_val,
						const_cast<tree_t&>(g));
		    tree_vals[u] = aggr_vals[u] - w * tree_vals[v];
		    aggr_vals[v] = tree_vals[v] + w * tree_vals[u];
#  ifdef PRINT
		    std::cerr << "TreeAggregator::tree_edge:\t";
		    printVertex(std::cerr, g, u) << "\t<= ";
		    printVertex(std::cerr, g, v) << std::endl;
#  endif
		}
	void	finish_vertex(vertex_t v, const tree_t& g) const
		{
		    const auto	u = get(vertex_predecessor, g, v);

		    if (u != v)
		    {
			auto	w = get(vertex_weight, g, v);
			auto	tree_vals = get(vertex_tree_val,
						const_cast<tree_t&>(g));
			auto	aggr_vals = get(vertex_aggr_val,
						const_cast<tree_t&>(g));
			tree_vals[v] = aggr_vals[v] - w * tree_vals[u];
		      //aggr_vals[u] = tree_vals[u] + w * tree_vals[v];
		    }
#  ifdef PRINT
		    std::cerr << "TreeAggregator::finish_vertex:\t";
		    printVertex(std::cerr, g, v) << "\t=> ";
		    if (u != v)
			printVertex(std::cerr, g, u) << std::endl;
		    else
			std::cerr << "root" << std::endl;
#  endif
		}
    };

    struct TreeAggregatorWithNormalization : public default_dfs_visitor
    {
	void	start_vertex(vertex_t v, const tree_t& g) const
		{
		    put(vertex_aggr_val,    g, v, get(vertex_tree_val, g, v));
		    put(vertex_aggr_weight, g, v, get(vertex_tree_weight,
						      g, v));
		}
	void	tree_edge(edge_t e, const tree_t& g) const
		{
		    auto	u = source(e, g);
		    auto	v = target(e, g);
		    if (get(vertex_predecessor, g, v) != u)
			std::swap(u, v);

		    const auto	w = get(vertex_weight, g, v);
		    auto	tree_vals = get(vertex_tree_val,
						const_cast<tree_t&>(g));
		    auto	aggr_vals = get(vertex_aggr_val,
						const_cast<tree_t&>(g));
		    tree_vals[u] = aggr_vals[u] - w * tree_vals[v];
		    aggr_vals[v] = tree_vals[v] + w * tree_vals[u];
		    auto	tree_weights = get(vertex_tree_weight,
						   const_cast<tree_t&>(g));
		    auto	aggr_weights = get(vertex_aggr_weight,
						   const_cast<tree_t&>(g));
		    tree_weights[u] = aggr_weights[u] - w * tree_weights[v];
		    aggr_weights[v] = tree_weights[v] + w * tree_weights[u];
		}
	void	finish_vertex(vertex_t v, const tree_t& g) const
		{
		    const auto	u = get(vertex_predecessor, g, v);

		    if (u != v)
		    {
			auto	w = get(vertex_weight, g, v);
			auto	tree_vals = get(vertex_tree_val,
						const_cast<tree_t&>(g));
			auto	aggr_vals = get(vertex_aggr_val,
						const_cast<tree_t&>(g));
			tree_vals[v] = aggr_vals[v] - w * tree_vals[u];
			auto	tree_weights = get(vertex_tree_weight,
						   const_cast<tree_t&>(g));
			auto	aggr_weights = get(vertex_aggr_weight,
						   const_cast<tree_t&>(g));
			tree_weights[v] = aggr_weights[v] - w * tree_weights[u];
		    }
		}
    };
    
  public:
		TreeFilter(const W& wfunc, weight_type sigma)
		    :pf_type(6), _wfunc(wfunc),
		     _sigma(sigma), _nrow(0), _ncol(0)	{}

    size_t	nrow()				const	{ return _nrow; }
    size_t	ncol()				const	{ return _ncol; }
    weight_type	sigma()				const	{ return _sigma; }
    void	setSigma(weight_type sigma)		{ _sigma = sigma; }
    template <class ROW_I, class ROW_G, class ROW_O>
    void	convolve(ROW_I rowI, ROW_I rowIe, ROW_G rowG, ROW_G rowGe,
			 ROW_O rowO, bool normalize=false)		;
    void	printVertices(std::ostream& out)		const	;
    void	printEdges(std::ostream& out)			const	;
    void	saveGrid(std::ostream& out)			const	;
    void	saveTree(std::ostream& out)			const	;

  private:
    void	resize(size_t nrows, size_t ncols)			;
    template <class ROW_I>
    void	initializeVertices(ROW_I rowI, ROW_I rowIe)		;
    template <class ROW_G>
    void	initializeEdges(ROW_G rowG, ROW_G rowGe)		;
    template <class ROW_O>
    void	outputResults(ROW_O rowO)			const	;
    template <class ROW_O>
    void	outputNormalizedResults(ROW_O rowO)		const	;
    void	initializeVertex(vertex_t v, const value_type& val)
		{
		    put(vertex_tree_val,    _grid, v, val);
		    put(vertex_aggr_val,    _grid, v, 0);
		    put(vertex_tree_weight, _grid, v, 1);
		    put(vertex_aggr_weight, _grid, v, 0);
		}
    void	initializeEdge(edge_t e, weight_type val)
		{
		    put(edge_weight, _grid, e, val);
		    put(edge_tree,   _grid, e, false);
		}
    vertex_t	vidx(size_t r, size_t c) const
		{
		    return r*_ncol + c;
		}
    template <class G> static std::ostream&
		printVertex(std::ostream& out, const G& g, vertex_t v)
		{
		    return out << v << ":("
		      //<< get(vertex_predecessor, g, v) << ", "
			       << get(vertex_tree_val, g, v) << ", "
			       << get(vertex_aggr_val, g, v) << ')';
		}
    template <class G> static std::ostream&
		printEdge(std::ostream& out, const G& g, edge_t e)
		{
		    return out << source(e, g) << "-("
			       << get(edge_weight, g, e) << ")-"
			       << target(e, g);
		}

  private:
    const W&	_wfunc;
    weight_type	_sigma;
    size_t	_nrow;
    size_t	_ncol;
    grid_t	_grid;
};

template <class T, class W, class CLOCK>
template <class ROW_I, class ROW_G, class ROW_O> void
TreeFilter<T, W, CLOCK>::convolve(ROW_I rowI, ROW_I rowIe, ROW_G rowG,
				  ROW_G rowGe, ROW_O rowO, bool normalize)
{
    pf_type::start(0);
    const size_t	nrows = std::distance(rowI, rowIe);
    const size_t	ncols = (nrows != 0 ? rowI->size() : 0);

    if (nrows == 0 || ncols == 0)
	return;

    resize(nrows, ncols);

    pf_type::start(1);
    initializeVertices(rowI, rowIe);
    initializeEdges(rowG, rowGe);

    pf_type::start(2);
    prim_minimum_spanning_tree(_grid, get(vertex_predecessor, _grid),
			       visitor(TreeExtractor(_sigma)));
    tree_t	tree(_grid, IsTreeEdge(get(edge_tree, _grid)));

    if (normalize)
    {
	pf_type::start(3);
	depth_first_search(tree, visitor(TreeInitializerWithNormalization()));

	pf_type::start(4);
	depth_first_search(tree, visitor(TreeAggregatorWithNormalization()));
	
	pf_type::start(5);
	outputNormalizedResults(rowO);
    }
    else
    {
	pf_type::start(3);
	depth_first_search(tree, visitor(TreeInitializer()));

	pf_type::start(4);
	depth_first_search(tree, visitor(TreeAggregator()));
	
	pf_type::start(5);
	outputResults(rowO);
    }

    pf_type::nextFrame();
}

template <class T, class W, class CLOCK> void
TreeFilter<T, W, CLOCK>::resize(size_t nrows, size_t ncols)
{
    if (nrows == _nrow && ncols == _ncol)
	return;
    
    _nrow = nrows;
    _ncol = ncols;
    _grid.clear();

    for (size_t n = _nrow * _ncol; n > 0; --n)
	add_vertex(_grid);

    for (size_t c = 1; c < ncol(); ++c)
	add_edge(c-1, c, _grid);

    for (size_t r = 1; r < nrow(); ++r)
    {
	add_edge(vidx(r-1, 0), vidx(r, 0), _grid);
	
	for (size_t c = 1; c < ncol(); ++c)
	{
	    add_edge(vidx(r-1, c), vidx(r, c), _grid);
	    add_edge(vidx(r, c-1), vidx(r, c), _grid);
	}
    }
}
    
template <class T, class W, class CLOCK> template <class ROW_I> void
TreeFilter<T, W, CLOCK>::initializeVertices(ROW_I rowI, ROW_I rowIe)
{
    auto	v = vertices(_grid).first;

    for (; rowI != rowIe; ++rowI)
	for (auto colI = rowI->begin(); colI != rowI->end(); ++colI, ++v)
	    initializeVertex(*v, *colI);
}
    
template <class T, class W, class CLOCK> template <class ROW_G> void
TreeFilter<T, W, CLOCK>::initializeEdges(ROW_G rowG, ROW_G rowGe)
{
    auto	e = edges(_grid).first;
    auto	colG = rowG->begin();

    for (auto colL = colG; ++colG != rowG->end(); colL = colG)
	initializeEdge(*e++, _wfunc(*colL, *colG));

    for (auto rowU = rowG; ++rowG != rowGe; rowU = rowG)
    {
	auto	colU = rowU->begin();
	auto	colG = rowG->begin();
	
	initializeEdge(*e++, _wfunc(*colU, *colG));
	
	for (auto colL = colG; ++colG != rowG->end(); colL = colG)
	{
	    initializeEdge(*e++, _wfunc(*++colU, *colG));
	    initializeEdge(*e++, _wfunc(*colL, *colG));
	}
    }
}

template <class T, class W, class CLOCK> template <class ROW_O> void
TreeFilter<T, W, CLOCK>::outputResults(ROW_O rowO) const
{
    auto	v = vertices(_grid).first;
    auto	vals = get(vertex_aggr_val, _grid);
    
    for (auto r = nrow(); r > 0; --r, ++rowO)
    {
	auto	colO = rowO->begin();

	for (auto c = ncol(); c > 0; --c, ++colO, ++v)
	    *colO = vals[*v];
    }
}
    
template <class T, class W, class CLOCK> template <class ROW_O> void
TreeFilter<T, W, CLOCK>::outputNormalizedResults(ROW_O rowO) const
{
    auto	v	= vertices(_grid).first;
    auto	vals	= get(vertex_aggr_val,    _grid);
    auto	weights = get(vertex_aggr_weight, _grid);
    
    for (auto r = nrow(); r > 0; --r, ++rowO)
    {
	auto	colO = rowO->begin();

	for (auto c = ncol(); c > 0; --c, ++colO, ++v)
	    *colO = vals[*v] / weights[*v];
    }
}
    
template <class T, class W, class CLOCK> void
TreeFilter<T, W, CLOCK>::printVertices(std::ostream& out) const
{
    BOOST_FOREACH (vertex_t v, vertices(_grid))
	printVertex(out, _grid, v) << std::endl;
}

template <class T, class W, class CLOCK> void
TreeFilter<T, W, CLOCK>::printEdges(std::ostream& out) const
{
    BOOST_FOREACH (edge_t e, edges(_grid))
	printEdge(out, _grid, e) << std::endl;
}

template <class T, class W, class CLOCK> void
TreeFilter<T, W, CLOCK>::saveGrid(std::ostream& out) const
{
    write_graphviz(out, _grid);
}

}
#endif	// !__TU_TREEFILTER_H
