/*
 *  $Id: BrepCanvasPane.cc,v 1.6 2012-08-29 21:17:14 ueshiba Exp $
 */
#include "TU/v/Vision++.h"
#include <limits.h>

namespace TU
{
namespace v
{
    using namespace	Brep;
	
/************************************************************************
*  class BrepCanvasPane							*
************************************************************************/
BrepCanvasPane::BrepCanvasPane(Window& parentWindow,
			       u_int width, u_int height, Root* root)
    :CanvasPane(parentWindow, width, height),
     _dc(*this), _root(root), _h_cache(0), _cmd(0)
{
#ifdef UseOverlay
    _dc << overlay;
#endif
}

BrepCanvasPane::~BrepCanvasPane()
{
}

void
BrepCanvasPane::callback(CmdId id, CmdVal val)
{
    switch (id)
    {
      case Id_MouseMove:
      {
	if (_h_cache != 0)
	{
	    const Geometry*	g = _cmd->getGeometry();
	    DrawMode		mode = Draw;
	    if (g == 0)
		mode = Draw;
	    else if (_h_cache == g->parent())
		mode = Highlight2;
	    else if (_h_cache->parent() == g->parent()->parent())
		mode = Highlight1;
	    draw(_h_cache, mode);
	    if (g != 0)
		draw(g, Highlight3);
	}
	const Geometry*	g = findGeometry(_dc.dev2logU(val.u),
					 _dc.dev2logV(val.v));
	if (g != 0)
	{
	    _h_cache = g->parent();
	    draw(_h_cache, Highlight3);
	    draw(g, Highlight3);
	}
	else
	    _h_cache = 0;
	sync();
      }
	break;

      case Id_MouseButton1Press:
      {
	const Geometry*	g = findGeometry(_dc.dev2logU(val.u),
					 _dc.dev2logV(val.v));
	if (g != 0)
	    _cmd->setGeometry(g);
      }
	break;
    }
}

void
BrepCanvasPane::drawDescendants(const Loop* l, DrawMode mode)
{
    draw(l, mode);
    for (Loop::ChildIterator iter(l); iter; ++iter)
	drawDescendants(iter, mode);
}

void
BrepCanvasPane::draw(const Loop* l, DrawMode mode)
{
    HalfEdge* h = l->head();
    if (h)
	do
	{
	    draw(h, mode);
	} while ((h = h->next()) != l->head());
}

void
BrepCanvasPane::draw(const HalfEdge* h, DrawMode mode)
{
#ifdef UseOverlay
    u_int	jnc = Color_BG, pnt = Color_BG;
#else
    BGR		jnc = Color_BG, pnt = Color_BG;
#endif
    switch (mode)
    {
      case Draw:
	jnc = Color_RED;
	pnt = Color_YELLOW;
	break;
      case Highlight1:
	jnc = Color_RED;
	pnt = Color_CYAN;
	break;
      case Highlight2:
	jnc = Color_BLUE;
	pnt = Color_MAGENDA;
	break;
      case Highlight3:
	jnc = Color_BLUE;
	pnt = Color_GREEN;
	break;
    }
    
    _dc << foreground(jnc);
    if (mode != Erase || h->juncnum() == 0)
	_dc << *h->point() << foreground(pnt);
    for (Geometry* g = h->geom(); (g = g->cdr())->consp(); )
	_dc << *g->point();
}

void
BrepCanvasPane::draw(const Geometry* g, DrawMode mode)
{
#ifdef UseOverlay
    u_int	pnt = Color_BG;
#else
    BGR		jnc = Color_BG, pnt = Color_BG;
#endif
    switch (mode)
    {
      case Draw:
	pnt = (g->parent()->geom() == g ? Color_RED : Color_YELLOW);
	break;
      case Highlight1:
	pnt = (g->parent()->geom() == g ? Color_RED : Color_CYAN);
	break;
      case Highlight2:
	pnt = (g->parent()->geom() == g ? Color_BLUE : Color_MAGENDA);
	break;
      case Highlight3:
	pnt = Color_GREEN;
	break;
    }
    _dc << foreground(pnt) << *g->point();
}

void
#ifdef UseOverlay
BrepCanvasPane::repaintOverlay()
#else
BrepCanvasPane::repaintUnderlay()
#endif
{
    _dc << clear;
    drawDescendants(_root, Draw);
    const Geometry*	g = _cmd->getGeometry();
    if (g != 0)
    {
	const HalfEdge*	h = g->parent();
	const Loop*	l = h->parent();
	draw(l, Highlight1);
	draw(h, Highlight2);
	draw(g, Highlight3);
    }
}

void
#ifdef UseOverlay
BrepCanvasPane::repaintUnderlay()
#else
BrepCanvasPane::repaintOverlay()
#endif
{
}

Geometry*
BrepCanvasPane::findGeometry(int u, int v) const
{
    Loop*	l = (_h_cache != 0 ? _h_cache->parent() : _root);
    Geometry*	g = l->p2g(u, v);
    return g;
}
 
}
}
