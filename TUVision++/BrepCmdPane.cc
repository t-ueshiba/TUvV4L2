/*
 *  $Id: BrepCmdPane.cc,v 1.3 2002-07-25 08:37:07 ueshiba Exp $
 */
#include "TU/v/Vision++.h"
#include <iomanip>
#include <sstream>

namespace TU
{
namespace v
{
    using namespace	Brep;

/************************************************************************
*  static functions							*
************************************************************************/
inline HalfEdge*
g2h(const Geometry* g)	{return (g != 0 ? g->parent() : 0);}

inline Loop*
h2l(const HalfEdge* h)	{return (h != 0 ? h->parent() : 0);}

static const HalfEdge*
n2h(const Loop* l, u_int n)
{
    const HalfEdge*	h = l->head();
    if (h != 0)
	do
	{
	    if (h->n == n)
		return h;
	} while ((h = h->next()) != l->head());

    for (Loop::ChildIterator iter(l); iter; ++iter)
	if ((h = n2h(iter, n)) != 0)
	    return h;
    return 0;
}

/************************************************************************
*  static variables							*
************************************************************************/
enum
{
    c_Frame,
    c_gp_txt, c_g_txt, c_gn_txt, c_pprev10, c_pprev, c_pnext, c_pnext10,
    c_hprev, c_hconj, c_hnext, c_hp_txt, c_h_txt, c_hn_txt, c_hc_txt
};

static CmdDef buttons[] =
{
    {C_Label,  c_gp_txt,  0, "",       noProp, CA_None, 0, 0, 2, 1, 0},
    {C_Label,  c_g_txt,   0, "",       noProp, CA_None, 2, 0, 1, 1, 0},
    {C_Label,  c_gn_txt,  0, "",       noProp, CA_None, 3, 0, 2, 1, 0},
    {C_Button, c_pprev10, 0, "<<",     noProp, CA_None, 0, 1, 1, 1, 0},
    {C_Button, c_pprev,   0, "<",      noProp, CA_None, 1, 1, 1, 1, 0},
    {C_Label,  c_hc_txt,  0, "",       noProp, CA_None, 2, 1, 1, 1, 0},
    {C_Button, c_pnext,   0, ">",      noProp, CA_None, 3, 1, 1, 1, 0},
    {C_Button, c_pnext10, 0, ">>",     noProp, CA_None, 4, 1, 1, 1, 0},
    {C_Button, c_hprev,   0, "prev",   noProp, CA_None, 0, 2, 2, 1, 0},
    {C_Button, c_hconj,   0, "conj",   noProp, CA_None, 2, 2, 1, 1, 0},
    {C_Button, c_hnext,   0, "next",   noProp, CA_None, 3, 2, 2, 1, 0},
    {C_Label,  c_hp_txt,  0, "",       noProp, CA_None, 0, 3, 2, 1, 0},
    {C_TextIn, c_h_txt,   0, "",       noProp, CA_None, 2, 3, 1, 1, 70},
    {C_Label,  c_hn_txt,  0, "",       noProp, CA_None, 3, 3, 2, 1, 0},
    EndOfCmds
};

static CmdDef frames[] =
{
    {C_Frame,	c_Frame, 0, "", buttons, CA_NoSpace, 0, 0, 1, 1, 0},
    EndOfCmds
};

/************************************************************************
*  class BrepCmdPane							*
************************************************************************/
BrepCmdPane::BrepCmdPane(Window& parentWindow, BrepCanvasPane& canvas)
    :CmdPane(parentWindow, frames), _g(0), _canvas(canvas)
{
    _canvas.setCmdPane(this);
}

void
BrepCmdPane::callback(CmdId id, CmdVal val)
{
    using namespace	std;

    if (_g != 0)
    {
	switch (id)
	{
	  case c_pprev10:
	  {
	    const Geometry*	g = _g;
	    for (int i = 0; i < 10; ++i)
		g = g->prev();
	    setGeometry(g);
	  }
	    break;

	  case c_pprev:
	    setGeometry(_g->prev());
	    break;
	
	  case c_pnext:
	    setGeometry(_g->next());
	    break;
	
	  case c_pnext10:
	  {
	    const Geometry*	g = _g;
	    for (int i = 0; i < 10; ++i)
		g = g->next();
	    setGeometry(g);
	  }
	    break;

	  case c_hprev:
	    setGeometry(_g->parent()->prev()->geom());
	    break;
	
	  case c_hnext:
	    setGeometry(_g->parent()->next()->geom());
	    break;
	
	  case c_hconj:
	    setGeometry(_g->parent()->conj()->geom());
	    break;
	}
    }

    if (id == c_h_txt)
    {
	const HalfEdge*	h = findHalfEdge(atoi(getString(c_h_txt)));
	if (h != 0)
	    setGeometry(h->geom());
	else if (_g != 0)
	{
	    ostringstream	s;
	    s << setw(4) << _g->parent()->n;
	    setString(c_h_txt, s.str().c_str());
	}
	else
	    setString(c_h_txt, "");
    }
}

void
BrepCmdPane::setGeometry(const Geometry* g)
{
    using namespace	std;
    
    const HalfEdge	*h_old = g2h(_g), *h = g->parent();
    const Loop	*l_old = h2l(h_old), *l = h->parent();
    if (l != l_old)
    {
	if (l_old != 0)
	    _canvas.draw(l_old, BrepCanvasPane::Draw);
	_canvas.draw(l, BrepCanvasPane::Highlight1);
	_canvas.draw(h, BrepCanvasPane::Highlight2);
	_canvas.draw(g, BrepCanvasPane::Highlight3);
    }
    else if (h != h_old)
    {
	if (h_old != 0)
	    _canvas.draw(h_old, BrepCanvasPane::Highlight1);
	_canvas.draw(h, BrepCanvasPane::Highlight2);
	_canvas.draw(g, BrepCanvasPane::Highlight3);
    }
    else if (g != _g)
    {
	if (_g != 0)
	    _canvas.draw(_g, BrepCanvasPane::Highlight2);
	_canvas.draw(g, BrepCanvasPane::Highlight3);
    }
    _canvas.sync();

    ostringstream	s;
    s << '(' << setw(3) << (*g->prev()->point())[0]
      << ',' << setw(3) << (*g->prev()->point())[1] << ')';
    setString(c_gp_txt, s.str().c_str());
    s.str("");
    s << '(' << setw(3) << (*g->point())[0]
      << ',' << setw(3) << (*g->point())[1] << ')';
    setString(c_g_txt, s.str().c_str());
    s.str("");
    s << '(' << setw(3) << (*g->next()->point())[0]
      << ',' << setw(3) << (*g->next()->point())[1] << ')';
    setString(c_gn_txt, s.str().c_str());
    s.str("");
    s << setw(4) << h->prev()->n;
    setString(c_hp_txt, s.str().c_str());
    s.str("");
    s << setw(4) << h->n;
    setString(c_h_txt, s.str().c_str());
    s.str("");
    s << setw(4) << h->next()->n;
    setString(c_hn_txt, s.str().c_str());
    s.str("");
    s << setw(4) << h->conj()->n;
    setString(c_hc_txt, s.str().c_str());

    _g = g;
}

const HalfEdge*
BrepCmdPane::findHalfEdge(u_int n) const
{
    return n2h(_canvas.root(), n);
}
 
}
}
