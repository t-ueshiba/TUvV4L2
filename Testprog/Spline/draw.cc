/*
 *  $Id: draw.cc,v 1.1 2002-07-25 04:36:11 ueshiba Exp $
 */
#include "draw.h"

namespace TU
{
namespace v
{
OglDC&
operator <<(OglDC& dc, const BSplineCurve<float, 3u>& c)
{
    gluBeginCurve(dc.nurbsRenderer());
    gluNurbsCurve(dc.nurbsRenderer(),
		  c.M() + 1, (float*)c.knots(),
		  c.dim(), (float*)c, c.degree() + 1, GL_MAP1_VERTEX_3);
    gluEndCurve(dc.nurbsRenderer());

  // draw control polygon
    glPushAttrib(GL_LINE_BIT);
    glPushAttrib(GL_POINT_BIT);

    glEnable(GL_LINE_STIPPLE);
    glLineStipple(2, 0xaaaa);
    glBegin(GL_LINE_STRIP);
      for (int i = 0; i <= c.N(); ++i)
	  glVertex3fv((float*)c[i]);
    glEnd();
    glPointSize(3.0);
    glBegin(GL_POINTS);
      for (int i = 0; i <= c.N(); ++i)
	  glVertex3fv((float*)c[i]);
    glEnd();

    glPopAttrib();
    glPopAttrib();
    
    return dc;
}

OglDC&
operator <<(OglDC& dc, const RationalBSplineCurve<float, 3u>& c)
{
    gluBeginCurve(dc.nurbsRenderer());
    gluNurbsCurve(dc.nurbsRenderer(),
		  c.M() + 1, (float*)c.knots(),
		  c.dim(), (float*)c, c.degree() + 1, GL_MAP1_VERTEX_4);
    gluEndCurve(dc.nurbsRenderer());

  // draw control polygon
    glPushAttrib(GL_LINE_BIT);
    glPushAttrib(GL_POINT_BIT);

    glEnable(GL_LINE_STIPPLE);
    glLineStipple(2, 0xaaaa);
    glBegin(GL_LINE_STRIP);
      for (int i = 0; i <= c.N(); ++i)
	  glVertex4fv((float*)c[i]);
    glEnd();
    glPointSize(3.0);
    glBegin(GL_POINTS);
      for (int i = 0; i <= c.N(); ++i)
	  glVertex4fv((float*)c[i]);
    glEnd();

    glPopAttrib();
    glPopAttrib();

    return dc;
}

OglDC&
operator <<(OglDC& dc, const BSplineSurface<float>& s)
{
    gluBeginSurface(dc.nurbsRenderer());
    gluNurbsSurface(dc.nurbsRenderer(),
		    s.uM() + 1, (float*)s.uKnots(),
		    s.vM() + 1, (float*)s.vKnots(),
		    s.dim(), (s.uN() + 1) * s.dim(),
		    (float*)s, s.uDegree() + 1, s.vDegree() + 1,
		    GL_MAP2_VERTEX_3);
    gluEndSurface(dc.nurbsRenderer());

  // draw control polygon
    glPushAttrib(GL_LINE_BIT);
    glPushAttrib(GL_POINT_BIT);

    glEnable(GL_LINE_STIPPLE);
    glLineStipple(2, 0xaaaa);
    for (int j = 0; j <= s.vN(); ++j)
    {
	glBegin(GL_LINE_STRIP);
	glColor3f(1.0, 1.0, 1.0);
	for (int i = 0; i <= s.uN(); ++i)
	    glVertex3fv((float*)s[j][i]);
	glEnd();
	glPointSize(3.0);
	glBegin(GL_POINTS);
	for (int i = 0; i <= s.uN(); ++i)
	    glVertex3fv((float*)s[j][i]);
	glEnd();
    }
    for (int i = 0; i <= s.uN(); ++i)
    {
	glBegin(GL_LINE_STRIP);
	glColor3f(1.0, 1.0, 1.0);
	for (int j = 0; j <= s.vN(); ++j)
	    glVertex3fv((float*)s[j][i]);
	glEnd();
	glPointSize(3.0);
	glBegin(GL_POINTS);
	for (int j = 0; j <= s.vN(); ++j)
	    glVertex3fv((float*)s[j][i]);
	glEnd();
    }

    glPopAttrib();
    glPopAttrib();

    return dc;
}
 
}
}

#ifdef __GNUG__
#  include "TU/Array++.cc"
#  include "TU/Nurbs++.cc"
#endif
