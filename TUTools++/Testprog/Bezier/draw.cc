/*
 *  $Id: draw.cc,v 1.3 2012-08-15 07:58:34 ueshiba Exp $
 */
#include "draw.h"

namespace TU
{
namespace v
{
OglDC&
operator <<(OglDC& dc, const BezierCurve3d& b)
{
    glPushAttrib(GL_EVAL_BIT);
    glEnable(GL_MAP1_VERTEX_3);
    glMap1d(GL_MAP1_VERTEX_3,
	    0.0, 1.0, b.dim(), b.degree()+1, (const double*)b);
    glBegin(GL_LINE_STRIP);
      for (int sample = 0; sample <= 20; ++sample)
	  glEvalCoord1f((GLfloat)sample/20.0);
    glEnd();
    glPopAttrib();
    
  // draw control polygon
    glPushAttrib(GL_LINE_BIT);
    glEnable(GL_LINE_STIPPLE);
    glLineStipple(2, 0xaaaa);
    glBegin(GL_LINE_STRIP);
      for (u_int i = 0; i <= b.degree(); ++i)
	  glVertex3dv((const double*)b[i]);
    glEnd();
    glPopAttrib();

    return dc;
}

OglDC&
operator <<(OglDC& dc, const RationalBezierCurve3d& b)
{
    glPushAttrib(GL_EVAL_BIT);
    glEnable(GL_MAP1_VERTEX_4);
    glMap1d(GL_MAP1_VERTEX_4,
	    0.0, 1.0, b.dim(), b.degree()+1, (const double*)b);
    glBegin(GL_LINE_STRIP);
      for (int sample = 0; sample <= 20; ++sample)
	  glEvalCoord1f((GLfloat)sample/20.0);
    glEnd();
    glPopAttrib();
    
  // draw control polygon
    glPushAttrib(GL_LINE_BIT);
    glEnable(GL_LINE_STIPPLE);
    glLineStipple(2, 0xaaaa);
    glBegin(GL_LINE_STRIP);
      for (u_int i = 0; i <= b.degree(); ++i)
	  glVertex4dv((const double*)b[i]);
    glEnd();
    glPopAttrib();
    
    return dc;
}
/*
OglDC&
operator <<(OglDC& dc, const BezierSurface3d& b)
{
    glPushAttrib(GL_EVAL_BIT);
    glEnable(GL_MAP2_VERTEX_3);
    glMap2d(GL_MAP2_VERTEX_3,
	    0.0, 1.0,			b.dim(), b.uDegree()+1,
	    0.0, 1.0, (b.uDegree()+1) * b.dim(), b.vDegree()+1, (const double*)b);
    for (int mesh = 0; mesh <= 8; ++mesh)
    {
      glBegin(GL_LINE_STRIP);
	for (int sample = 0; sample <= 20; ++sample)
	    glEvalCoord2f((GLfloat)sample/20.0, (GLfloat)mesh/8.0);
      glEnd();
      glBegin(GL_LINE_STRIP);
	for (sample = 0; sample <= 20; ++sample)
	    glEvalCoord2f((GLfloat)mesh/8.0, (GLfloat)sample/20.0);
      glEnd();
    }
    glPopAttrib();
    
  // draw control polygon
    glPushAttrib(GL_LINE_BIT);
    glEnable(GL_LINE_STIPPLE);
    glLineStipple(2, 0xaaaa);
    for (int j = 0; j <= b.vDegree(); ++j)
    {
	glBegin(GL_LINE_STRIP);
	for (int i = 0; i <= b.uDegree(); ++i)
	    glVertex3dv((const double*)b[j][i]);
	glEnd();
    }
    for (int i = 0; i <= b.uDegree(); ++i)
    {
	glBegin(GL_LINE_STRIP);
	for (int j = 0; j <= b.vDegree(); ++j)
	    glVertex3dv((const double*)b[j][i]);
	glEnd();
    }
    glPopAttrib();
    
    return dc;
}
*/
OglDC&
operator <<(OglDC& dc, const BezierSurface3d& b)
{
    glPushAttrib(GL_EVAL_BIT);
    glEnable(GL_MAP2_VERTEX_3);
    glMap2d(GL_MAP2_VERTEX_3,
	    0.0, 1.0,			b.dim(), b.uDegree()+1,
	    0.0, 1.0, (b.uDegree()+1) *	b.dim(), b.vDegree()+1,
	    (const double*)b);
    glMapGrid2d(20, 0.0, 1.0, 20, 0.0, 1.0);
    glFrontFace(GL_CW);
    glEvalMesh2(GL_FILL, 0, 20, 0, 20);
    glFrontFace(GL_CCW);
    glPopAttrib();
    
  // draw control polygon
    glPushAttrib(GL_LINE_BIT);
    glEnable(GL_LINE_STIPPLE);
    glLineStipple(2, 0xaaaa);
    for (u_int j = 0; j <= b.vDegree(); ++j)
    {
	glBegin(GL_LINE_STRIP);
	for (u_int i = 0; i <= b.uDegree(); ++i)
	    glVertex3dv((const double*)b[j][i]);
	glEnd();
    }
    for (u_int i = 0; i <= b.uDegree(); ++i)
    {
	glBegin(GL_LINE_STRIP);
	for (u_int j = 0; j <= b.vDegree(); ++j)
	    glVertex3dv((const double*)b[j][i]);
	glEnd();
    }
    glPopAttrib();
    
    return dc;
}
 
}
}
