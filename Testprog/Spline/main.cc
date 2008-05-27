/*
 *  $Id: main.cc,v 1.4 2008-05-27 11:38:25 ueshiba Exp $
 */
#include "TU/v/App.h"
#include "TU/v/CmdWindow.h"
#include "TU/v/CanvasPane.h"
#include "draw.h"
#ifdef __GNUG__
#  include "TU/Nurbs++.cc"
#endif

namespace TU
{
namespace v
{
/************************************************************************
*  class MyCanvasPane							*
************************************************************************/
class MyCanvasPane : public CanvasPane
{
  public:
    MyCanvasPane(Window& parentWin,
		 const BSplineCurve3f& b,
		 const BSplineCurve3f& bb,
		 const RationalBSplineCurve3f& c,
	 	 const BSplineSurface3f& s,
	 	 const BSplineSurface3f& ss)
	:CanvasPane(parentWin, 640, 480),
	 _dc(*this), _b(b), _bb(bb), _c(c), _s(s), _ss(ss)		{}
    
    virtual void	repaintUnderlay()				;

  protected:
    virtual void	initializeGraphics()				;
    
  private:
    enum	{BSplineCurv = 1, BSplineCurv1, RBSplineCurv,
		 BSplineSurf, BSplineSurf1, RBSplineSurf};
    
    OglDC				_dc;
    const BSplineCurve3f&		_b, _bb;
    const RationalBSplineCurve3f&	_c;
    const BSplineSurface3f&		_s, _ss;
};

void
MyCanvasPane::repaintUnderlay()
{
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    
      glCallList(BSplineCurv);
      glCallList(BSplineCurv1);
      glCallList(RBSplineCurv);
      glCallList(BSplineSurf);
    //      glCallList(BSplineSurf1);

    glFlush();
    _dc.swapBuffers();
}

void
MyCanvasPane::initializeGraphics()
{
    Vector<double>	t(3);
    Matrix<double>	Rt(3, 3);
    t[2] = 20.0;
    Rt[0][0] = 1.0;
    Rt[1][1] = Rt[2][2] = -1.0;
    _dc.setExternal(t, Rt) << distance(t[2]);
    
  //    glClearColor(0.3, 0.3, 0.3, 1.0);
    glEnable(GL_DEPTH_TEST);

    glEnable(GL_LIGHTING);
    glEnable(GL_LIGHT0);
    glEnable(GL_COLOR_MATERIAL);
    glEnable(GL_CULL_FACE);
    glEnable(GL_AUTO_NORMAL);

    GLfloat	position[] = {1.0, 1.0, 1.0, 0.0};
    glLightfv(GL_LIGHT0, GL_POSITION, position);

  /*    gluNurbsProperty(_dc.nurbsRenderer(),
	GLU_DISPLAY_MODE, GLU_OUTLINE_PATCH);*/
    gluNurbsProperty(_dc.nurbsRenderer(), GLU_DISPLAY_MODE, GLU_FILL);

    glNewList(BSplineCurv, GL_COMPILE);
      glColor3f(1.0, 0.0, 0.0);
      _dc << _b;
      glColor3f(1.0, 1.0, 1.0);
      glPointSize(3.0);
      glBegin(GL_POINTS);
	for (int i = _b.degree(); i <= _b.M() - _b.degree(); ++i)
	{
	    Vector3f	p = _b(_b.knots(i));
	    glVertex3fv((const float*)p);
	}
      glEnd();
    /*      for (float u = _b.knots(0); u <= _b.knots(_b.M()); u += 0.05)
      {
	  Array<Vector3f>	p = _b.derivatives(u, 1);
	  glBegin(GL_LINE_STRIP);
	    glVertex3fv((const float*)p[0]);
	    glVertex3f(p[0][0]+p[1][0], p[0][1]+p[1][1], p[0][2]+p[1][2]);
	  glEnd();
	  }*/
    glEndList();
    glNewList(BSplineCurv1, GL_COMPILE);
      glColor3f(1.0, 1.0, 0.0);
      _dc << _bb;
      glColor3f(1.0, 1.0, 1.0);
      glPointSize(3.0);
      glBegin(GL_POINTS);
	for (int i = _bb.degree(); i <= _bb.M() - _bb.degree(); ++i)
	{
	    Vector3f	p = _bb(_bb.knots(i));
	    glVertex3fv((const float*)p);
	}
      glEnd();
    glEndList();

    glNewList(RBSplineCurv, GL_COMPILE);
      glColor3f(0.0, 1.0, 0.0);
      _dc << _c;
      glColor3f(1.0, 1.0, 1.0);
      glPointSize(3.0);
      glBegin(GL_POINTS);
	for (int i = _c.degree(); i <= _c.M() - _c.degree(); ++i)
	{
	    Vector4f	p = _c(_c.knots()[i]);
	    glVertex4fv((const float*)p);
	}
      glEnd();
    glEndList();

    glNewList(BSplineSurf, GL_COMPILE);
      glColor3f(0.0, 0.0, 1.0);
      _dc << _s;
      glColor3f(1.0, 1.0, 1.0);
      glPointSize(3.0);
      glBegin(GL_POINTS);
	for (int j = _s.vDegree(); j <= _s.vM() - _s.vDegree(); ++j)
	    for (int i = _s.uDegree(); i <= _s.uM() - _s.uDegree(); ++i)
	    {
		Vector3f	p = _s(_s.uKnots(i), _s.vKnots(j));
		glVertex3fv((const float*)p);
	    }
      glEnd();
    glEndList();

    glNewList(BSplineSurf1, GL_COMPILE);
      glColor3f(0.0, 1.0, 1.0);
      _dc << _ss;
      glColor3f(1.0, 1.0, 1.0);
      glPointSize(3.0);
      glBegin(GL_POINTS);
	for (int j = _ss.vDegree(); j <= _ss.vM() - _ss.vDegree(); ++j)
	    for (int i = _ss.uDegree(); i <= _ss.uM() - _ss.uDegree(); ++i)
	    {
		Vector3f	p = _ss(_ss.uKnots(i), _ss.vKnots(j));
		glVertex3fv((const float*)p);
	    }
      glEnd();
    glEndList();
}

/************************************************************************
*  class MyCmdWindow							*
************************************************************************/
class MyCmdWindow : public CmdWindow
{
  public:
    MyCmdWindow(App&				parentApp,
		const char*			name,
		const XVisualInfo*		vinfo,
		const BSplineCurve3f&		b,
		const BSplineCurve3f&		bb,
	      	const RationalBSplineCurve3f&	c,
		const BSplineSurface3f&		s,
		const BSplineSurface3f&		ss)		;

  private:
    MyCanvasPane	_canvas;
};

MyCmdWindow::MyCmdWindow(App& parentApp, const char* name,
			 const XVisualInfo* vinfo,
			 const BSplineCurve3f& b,
			 const BSplineCurve3f& bb,
		       	 const RationalBSplineCurve3f& c,
			 const BSplineSurface3f& s,
			 const BSplineSurface3f& ss)
    :CmdWindow(parentApp, name, vinfo, Colormap::RGBColor, 16, 0, 0),
     _canvas(*this, b, bb, c, s, ss)
{
    show();
}
 
}
}
/************************************************************************
*  global functions							*
************************************************************************/
int
main(int argc, char* argv[])
{
    using namespace	std;
    using namespace	TU;
	
    v::App				vapp(argc, argv);

  /* Create a B-spline curve */
    BSplineCurve3f		b(3);
    b.insertKnot(0.25);
    b.insertKnot(0.5);
    b.insertKnot(0.75);
    b.insertKnot(0.75);
    
    b[0][0] = -4.0;    b[0][1] = -4.0;    b[0][2] =  0.0;
    b[1][0] = -3.0;    b[1][1] =  0.0;    b[1][2] =  1.0;
    b[2][0] = -1.0;    b[2][1] =  3.0;    b[2][2] =  0.0;
    b[3][0] =  0.0;    b[3][1] =  0.0;    b[3][2] = -1.0;
    b[4][0] =  0.0;    b[4][1] = -4.0;    b[4][2] =  0.0;
    b[5][0] =  3.0;    b[5][1] = -4.0;    b[5][2] =  1.0;
    b[6][0] =  5.0;    b[6][1] =  0.0;    b[6][2] =  0.0;
    b[7][0] =  4.0;    b[7][1] =  3.0;    b[7][2] = -1.0;

    b.insertKnot(0.75);
    
  /* Create a B-spline curve */
    BSplineCurve3f		bb(b);
  //  bb.removeKnot(6);
    bb.elevateDegree();
    
  /* Create a rational B-Spline curve */
    RationalBSplineCurve3f	c(3);
    c.insertKnot(0.25);
    c.insertKnot(0.5);
    c.insertKnot(0.75);
    c.insertKnot(0.75);
    
    c[0](0, 3) = b[0]; c[0][3] = 1;
    c[1](0, 3) = b[1]; c[1][3] = 1;
    c[2](0, 3) = b[2]; c[2][3] = 1;
    c[3](0, 3) = b[3]; c[3][3] = 1;
    c[4](0, 3) = b[4]; c[4][3] = 1;
    c[5](0, 3) = b[5]; c[5][3] = 1;
    c[6](0, 3) = b[6]; c[6][3] = 1;
    c[7](0, 3) = b[7]; c[7][3] = 1;
    c[4] *= 3.0;
    c[5] *= 2.0;
    c.insertKnot(0.25);
    
  /* Create a B-Spline surface */
    BSplineSurface3f	s(3, 2);
    s.uInsertKnot(0.33);
    s.uInsertKnot(0.66);
    s.vInsertKnot(0.33);
    s.vInsertKnot(0.66);
    s[0][0][0] = -5.0;    s[0][0][1] = -1.5;    s[0][0][2] =  2.0;
    s[0][1][0] = -3.0;    s[0][1][1] = -1.5;    s[0][1][2] =  1.0;
    s[0][2][0] = -1.0;    s[0][2][1] = -1.5;    s[0][2][2] = -0.5;
    s[0][3][0] =  1.0;    s[0][3][1] = -1.5;    s[0][3][2] = -0.5;
    s[0][4][0] =  3.0;    s[0][4][1] = -1.5;    s[0][4][2] =  1.0;
    s[0][5][0] =  5.0;    s[0][5][1] = -1.5;    s[0][5][2] =  2.0;

    s[1][0][0] = -4.0;    s[1][0][1] = -0.5;    s[1][0][2] =  1.0;
    s[1][1][0] = -2.4;    s[1][1][1] = -0.5;    s[1][1][2] =  0.5;
    s[1][2][0] = -0.8;    s[1][2][1] = -0.5;    s[1][2][2] =  0.0;
    s[1][3][0] =  0.8;    s[1][3][1] = -0.5;    s[1][3][2] =  0.0;
    s[1][4][0] =  2.4;    s[1][4][1] = -0.5;    s[1][4][2] =  0.5;
    s[1][5][0] =  4.0;    s[1][5][1] = -0.5;    s[1][5][2] =  1.0;

    s[2][0][0] = -3.5;    s[2][0][1] =  0.0;    s[2][0][2] =  0.0;
    s[2][1][0] = -2.1;    s[2][1][1] =  0.0;    s[2][1][2] =  0.0;
    s[2][2][0] = -0.7;    s[2][2][1] =  0.0;    s[2][2][2] =  0.0;
    s[2][3][0] =  0.7;    s[2][3][1] =  0.0;    s[2][3][2] =  0.0;
    s[2][4][0] =  2.1;    s[2][4][1] =  0.0;    s[2][4][2] =  0.0;
    s[2][5][0] =  3.5;    s[2][5][1] =  0.0;    s[2][5][2] =  0.0;

    s[3][0][0] = -4.0;    s[3][0][1] =  0.5;    s[3][0][2] = -1.0;
    s[3][1][0] = -2.4;    s[3][1][1] =  0.5;    s[3][1][2] = -0.5;
    s[3][2][0] = -0.8;    s[3][2][1] =  0.5;    s[3][2][2] =  0.0;
    s[3][3][0] =  0.8;    s[3][3][1] =  0.5;    s[3][3][2] =  0.0;
    s[3][4][0] =  2.4;    s[3][4][1] =  0.5;    s[3][4][2] = -0.5;
    s[3][5][0] =  4.0;    s[3][5][1] =  0.5;    s[3][5][2] = -1.0;

    s[4][0][0] = -5.0;    s[4][0][1] =  1.5;    s[4][0][2] = -2.0;
    s[4][1][0] = -3.0;    s[4][1][1] =  1.5;    s[4][1][2] = -1.0;
    s[4][2][0] = -1.0;    s[4][2][1] =  1.5;    s[4][2][2] =  0.5;
    s[4][3][0] =  1.0;    s[4][3][1] =  1.5;    s[4][3][2] =  0.5;
    s[4][4][0] =  3.0;    s[4][4][1] =  1.5;    s[4][4][2] = -1.0;
    s[4][5][0] =  5.0;    s[4][5][1] =  1.5;    s[4][5][2] = -2.0;
    
  /* Create a B-spline curve */
    BSplineSurface3f		ss(s);
  //    ss.vRemoveKnot(4);
    ss.vElevateDegree();
    
    int			attrs[] = {GLX_RGBA,
				   GLX_RED_SIZE,	1,
				   GLX_GREEN_SIZE,	1,
				   GLX_BLUE_SIZE,	1,
				   GLX_DEPTH_SIZE,	1,
				   GLX_DOUBLEBUFFER,
				   None};
    XVisualInfo*	vinfo = glXChooseVisual(vapp.colormap().display(),
						vapp.colormap().vinfo().screen,
						attrs);
    if (vinfo == 0)
    {
	cerr << "No appropriate visual!!" << endl;
	return 1;
    }

    v::MyCmdWindow	myWin0(vapp, "BSpline curve", vinfo, b, bb, c, s, ss);
    vapp.run();

    cerr << "Loop exited!" << endl;

    return 0;
}
