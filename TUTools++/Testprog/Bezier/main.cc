/*
 *  $Id: main.cc,v 1.4 2008-05-27 11:38:25 ueshiba Exp $
 */
#include "TU/v/App.h"
#include "TU/v/CmdWindow.h"
#include "TU/v/CanvasPane.h"
#include "draw.h"
#ifdef __GNUG__
#  include "TU/Bezier++.cc"
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
    MyCanvasPane(Window&			parentWin,
		 const BezierCurve3d&		b,
		 const RationalBezierCurve3d&	c,
		 const BezierSurface3d&		s)
	:CanvasPane(parentWin, 640, 480),
	 _dc(*this), _b(b), _c(c), _s(s)				{}
    
    virtual void	repaintUnderlay()				;

  protected:
    virtual void	initializeGraphics()				;
    
  private:
    enum	{BezierCurv = 1, RBezierCurv, BezierSurf, RBezierSurf};
    
    OglDC				_dc;
    const BezierCurve3d&		_b;
    const RationalBezierCurve3d&	_c;
    const BezierSurface3d&		_s;
};

void
MyCanvasPane::repaintUnderlay()
{
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    
    glCallList(BezierCurv);
    glCallList(RBezierCurv);
    glCallList(BezierSurf);
    
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
    
    glNewList(BezierCurv, GL_COMPILE);
      glColor3f(1.0, 0.0, 0.0);
      _dc << _b;

      glColor3f(1.0, 1.0, 1.0);
      glPointSize(3.0);
      glBegin(GL_POINTS);
	for (int i = 0; i <= 20; ++i)
	{
	    Vector3d	p = _b(i / 20.0);
	    glVertex3dv((const double*)p);
	}
      glEnd();

      glColor3f(0.0, 0.0, 1.0);
	for (int r = 1; r <= _b.degree(); ++r)
	{
	    glBegin(GL_LINE_STRIP);
	    Array<Vector3d>	p = _b.deCasteljau(0.3, r);
	    for (int i = 0; i < p.dim(); ++i)
		glVertex3dv((const double*)p[i]);
	    glEnd();
	}
    glEndList();

    glNewList(RBezierCurv, GL_COMPILE);
      glColor3f(0.0, 1.0, 0.0);
      _dc << _c;

      glColor3f(1.0, 1.0, 1.0);
      glPointSize(3.0);
      glBegin(GL_POINTS);
	for (int i = 0; i <= 20; ++i)
	{
	    Vector4d	p = _c(i / 20.0);
	    glVertex4dv((const double*)p);
	}
      glEnd();

      glColor3f(0.0, 1.0, 1.0);
	for (int r = 1; r <= _c.degree(); ++r)
	{
	    glBegin(GL_LINE_STRIP);
	    Array<Vector4d>	p = _c.deCasteljau(0.3, r);
	    for (int i = 0; i < p.dim(); ++i)
		glVertex4dv((const double*)p[i]);
	    glEnd();
	}
    glEndList();

    glNewList(BezierSurf, GL_COMPILE);
      glColor3f(0.0, 0.0, 1.0);
      _dc << _s;

    glColor3f(1.0, 1.0, 1.0);
      glPointSize(3.0);
      glBegin(GL_POINTS);
	for (int j = 0; j <= 20; ++j)
	    for (int i = 0; i <= 20; ++i)
	    {
		Vector3d	p = _s(i / 20.0, j / 20.0);
		glVertex3dv((const double*)p);
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
		const BezierCurve3d&		b,
		const RationalBezierCurve3d&	c,
		const BezierSurface3d&		s)			;

  private:
    MyCanvasPane	_canvas;
};

MyCmdWindow::MyCmdWindow(App& parentApp, const char*	name,
			 const XVisualInfo*		vinfo,
			 const BezierCurve3d&		b,
			 const RationalBezierCurve3d&	c,
			 const BezierSurface3d&		s)
    :CmdWindow(parentApp, name, vinfo, Colormap::RGBColor, 16, 0, 0),
     _canvas(*this, b, c, s)
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
    using namespace std;
    using namespace TU;
	
    v::App	vapp(argc, argv);

  /* Create a Bezier curve */
    BezierCurve3d		b(3);
    b[0][0] = -4.0;    b[0][1] = -4.0;    b[0][2] =  0.0;
    b[1][0] = -2.0;    b[1][1] =  4.0;    b[1][2] =  5.0;
    b[2][0] =  2.0;    b[2][1] = -4.0;    b[2][2] = -1.0;
    b[3][0] =  4.0;    b[3][1] =  4.0;    b[3][2] =  0.0;

  /* Create a rational Bezier curve */
    RationalBezierCurve3d	c(3);
    c[0](0, 3) = b[0]; c[0][3] = 1;
    c[1](0, 3) = b[1]; c[1][3] = 1;
    c[2](0, 3) = b[2]; c[2][3] = 1;
    c[3](0, 3) = b[3]; c[3][3] = 1;
    c[1] *= 2.0;
    c[2] *= 2.0;
    c.elevateDegree();

  /* Create a Bezier surface */
    BezierSurface3d		s(3, 2);
    s[0][0][0] = -1.5;    s[0][0][1] = -1.5;    s[0][0][2] =  4.0;
    s[0][1][0] = -0.5;    s[0][1][1] = -1.5;    s[0][1][2] =  2.0;
    s[0][2][0] =  0.5;    s[0][2][1] = -1.5;    s[0][2][2] = -1.0;
    s[0][3][0] =  1.5;    s[0][3][1] = -1.5;    s[0][3][2] =  2.0;

    s[1][0][0] = -1.5;    s[1][0][1] = -0.5;    s[1][0][2] =  1.0;
    s[1][1][0] = -0.5;    s[1][1][1] = -0.5;    s[1][1][2] =  3.0;
    s[1][2][0] =  0.5;    s[1][2][1] = -0.5;    s[1][2][2] =  0.0;
    s[1][3][0] =  1.5;    s[1][3][1] = -0.5;    s[1][3][2] =  1.0;

    s[2][0][0] = -1.5;    s[2][0][1] =  0.5;    s[2][0][2] =  4.0;
    s[2][1][0] = -0.5;    s[2][1][1] =  0.5;    s[2][1][2] =  0.0;
    s[2][2][0] =  0.5;    s[2][2][1] =  0.5;    s[2][2][2] =  3.0;
    s[2][3][0] =  1.5;    s[2][3][1] =  0.5;    s[2][3][2] =  4.0;

  /*    s[3][0][0] = -1.5;    s[3][0][1] =  1.5;    s[3][0][2] = -2.0;
    s[3][1][0] = -0.5;    s[3][1][1] =  1.5;    s[3][1][2] = -2.0;
    s[3][2][0] =  0.5;    s[3][2][1] =  1.5;    s[3][2][2] =  0.0;
    s[3][3][0] =  1.5;    s[3][3][1] =  1.5;    s[3][3][2] = -1.0;*/

    int			attrs[] = {GLX_RGBA,
				   GLX_RED_SIZE,	1,
				   GLX_GREEN_SIZE,	1,
				   GLX_BLUE_SIZE,	1,
				   GLX_DEPTH_SIZE,	8,
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

    v::MyCmdWindow	myWin0(vapp, "Bezier curve", vinfo, b, c, s);
    vapp.run();

    cerr << "Loop exited!" << endl;

    return 0;
}
