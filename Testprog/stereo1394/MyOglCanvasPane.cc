/*
 *  平成14-19年（独）産業技術総合研究所 著作権所有
 *  
 *  創作者：植芝俊夫
 *
 *  本プログラムは（独）産業技術総合研究所の職員である植芝俊夫が創作し，
 *  （独）産業技術総合研究所が著作権を所有する秘密情報です．創作者によ
 *  る許可なしに本プログラムを使用，複製，改変，第三者へ開示する等の著
 *  作権を侵害する行為を禁止します．
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
 *  without permission by the creator are strictly prohibited.
 *
 *  [No Warranty.]
 *  The copyright holders or the creator are not responsible for any
 *  damages in the use of this program.
 *  
 *  $Id: MyOglCanvasPane.cc 1495 2014-02-27 15:07:51Z ueshiba $
 */
#include "MyOglCanvasPane.h"

namespace TU
{
namespace v
{
/************************************************************************
*  class MyOglCanvasPaneBase						*
************************************************************************/
MyOglCanvasPaneBase::MyOglCanvasPaneBase(Window&		parentWin,
					 size_t			width,
					 size_t			height,
					 const Image<float>&	disparityMap)
    :CanvasPane(parentWin, width, height), _draw(), _dc(*this),
     _disparityMap(disparityMap), _drawMode(Texture), _tick(0), _parallax(-1.0)
{
}

void
MyOglCanvasPaneBase::initialize(const Matrix34d& Pl,
				const Matrix34d& Pr, double scale)
{
    typedef Camera<IntrinsicBase<double> >	camera_type;
    
    _draw.initialize(Pl, Pr);

    camera_type	camera(Pl);
    _dc.setInternal(camera.u0()[0] * scale, camera.u0()[1] * scale,
		    camera.k() * scale, camera.k() * scale, 0.01)
       .setExternal(camera.t(), camera.Rt());
    _dc << distance(1300.0);

    glPushMatrix();
}

void
MyOglCanvasPaneBase::swingView()
{
    const double	RAD = M_PI / 180.0;
    const double	magnitudeX = 45*RAD, magnitudeY = 45*RAD;
    const int		periodX = 400, periodY = 600;
    
    double		angleX = magnitudeX * sin(2.0*M_PI*_tick / periodX),
			angleY = magnitudeY * sin(2.0*M_PI*_tick / periodY);
    glPopMatrix();
    glPushMatrix();
    _dc << TU::v::axis(DC3::X) << TU::v::rotate(angleX)
	<< TU::v::axis(DC3::Y) << TU::v::rotate(angleY);
    ++_tick;
}

void
MyOglCanvasPaneBase::initializeGraphics()
{
  //    glClearColor(0.0, 0.1, 0.0, 1.0);
    glEnable(GL_DEPTH_TEST);

    glFrontFace(GL_CW);
    glCullFace(GL_FRONT);
    glEnable(GL_CULL_FACE);

    glEnable(GL_COLOR_MATERIAL);
    glDisable(GL_AUTO_NORMAL);
    glEnable(GL_NORMALIZE);
    glShadeModel(GL_FLAT);

    GLfloat	position[] = {1.0, 1.0, -1.0, 0.0};
    glLightfv(GL_LIGHT0, GL_POSITION, position);
    glEnable(GL_LIGHT0);

    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP);
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP);
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexEnvf(GL_TEXTURE_2D, GL_TEXTURE_ENV_MODE, GL_DECAL);

    setDrawMode(_drawMode);
}

}
}
