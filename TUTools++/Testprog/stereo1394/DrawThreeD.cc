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
 *  $Id: DrawThreeD.cc 1563 2014-05-25 21:51:28Z ueshiba $
 */
#include <GL/gl.h>
#include "DrawThreeD.h"
#include "TU/mmInstructions.h"

namespace TU
{
/************************************************************************
*  static functions							*
************************************************************************/
template <class T> GLint  alignment()			;
template <> inline GLint  alignment<u_char>()		{return 1;}
template <> inline GLint  alignment<RGB>()		{return 1;}
template <> inline GLint  alignment<RGBA>()		{return 4;}

template <class T> GLint  components()			;
template <> inline GLint  components<u_char>()		{return 1;}
template <> inline GLint  components<RGB>()		{return 3;}
template <> inline GLint  components<RGBA>()		{return 4;}

template <class F> GLenum format()			;
template <> inline GLenum format<u_char>()		{return GL_LUMINANCE;}
template <> inline GLenum format<RGB>()			{return GL_RGB;}
template <> inline GLenum format<RGBA>()		{return GL_RGBA;}
template <> inline GLenum format<DrawThreeD::C4UB_V3F>(){return GL_C4UB_V3F;}
template <> inline GLenum format<DrawThreeD::T2F_V3F>()	{return GL_T2F_V3F;}
template <> inline GLenum format<DrawThreeD::N3F_V3F>()	{return GL_N3F_V3F;}
template <> inline GLenum format<DrawThreeD::V3F>()	{return GL_V3F;}

template <class T> inline GLenum type()		{return GL_UNSIGNED_BYTE;}
template <>	   inline GLenum type<short>()	{return GL_SHORT;}
    
template <class T> inline void
setRGB(DrawThreeD::C4UB_V3F& vertex, T pixel)
{
    vertex.r = pixel.r;
    vertex.g = pixel.g;
    vertex.b = pixel.b;
    vertex.a = 255;
}

template <> inline void
setRGB(DrawThreeD::C4UB_V3F& vertex, u_char pixel)
{
    vertex.r = pixel;
    vertex.g = pixel;
    vertex.b = pixel;
    vertex.a = 255;
}

/************************************************************************
*  class DrawThreeD							*
************************************************************************/
DrawThreeD::DrawThreeD()
    :_vertices(), _Mt(), _gap(2.0), _u(0.0), _v(0.0), _d(0.0)
{
}

void
DrawThreeD::initialize(const Matrix34d& Pl, const Matrix34d& Pr, float gap)
{
    Vector4d	t;
    t[0] = -Pr[0][3];
    t[1] = -Pr[1][3];
    t[2] = -Pr[2][3];
    t[3] = 1.0;
    t(0, 3).solve(Pr(0, 0, 3, 3).trns());	// t = camera center of Pr.
    Matrix33d	Tt = (Pl[0] * t) * Pl(0, 0, 3, 3).trns().inv();
    _Mt[0](0, 3) = Tt[0];
    _Mt[1](0, 3) = Tt[1];
    t[0] = -Pl[0][3];
    t[1] = -Pl[1][3];
    t[2] = -Pl[2][3];
    t(0, 3).solve(Pl(0, 0, 3, 3).trns());	// t = camera center of Pl.
    _Mt[2] = t;
    _Mt[3](0, 3) = Tt[2];

    _gap = gap;
}

template <class D, class T> void
DrawThreeD::draw(const Image<D>& disparityMap, const Image<T>& image)
{
    typedef C4UB_V3F		Vertex;
    
    resize<Vertex>(disparityMap.width());

  // 頂点配列を作って頂点を描画．
    glPushMatrix();
    glMultMatrixd(_Mt.data());

    for (size_t v = 1; v < disparityMap.height(); ++v)
    {
	const size_t		width = disparityMap.width();
	const ImageLine<T>	&imgp = image[v-1], &imgq = image[v];
	const ImageLine<D>	&mapp = disparityMap[v-1],
				&mapq = disparityMap[v];
	D			dp_prev = 0, dq_prev = 0;
	Vertex* const		vertex0 = (Vertex*)_vertices.data();
	Vertex*			vertex  = vertex0;
	for (size_t u = 0; u < width; ++u)
	{
	    const D	dp = mapp[u], dq = mapq[u];
	    if (dp != 0 && dq != 0 && std::abs(dp - dq) <= _gap)
	    {
		if (std::abs(dp - dp_prev) > _gap ||
		    std::abs(dq - dq_prev) > _gap)
		{
		    glDrawArrays(GL_QUAD_STRIP, 0, vertex - vertex0);
		    vertex = vertex0;
		}
		vertex->u = u;
		vertex->v = v - 1;
		vertex->d = dp;
		setRGB(*vertex++, imgp[u]);
		vertex->u = u;
		vertex->v = v;
		vertex->d = dq;
		setRGB(*vertex++, imgq[u]);
	    }
	    else
	    {
		glDrawArrays(GL_QUAD_STRIP, 0, vertex - vertex0);
		vertex = vertex0;
	    }
	    dp_prev = dp;
	    dq_prev = dq;
	}
	glDrawArrays(GL_QUAD_STRIP, 0, vertex - vertex0);
    }

    drawCursor();
    glPopMatrix();
}

template <class D, class T> void
DrawThreeD::draw(const Image<D>& disparityMap, const Image<T>& image,
		 const Warp& warp)
{
    typedef T2F_V3F	Vertex;
    
    resize<Vertex>(disparityMap.width());

  // テクスチャ画像をセット．
    glPixelStorei(GL_UNPACK_ALIGNMENT, alignment<T>());
    glTexImage2D(GL_TEXTURE_2D, 0, components<T>(),
		 image.width(), image.height(), 0,
		 format<T>(), type<T>(), image.data());
		 
  // 頂点配列を作って頂点を描画．
    glPushMatrix();
    glMultMatrixd(_Mt.data());
    glColor3f(1.0, 1.0, 1.0);

//#if defined(SSE2)
#if 0
    using namespace	mm;
    
    const F32vec	textureWidthHeight(image.height(), image.width(),
					   image.height(), image.width());
#else
    const float	textureWidth = image.width(), textureHeight = image.height();
#endif
    for (size_t v = 1; v < disparityMap.height(); ++v)
    {
	const size_t		width = disparityMap.width();
	const ImageLine<D>	&mapp = disparityMap[v-1],
				&mapq = disparityMap[v];
	D			dp_prev = 0.0, dq_prev = 0.0;
	Vertex* const		vertex0 = (Vertex*)_vertices.data();
	Vertex*			vertex  = vertex0;
	for (size_t u = 0; u < width; ++u)
	{
	    const D	dp = mapp[u], dq = mapq[u];
	    if (dp != 0.0 && dq != 0.0 && std::abs(dp - dq) <= _gap)
	    {
		if (std::abs(dp - dp_prev) > _gap ||
		    std::abs(dq - dq_prev) > _gap)
		{
		    glDrawArrays(GL_QUAD_STRIP, 0, vertex - vertex0);
		    vertex = vertex0;
		}
//#if defined(SSE2)
#if 0
		const F32vec	src = warp.src(u, v) / textureWidthHeight;
		const F32vec	dst = cvt<float>(Is32vec(v, u, v-1, u));
		store<false>(vertex->st.data(),
			     cast<float>(unpack_low(cast<u_int64_t>(src),
						    cast<u_int64_t>(dst))));
		vertex->d = dp;
		++vertex;
		store<false>(vertex->st.data(),
			     cast<float>(unpack_high(cast<u_int64_t>(src),
						     cast<u_int64_t>(dst))));
		vertex->d = dq;
		++vertex;
#else
		vertex->u = u;
		vertex->v = v - 1;
		vertex->d = dp;
		vertex->st = warp(u, v - 1);
		vertex->st[0] /= textureWidth;
		vertex->st[1] /= textureHeight;
		++vertex;
		vertex->u = u;
		vertex->v = v;
		vertex->d = dq;
		vertex->st = warp(u, v);
		vertex->st[0] /= textureWidth;
		vertex->st[1] /= textureHeight;
		++vertex;
#endif
	    }
	    else
	    {
		glDrawArrays(GL_QUAD_STRIP, 0, vertex - vertex0);
		vertex = vertex0;
	    }
	    dp_prev = dp;
	    dq_prev = dq;
	}
	glDrawArrays(GL_QUAD_STRIP, 0, vertex - vertex0);
    }
//#if defined(SSE2)
#if 0
    mm::empty();
#endif

    drawCursor();
    glPopMatrix();
}

template <class F, class D> void
DrawThreeD::draw(const Image<D>& disparityMap)
{
    resize<F>(disparityMap.width());

  // 頂点配列を作って頂点を描画．
    glPushMatrix();
    glMultMatrixd(_Mt.data());
    glColor3f(1.0, 1.0, 0.0);

    for (size_t v = 1; v < disparityMap.height(); ++v)
    {
	const size_t		width = disparityMap.width();
	const ImageLine<D>	&mapp = disparityMap[v-1],
				&mapq = disparityMap[v];
	D			dp_prev = 0.0, dq_prev = 0.0;
	F*			vertex0 = (F*)_vertices.data();
	F*			vertex  = vertex0;
	
	for (size_t u = 0; u < width; ++u)
	{
	    const D	dp = mapp[u], dq = mapq[u];
	    if (dp != 0.0 && dq != 0.0 && std::abs(dp - dq) <= _gap)
	    {
		if (std::abs(dp - dp_prev) > _gap ||
		    std::abs(dq - dq_prev) > _gap)
		{
		    glDrawArrays(GL_QUAD_STRIP, 0, vertex - vertex0);
		    vertex = vertex0;
		}
		vertex->u = u;
		vertex->v = v - 1;
		vertex->d = dp;
		++vertex;
		vertex->u = u;
		vertex->v = v;
		vertex->d = dq;
		setNormal(vertex++);
	    }
	    else
	    {
		glDrawArrays(GL_QUAD_STRIP, 0, vertex - vertex0);
		vertex = vertex0;
	    }
	    dp_prev = dp;
	    dq_prev = dq;
	}
	glDrawArrays(GL_QUAD_STRIP, 0, vertex - vertex0);
    }

    drawCursor();
    glPopMatrix();
}

template <class F> void
DrawThreeD::resize(size_t width)
{
    if (2*sizeof(F)*width != _vertices.dim())
	_vertices.resize(2*sizeof(F)*width);
    glInterleavedArrays(format<F>(), 0, _vertices.data());
}

template <class F> inline void
DrawThreeD::setNormal(F* vq)
{
    if (vq == (F*)_vertices.data())
    {
	F* const	vp = vq - 1;

	vp->nu = vq->nu = 1.0;
	vp->nv = vq->nv = 1.0;
	vp->nd = vq->nd = 1.0;
    }
    else
    {
	F* const	vp = vq - 1;
	F* const	vr = vq - 2;
	const float	d0 = -_Mt[3][3];
	
	vp->nu = vq->nu = vq->d - vr->d;
	vp->nv = vq->nv = vq->d - vp->d;
	vp->nd = vq->nd = (vq->d - d0 - vq->nu * vq->u - vq->nv * vq->v) / d0;
    }
}

template <> inline void
DrawThreeD::setNormal(V3F*)
{
}

void
DrawThreeD::drawCursor() const
{
    if (_d != 0.0)
    {
	glBegin(GL_LINES);
	  glColor3f(1.0, 0.0, 0.0);
	  glVertex3f(_u-5.0, _v, _d);
	  glVertex3f(_u+5.0, _v, _d);
	  glColor3f(0.0, 1.0, 0.0);
	  glVertex3f(_u, _v-5.0, _d);
	  glVertex3f(_u, _v+5.0, _d);
	  glColor3f(0.0, 0.0, 1.0);
	  glVertex3f(_u, _v, _d-1.0);
	  glVertex3f(_u, _v, _d+1.0);
	glEnd();
    }
}

template void
DrawThreeD::draw(const Image<float>& disparityMap, const Image<u_char>& image);
template void
DrawThreeD::draw(const Image<float>& disparityMap, const Image<ABGR>& image);
template void
DrawThreeD::draw(const Image<float>& disparityMap, const Image<RGBA>& image);
template void
DrawThreeD::draw(const Image<float>& disparityMap,
		 const Image<u_char>& image, const Warp& warp);
template void
DrawThreeD::draw(const Image<float>& disparityMap,
		 const Image<RGBA>& image, const Warp& warp);
template void
DrawThreeD::draw<DrawThreeD::N3F_V3F>(const Image<float>& disparityMap);
    
template void
DrawThreeD::draw<DrawThreeD::V3F>(const Image<float>& disparityMap);
    
template void
DrawThreeD::draw(const Image<u_char>& disparityMap, const Image<u_char>& image);
template void
DrawThreeD::draw(const Image<u_char>& disparityMap, const Image<ABGR>& image);
template void
DrawThreeD::draw(const Image<u_char>& disparityMap, const Image<RGBA>& image);
template void
DrawThreeD::draw(const Image<u_char>& disparityMap,
		 const Image<u_char>& image, const Warp& warp);
template void
DrawThreeD::draw(const Image<u_char>& disparityMap,
		 const Image<RGBA>& image, const Warp& warp);
template void
DrawThreeD::draw<DrawThreeD::N3F_V3F>(const Image<u_char>& disparityMap);
    
template void
DrawThreeD::draw<DrawThreeD::V3F>(const Image<u_char>& disparityMap);
    
}
