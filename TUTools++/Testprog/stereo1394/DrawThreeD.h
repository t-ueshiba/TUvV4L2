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
 *  $Id: DrawThreeD.h 1495 2014-02-27 15:07:51Z ueshiba $
 */
#include "TU/Warp.h"

namespace TU
{
/************************************************************************
*  class DrawThreeD							*
************************************************************************/
class DrawThreeD
{
  public:
    struct C4UB_V3F
    {
	u_char		r, g, b, a;
	float		u, v, d;
    };

    struct T2F_V3F
    {
	Vector2f	st;
	float		u, v, d;
    };
    
    struct N3F_V3F
    {
	float		nu, nv, nd;
	float		u, v, d;
    };

    struct V3F
    {
	float	u, v, d;
    };

  public:
    DrawThreeD()							;
    
    void	initialize(const Matrix34d& Pl, const Matrix34d& Pr,
			   float gap=1.0)				;
    template <class D, class T>
    void	draw(const Image<D>& disparityMap,
		     const Image<T>& image)				;
    template <class D, class T>
    void	draw(const Image<D>& disparityMap,
		     const Image<T>& image, const Warp& warp)		;
    template <class F, class D>
    void	draw(const Image<D>& disparityMap)			;
    void	setCursor(int u, int v, float d)			;
    
  private:
    template <class F>
    void	resize(size_t width)					;
    template <class F>
    void	setNormal(F* vq)					;
    void	drawCursor()					const	;
    
    Array<u_char>	_vertices;
    Matrix44d		_Mt;
    float		_gap;
    float		_u, _v, _d;
};

inline void
DrawThreeD::setCursor(int u, int v, float d)
{
    _u = u;
    _v = v;
    _d = d;
}
    
}
