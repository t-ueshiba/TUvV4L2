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
 *  $Id: Rectify.cc 1495 2014-02-27 15:07:51Z ueshiba $
 */
#include "TU/Rectify.h"
#include "TU/algorithm.h"

namespace TU
{
/************************************************************************
*  static functions							*
************************************************************************/
template <class T> static BoundingBox<Point2<T> >
computeBoundingBox(size_t width, size_t height, const Homography<T>& H)
{
    BoundingBox<Point2<T> >	bbox;
    bbox.expand(H(  0.0,    0.0));	// upper-left
    bbox.expand(H(width,    0.0));	// upper-right
    bbox.expand(H(  0.0, height));	// lower-left
    bbox.expand(H(width, height));	// lower-right

    return bbox;
}

/************************************************************************
*  class Rectify							*
************************************************************************/
void
Rectify::computeBaseHomographies(const camera_type& cameraL,
				 const camera_type& cameraR,
				 size_t widthL, size_t heightL,
				 size_t widthR, size_t heightR)
{
    typedef Vector<element_type>			vector_type;
    typedef Matrix<element_type>			matrix_type;
    typedef Vector<element_type,
		   FixedSizedBuf<element_type, 3> >	vector3_type;

  // _H[2] is not used, but initialized with an identiy matrix.
    _H[2] = matrix_type::I(3);
    
  // Compute basic homographies.
    const vector_type&	eR = (cameraL.K() * cameraL.Rt() *
			      (cameraR.t() - cameraL.t()));
    _H[0][0]    =  eR;
    _H[0][1][0] = -eR[1];
    _H[0][1][1] =  eR[0];
    _H[0][1][2] =  0.0;
    _H[0][0].normalize();
    _H[0][1].normalize();		// g = Normalize[(-e2, e1, 0)]
    _H[0][2] = _H[0][0] ^ _H[0][1];	// g = e ^ f
    _H[1] = _H[0]
	  * cameraL.K() * cameraL.Rt() * cameraR.Rt().trns() * cameraR.Kinv();

  // Compute 
    matrix_type		Al(3, 3), Bl, Ar(3, 3), Br;
    vector3_type	tmp;
    Al[0][0] = widthL * widthL  * widthL  * heightL / element_type(12);
    Al[1][1] = widthL * heightL * heightL * heightL / element_type(12);
    Al = _H[0](1, 0, 2, 3) * Al * _H[0](1, 0, 2, 3).trns();
    tmp[0] = widthL  / element_type(2);
    tmp[1] = heightL / element_type(2);
    tmp[2] = element_type(1);
    Bl = (_H[0](1, 0, 2, 3) * tmp) % (_H[0](1, 0, 2, 3) * tmp);
    Ar[0][0] = widthR * widthR  * widthR  * heightR / element_type(12);
    Ar[1][1] = widthR * heightR * heightR * heightR / element_type(12);
    Ar = _H[1](1, 0, 2, 3) * Ar * _H[1](1, 0, 2, 3).trns();
    tmp[0] = widthR  / element_type(2);
    tmp[1] = heightR / element_type(2);
    tmp[2] = element_type(1);
    Br = (_H[1](1, 0, 2, 3) * tmp) % (_H[1](1, 0, 2, 3) * tmp);

  // Estimate optimal weight parameters.    
    vector_type		evalue(2), w = matrix_type(Al + Ar).eigen(evalue)[1];
    for (;;)
    {
	element_type	Jl = (w*Al*w) / (w*Bl*w), Jr = (w*Ar*w) / (w*Br*w);
	matrix_type	Cl = (Al - Jl * Bl) / (w*Bl*w),
			Cr = (Ar - Jr * Br) / (w*Br*w),
			Dl = Bl / (w*Bl*w), Dr = Br / (w*Br*w);
	vector_type	grad = w * (Cl + Cr);
	matrix_type	Hessian = Cl - (Cl*w)%(w*Dl) - (Dl*w)%(w*Cl)
				+ Cr - (Cr*w)%(w*Dr) - (Dr*w)%(w*Cr);
	grad = grad * Hessian.pinv(1.0e6);
      /*#if defined(_DEBUG)
	cerr << grad;
	#endif*/
	if (grad.length() < 1.0e-12)
	    break;
	(w -= grad).normalize();
    }
    (_H[0][2] *= w[1]) += (w[0] * _H[0][1]);
    (_H[1][2] *= w[1]) += (w[0] * _H[1][1]);

#if 1
  // Perform Affine correction.
    vector3_type	mc;
    mc[0] = widthL  / 2;
    mc[1] = heightL / 2;
    mc[2] = 1;
    vector3_type	p = mc ^ (_H[0][0] ^ _H[0][2]);
    vector3_type	q = mc ^ (_H[0][1] ^ _H[0][2]);
    const element_type	ppL = p[0]*p[0] + p[1]*p[1],
			pqL = p[0]*q[0] + p[1]*q[1],
			qqL = q[0]*q[0] + q[1]*q[1];
    mc[0] = widthR  / 2;
    mc[1] = heightR / 2;
    mc[2] = 1;
    p = mc ^ (_H[1][0] ^ _H[1][2]);
    q = mc ^ (_H[1][1] ^ _H[1][2]);
    const element_type	ppR = p[0]*p[0] + p[1]*p[1],
			pqR = p[0]*q[0] + p[1]*q[1],
			qqR = q[0]*q[0] + q[1]*q[1];
    element_type		s = -0.5 * (pqL/qqL + pqR/qqR);
    const element_type	a = sqrt(0.5 * (qqL/(ppL + 2*s*pqL + s*s*qqL) +
					qqR/(ppR + 2*s*pqR + s*s*qqR)));
    s *= a;

    (_H[0][0] *= a) += (s * _H[0][1]);
    (_H[1][0] *= a) += (s * _H[1][1]);
#endif
}
    
void
Rectify::computeBaseHomographies(const camera_type& cameraL,
				 const camera_type& cameraR,
				 const camera_type& cameraV)
{
    typedef Vector<element_type>	vector_type;
    
    const vector_type&	eR = (cameraL.K() * cameraL.Rt() *
			      (cameraR.t() - cameraL.t()));
    const vector_type&	eV = (cameraL.K() * cameraL.Rt() *
			      (cameraV.t() - cameraL.t()));
    _H[0][0][0] = -eV[1];	// eV[1] < 0 assumed.
    _H[0][0][1] =  eV[0];
    _H[0][0][2] =  0.0;
    _H[0][1][0] = -eR[1];	// eR[1] > 0 assumed.
    _H[0][1][1] =  eR[0];
    _H[0][1][2] =  0.0;
    _H[0][2] = eV ^ eR;
    _H[1] = _H[0] * cameraL.K() * cameraL.Rt()
		  * cameraR.Rt().trns() * cameraR.Kinv();
    _H[2][0] = -_H[0][1];
    _H[2][1] =  _H[0][0];
    _H[2][2] =  _H[0][2];
    auto	H2 = _H[2] * (cameraL.K() * cameraL.Rt() *
			      cameraV.Rt().trns() * cameraV.Kinv());
    _H[2] = H2;
}
    
void
Rectify::scaleHomographies(size_t widthL, size_t heightL, element_type scale)
{
    typedef Vector<element_type>	vector_type;

  // 左画像について，変換後の四隅の点が成す四角形の面積を求める．
    vector_type	p[4];
    p[0] = _H[0](   0.0,     0.0).homogeneous();
    p[1] = _H[0](widthL,     0.0).homogeneous();
    p[2] = _H[0](   0.0, heightL).homogeneous();
    p[3] = _H[0](widthL, heightL).homogeneous();
    element_type
	area = (vector_type((p[0] ^ p[1]) +
			    (p[1] ^ p[2]) + (p[2] ^ p[0])).length() +
		vector_type((p[1] ^ p[2]) +
			    (p[2] ^ p[3]) + (p[3] ^ p[1])).length()) / 2;

  // 変換前後の面積比がscaleに一致するように射影変換をスケーリングする．
    element_type	k = scale * sqrt(widthL*heightL / area);
    for (int i = 0; i < 3; ++i)
    {
	_H[i][0] *= k;
	_H[i][1] *= k;
    }
}

Rectify::element_type
Rectify::translateHomographies(const camera_type& cameraL,
			       const camera_type& cameraR,
			       size_t widthL, size_t heightL,
			       size_t widthR, size_t heightR,
			       size_t disparitySearchWidth,
			       size_t disparityMax)
{
    using namespace	std;		// Use std::min() and std::max().

    typedef BoundingBox<Point2<element_type> >	bbox_type;
    
  // Compute the horizontal shift values and the rectified image widths.
    bbox_type		bboxL = computeBoundingBox(widthL, heightL, _H[0]),
			bboxR = computeBoundingBox(widthR, heightR, _H[1]);
    element_type	u0, u0R;
    size_t		widthOutL, widthOutR;
    if (disparitySearchWidth == 0)
    {
	u0	  = -bboxL.min(0);
	u0R	  = -bboxR.min(0);
	widthOutL = size_t(bboxL.length(0) + 0.5);
	widthOutR = size_t(bboxR.length(0) + 0.5);
    }
    else
    {
	const size_t	disparityMin = disparityMax + 1 - disparitySearchWidth;
	
	u0	  = -max(bboxL.min(0), bboxR.min(0) + disparityMin);
	u0R	  = u0 + disparityMax;
	widthOutL = size_t(u0 +
			  min(bboxL.max(0), bboxR.max(0) + disparityMax) + 0.5);
	widthOutR = widthOutL + disparitySearchWidth - 1;
    }

  // Compute the vertical shift value and the rectified image heights.
    element_type	v0	  = -max(bboxL.min(1), bboxR.min(1));
    size_t	heightOut = size_t(v0 + min(bboxL.max(1), bboxR.max(1)));
#if defined(_DEBUG)
    cerr << "(u0, v0): (" << u0 << ", " << v0 << ")\n"
	 << "u0R:       " << u0R << endl;
#endif    

  // Translation.
    _H[0][0] += (u0  * _H[0][2]);
    _H[0][1] += (v0  * _H[0][2]);
    _H[1][0] += (u0R * _H[1][2]);
    _H[1][1] += (v0  * _H[1][2]);

  // Compute table for rectification.
    _warp[0].initialize(_H[0].trns().inv(), cameraL,
			widthL, heightL, widthOutL, heightOut);
    _warp[1].initialize(_H[1].trns().inv(), cameraR,
			widthR, heightR, widthOutR, heightOut);

#if defined(_DEBUG)
    cerr <<   "left image:  " << widthOutL << 'x' << heightOut
	 << "\nright image: " << widthOutR << 'x' << heightOut
	 << endl;
#endif
    return u0 - u0R;
}
    
Rectify::element_type
Rectify::translateHomographies(const camera_type& cameraL,
			       const camera_type& cameraR,
			       const camera_type& cameraV,
			       size_t widthL, size_t heightL,
			       size_t widthR, size_t heightR,
			       size_t widthV, size_t heightV,
			       size_t disparitySearchWidth,
			       size_t disparityMax)
{
    using namespace	std;		// Use std::min() and std::max().

    typedef BoundingBox<Point2<element_type> >	bbox_type;

    bbox_type		bboxL = computeBoundingBox(widthL, heightL, _H[0]),
			bboxR = computeBoundingBox(widthR, heightR, _H[1]),
			bboxV = computeBoundingBox(widthV, heightV, _H[2]);
    element_type	u0, v0, u0R, u0V;
    size_t		widthOutL, heightOutL,
			widthOutR, heightOutR, widthOutV, heightOutV;
    if (disparitySearchWidth == 0)
    {
	u0 = -max(bboxL.min(0), bboxV.min(1));
	v0 = -max(bboxL.min(1), bboxR.min(1));
	widthOutL = heightOutV = size_t(u0 + min(bboxL.max(0), bboxV.max(1))
				       + 0.5);
	heightOutL = heightOutR = size_t(v0 + min(bboxL.max(1), bboxR.max(1))
					+ 0.5);
	u0R = -bboxR.min(0);
	u0V = -bboxV.min(0);
	widthOutR = size_t(bboxR.length(0) + 0.5);
	widthOutV = size_t(bboxV.length(0) + 0.5);
    }
    else
    {
	const size_t	disparityMin = disparityMax + 1 - disparitySearchWidth;

	u0 = -max(bboxL.min(0), bboxR.min(0) + disparityMin, bboxV.min(1));
	v0 = -max(bboxL.min(1), bboxR.min(1), -bboxV.max(0) - disparityMax);
	widthOutL = heightOutV = size_t(u0 + min(bboxL.max(0),
						bboxR.max(0) + disparityMax,
						bboxV.max(1))
					+ 0.5);
	heightOutL = heightOutR = size_t(v0 + min(bboxL.max(1),
						 bboxR.max(1),
						 -bboxV.min(0) - disparityMin)
					 + 0.5);
	u0R = u0 + disparityMax;
	u0V = -v0 + disparityMax + heightOutL - 1;
	widthOutR = widthOutL  + disparitySearchWidth - 1;
	widthOutV = heightOutL + disparitySearchWidth - 1;
    }
#if defined(_DEBUG)
    cerr << "(u0, v0): (" << u0 << ", " << v0 << ")\n"
	 << "u0R:       " << u0R << '\n'
	 << "u0V:       " << u0V << '\n'
	 << "triInv.:   " << u0 - u0R + v0 + u0V << endl;
#endif    

  // Translation.
    _H[0][0] += (u0  * _H[0][2]);
    _H[0][1] += (v0  * _H[0][2]);
    _H[1][0] += (u0R * _H[1][2]);
    _H[1][1] += (v0  * _H[1][2]);
    _H[2][0] += (u0V * _H[2][2]);
    _H[2][1] += (u0  * _H[2][2]);

  // Compute table for rectification.
    _warp[0].initialize(_H[0].trns().inv(), cameraL,
			widthL, heightL, widthOutL, heightOutL);
    _warp[1].initialize(_H[1].trns().inv(), cameraR,
			widthR, heightR, widthOutR, heightOutR);
    _warp[2].initialize(_H[2].trns().inv(), cameraV,
			widthV, heightV, widthOutV, heightOutV);
#if defined(_DEBUG)
    cerr <<   "0th image: " << widthOutL << 'x' << heightOutL
	 << "\n1st image: " << widthOutR << 'x' << heightOutR
	 << "\n2nd image: " << widthOutV << 'x' << heightOutV
	 << endl;
#endif
    return u0 - u0R;
}

Rectify::element_type
Rectify::baselineLength(const camera_type& cameraL,
			const camera_type& cameraR) const
{
    return _H[0][0] * cameraL.K() * cameraL.Rt() * (cameraR.t() - cameraL.t())
	 / Vector<element_type>(_H[0][2] * cameraL.K()).length();
}

}
