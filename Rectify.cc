/*!
  \file		Rectify.cc
  \author	Toshio UESHIBA
  \brief	クラス TU::Rectify の実装
*/
#include "TU/Rectify.h"

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
    using matrix22_type	= Matrix<element_type, 2, 2>;
    using vector2_type	= Vector<element_type, 2>;
    using vector3_type	= Vector<element_type, 3>;

  // _H[2] is not used, but initialized with an identiy matrix.
    _H[2] = diag<3, element_type>(1);
    
  // Compute basic homographies.
    const auto	eR = evaluate(cameraL.K() * cameraL.Rt() *
			      (cameraR.t() - cameraL.t()));
    normalize(_H[0][0] = eR);
    normalize(_H[0][1] = {-eR[1], eR[0], 0});	// g = Normalize[(-e2, e1, 0)]
    _H[0][2] = _H[0][0] ^ _H[0][1];		// g = e ^ f
    _H[1] = _H[0] * cameraL.K() * cameraL.Rt()
		  * transpose(cameraR.Rt()) * cameraR.Kinv();

  // Compute 
    matrix22_type	mtmp;
    vector3_type	vtmp;
    mtmp[0][0] = widthL * widthL  * widthL  * heightL / element_type(12);
    mtmp[1][1] = widthL * heightL * heightL * heightL / element_type(12);
    const auto	Al = evaluate(slice(_H[0], 1, 2, 0, 2) * mtmp *
			      transpose(slice(_H[0], 1, 2, 0, 2)));
    vtmp = {widthL/element_type(2), heightL/element_type(2), element_type(1)};
    const auto	Bl = evaluate((slice(_H[0], 1, 2, 0, 3) * vtmp) %
			      (slice(_H[0], 1, 2, 0, 3) * vtmp));
    mtmp[0][0] = widthR * widthR  * widthR  * heightR / element_type(12);
    mtmp[1][1] = widthR * heightR * heightR * heightR / element_type(12);
    const auto	Ar = evaluate(slice(_H[1], 1, 2, 0, 2) * mtmp *
			      transpose(slice(_H[1], 1, 2, 0, 2)));
    vtmp = {widthR/element_type(2), heightR/element_type(2), element_type(1)};
    const auto	Br = evaluate((slice(_H[1], 1, 2, 0, 3) * vtmp) %
			      (slice(_H[1], 1, 2, 0, 3) * vtmp));

  // Estimate optimal weight parameters.    
    vector2_type	evalue(2);
    vector2_type	w = eigen(Al + Ar, evalue)[1];
    for (;;)
    {
	const auto	Jl = (w*Al*w) / (w*Bl*w);
	const auto	Jr = (w*Ar*w) / (w*Br*w);
	const auto	Cl = evaluate((Al - Jl * Bl) / (w*Bl*w));
	const auto	Cr = evaluate((Ar - Jr * Br) / (w*Br*w));
	const auto	Dl = evaluate(Bl / (w*Bl*w));
	const auto	Dr = evaluate(Br / (w*Br*w));
	const auto	gg = evaluate(w * (Cl + Cr));
	const auto	Hessian = evaluate(Cl - (Cl*w)%(w*Dl) - (Dl*w)%(w*Cl) +
					   Cr - (Cr*w)%(w*Dr) - (Dr*w)%(w*Cr));
	const auto	grad = evaluate(gg * pseudo_inverse(Hessian, 1.0e6));
#if defined(_DEBUG)
	cerr << grad;
#endif
	if (length(grad) < 1.0e-12)
	    break;
	normalize(w -= grad);
    }
    (_H[0][2] *= w[1]) += (w[0] * _H[0][1]);
    (_H[1][2] *= w[1]) += (w[0] * _H[1][1]);

#if 1
  // Perform Affine correction.
    vector3_type	mc({element_type(widthL)/2, element_type(heightL)/2, 1});
    auto		p = evaluate(mc ^ (_H[0][0] ^ _H[0][2]));
    auto		q = evaluate(mc ^ (_H[0][1] ^ _H[0][2]));
    const auto		ppL = p[0]*p[0] + p[1]*p[1];
    const auto		pqL = p[0]*q[0] + p[1]*q[1];
    const auto		qqL = q[0]*q[0] + q[1]*q[1];
    mc = {element_type(widthR)/2, element_type(heightR)/2, 1};
    p = mc ^ (_H[1][0] ^ _H[1][2]);
    q = mc ^ (_H[1][1] ^ _H[1][2]);
    const auto		ppR = p[0]*p[0] + p[1]*p[1];
    const auto		pqR = p[0]*q[0] + p[1]*q[1];
    const auto		qqR = q[0]*q[0] + q[1]*q[1];
    auto		s = -0.5 * (pqL/qqL + pqR/qqR);
    const auto		a = sqrt(0.5 * (qqL/(ppL + 2*s*pqL + s*s*qqL) +
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
    const auto	eR = evaluate(cameraL.K() * cameraL.Rt() *
			      (cameraR.t() - cameraL.t()));
    const auto	eV = evaluate(cameraL.K() * cameraL.Rt() *
			      (cameraV.t() - cameraL.t()));
    _H[0][0] = {-eV[1], eV[0], 0};	// eV[1] < 0 assumed.
    _H[0][1] = {-eR[1], eR[0], 0};	// eR[1] > 0 assumed.
    _H[0][2] = eV ^ eR;

    _H[1] = _H[0] * cameraL.K() * cameraL.Rt()
		  * transpose(cameraR.Rt()) * cameraR.Kinv();

    homography_type::base_type	H2;
    H2[0] = -_H[0][1];
    H2[1] =  _H[0][0];
    H2[2] =  _H[0][2];
    _H[2] =  H2 * (cameraL.K() * cameraL.Rt() *
		   transpose(cameraV.Rt()) * cameraV.Kinv());
}
    
void
Rectify::scaleHomographies(size_t widthL, size_t heightL, element_type scale)
{
    using vector3_type	= Vector<element_type, 3>;

  // 左画像について，変換後の四隅の点が成す四角形の面積を求める．
    const vector3_type	p[] = {homogeneous(_H[0](0,      0)),
			       homogeneous(_H[0](widthL, 0)),
			       homogeneous(_H[0](0,	 heightL)),
			       homogeneous(_H[0](widthL, heightL))};
    const auto		area = (length(vector3_type((p[0] ^ p[1]) +
						    (p[1] ^ p[2]) +
						    (p[2] ^ p[0]))) +
				length(vector3_type((p[1] ^ p[2]) +
						    (p[2] ^ p[3]) +
						    (p[3] ^ p[1]))))/2;

  // 変換前後の面積比がscaleに一致するように射影変換をスケーリングする．
    const auto		k = scale * sqrt(widthL*heightL / area);
    for (auto& H : _H)
    {
	H[0] *= k;
	H[1] *= k;
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
  // Compute the horizontal shift values and the rectified image widths.
    const auto		bboxL = computeBoundingBox(widthL, heightL, _H[0]);
    const auto		bboxR = computeBoundingBox(widthR, heightR, _H[1]);
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
	const auto	disparityMin = disparityMax + 1 - disparitySearchWidth;
	
	u0	  = -std::max(bboxL.min(0), bboxR.min(0) + disparityMin);
	u0R	  = u0 + disparityMax;
	widthOutL = size_t(u0 + std::min(bboxL.max(0),
					 bboxR.max(0) + disparityMax) + 0.5);
	widthOutR = widthOutL + disparitySearchWidth - 1;
    }

  // Compute the vertical shift value and the rectified image heights.
    const auto	v0	  = -std::max(bboxL.min(1), bboxR.min(1));
    const auto	heightOut = size_t(v0 + std::min(bboxL.max(1), bboxR.max(1)));
#if defined(_DEBUG)
    std::cerr << "(u0, v0): (" << u0 << ", " << v0 << ")\n"
	      << "u0R:       " << u0R
	      << "\nleft image:  " << widthOutL << 'x' << heightOut
	      << "\nright image: " << widthOutR << 'x' << heightOut
	      << std::endl;
#endif    

  // Translation.
    _H[0][0] += (u0  * _H[0][2]);
    _H[0][1] += (v0  * _H[0][2]);
    _H[1][0] += (u0R * _H[1][2]);
    _H[1][1] += (v0  * _H[1][2]);

  // Compute table for rectification.
    _warp[0].initialize(inverse(transpose(_H[0])), cameraL,
			widthL, heightL, widthOutL, heightOut);
    _warp[1].initialize(inverse(transpose(_H[1])), cameraR,
			widthR, heightR, widthOutR, heightOut);

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
    const auto		bboxL = computeBoundingBox(widthL, heightL, _H[0]);
    const auto		bboxR = computeBoundingBox(widthR, heightR, _H[1]);
    const auto		bboxV = computeBoundingBox(widthV, heightV, _H[2]);
    element_type	u0, v0, u0R, u0V;
    size_t		widthOutL, heightOutL,
			widthOutR, heightOutR, widthOutV, heightOutV;
    if (disparitySearchWidth == 0)
    {
	u0 = -std::max(bboxL.min(0), bboxV.min(1));
	v0 = -std::max(bboxL.min(1), bboxR.min(1));
	widthOutL = heightOutV = size_t(u0 +
					std::min(bboxL.max(0), bboxV.max(1))
					+ 0.5);
	heightOutL = heightOutR = size_t(v0 +
					 std::min(bboxL.max(1), bboxR.max(1))
					+ 0.5);
	u0R = -bboxR.min(0);
	u0V = -bboxV.min(0);
	widthOutR = size_t(bboxR.length(0) + 0.5);
	widthOutV = size_t(bboxV.length(0) + 0.5);
    }
    else
    {
	const auto	disparityMin = disparityMax + 1 - disparitySearchWidth;

	u0 = -std::max({bboxL.min(0),
			bboxR.min(0) + disparityMin, bboxV.min(1)});
	v0 = -std::max({bboxL.min(1),
			bboxR.min(1), -bboxV.max(0) - disparityMax});
	widthOutL = heightOutV = size_t(u0 +
					std::min({bboxL.max(0),
						  bboxR.max(0) + disparityMax,
						  bboxV.max(1)})
					+ 0.5);
	heightOutL = heightOutR = size_t(v0 + std::min({bboxL.max(1),
						        bboxR.max(1),
						        -bboxV.min(0) -
						        disparityMin})
					 + 0.5);
	u0R = u0 + disparityMax;
	u0V = -v0 + disparityMax + heightOutL - 1;
	widthOutR = widthOutL  + disparitySearchWidth - 1;
	widthOutV = heightOutL + disparitySearchWidth - 1;
    }
#if defined(_DEBUG)
    std::cerr << "(u0, v0): (" << u0 << ", " << v0 << ")\n"
	      << "u0R:       " << u0R << '\n'
	      << "u0V:       " << u0V << '\n'
	      << "triInv.:   " << u0 - u0R + v0 + u0V
	      << "\n0th image: " << widthOutL << 'x' << heightOutL
	      << "\n1st image: " << widthOutR << 'x' << heightOutR
	      << "\n2nd image: " << widthOutV << 'x' << heightOutV
	      << std::endl;
#endif    

  // Translation.
    _H[0][0] += (u0  * _H[0][2]);
    _H[0][1] += (v0  * _H[0][2]);
    _H[1][0] += (u0R * _H[1][2]);
    _H[1][1] += (v0  * _H[1][2]);
    _H[2][0] += (u0V * _H[2][2]);
    _H[2][1] += (u0  * _H[2][2]);

  // Compute table for rectification.
    _warp[0].initialize(inverse(transpose(_H[0])), cameraL,
			widthL, heightL, widthOutL, heightOutL);
    _warp[1].initialize(inverse(transpose(_H[1])), cameraR,
			widthR, heightR, widthOutR, heightOutR);
    _warp[2].initialize(inverse(transpose(_H[2])), cameraV,
			widthV, heightV, widthOutV, heightOutV);

    return u0 - u0R;
}

Rectify::element_type
Rectify::baselineLength(const camera_type& cameraL,
			const camera_type& cameraR) const
{
    return _H[0][0] * cameraL.K() * cameraL.Rt() * (cameraR.t() - cameraL.t())
	 / length(_H[0][2] * cameraL.K());
}

}
