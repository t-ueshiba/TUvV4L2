#include "TU/v/OglDC.h"
#include "TU/Bezier++.h"

namespace TU
{
namespace v
{
extern OglDC&
operator <<(OglDC& dc, const BezierCurve<double, 2u>& b)		;

extern OglDC&
operator <<(OglDC& dc, const RationalBezierCurve<double, 2u>& b)	;

extern OglDC&
operator <<(OglDC& dc, const BezierCurve<double, 3u>& b)		;

extern OglDC&
operator <<(OglDC& dc, const RationalBezierCurve<double, 3u>& b)	;

extern OglDC&
operator <<(OglDC& dc, const BezierSurface<double>& b)			;

extern OglDC&
operator <<(OglDC& dc, const RationalBezierSurface<double>& b)		;
}
}
