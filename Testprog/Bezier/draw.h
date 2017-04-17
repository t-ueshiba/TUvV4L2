#include "TU/v/OglDC.h"
#include "TU/Bezier++.h"

namespace TU
{
namespace v
{
extern OglDC&
operator <<(OglDC& dc, const BezierCurve2d& b)				;

extern OglDC&
operator <<(OglDC& dc, const RationalBezierCurve2d& b)			;

extern OglDC&
operator <<(OglDC& dc, const BezierCurve3d& b)				;

extern OglDC&
operator <<(OglDC& dc, const RationalBezierCurve3d& b)			;

extern OglDC&
operator <<(OglDC& dc, const BezierSurface3d& b)			;

extern OglDC&
operator <<(OglDC& dc, const RationalBezierSurface3d& b)		;
}
}
