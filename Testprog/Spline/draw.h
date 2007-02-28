#include "TU/v/OglDC.h"
#include "TU/Nurbs++.h"

namespace TU
{
namespace v
{
extern OglDC&
operator <<(OglDC& dc, const BSplineCurve2f& c)			;

extern OglDC&
operator <<(OglDC& dc, const RationalBSplineCurve2f& c)		;

extern OglDC&
operator <<(OglDC& dc, const BSplineCurve3f& c)			;

extern OglDC&
operator <<(OglDC& dc, const RationalBSplineCurve3f& c)		;

extern OglDC&
operator <<(OglDC& dc, const BSplineSurface3f& s)		;
/*
extern OglDC&
operator <<(OglDC& dc, const RationalBSplineSurface3f& s)	;
*/
}
}
