#include "TU/v/OglDC.h"
#include "TU/Nurbs++.h"

namespace TU
{
namespace v
{
extern OglDC&
operator <<(OglDC& dc, const BSplineCurve<float, 2u>& c)		;

extern OglDC&
operator <<(OglDC& dc, const RationalBSplineCurve<float, 2u>& c)	;

extern OglDC&
operator <<(OglDC& dc, const BSplineCurve<float, 3u>& c)		;

extern OglDC&
operator <<(OglDC& dc, const RationalBSplineCurve<float, 3u>& c)	;

extern OglDC&
operator <<(OglDC& dc, const BSplineSurface<float>& s)		;
/*
extern OglDC&
operator <<(OglDC& dc, const RationalBSplineSurface<float>& s)	;
*/
}
}
