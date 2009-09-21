/* MODIFIED ATHENA SLIDER (USING ARROWHEADS AT ENDS OF TRAVEL) */
/* Modifications Copyright 1992 by Mitch Trachtenberg             */
/* Rights, permissions, and disclaimer of warranty are as in the  */
/* DEC and MIT notice below.                                      */
/* $XConsortium: Slider.c,v 1.70 91/10/16 21:39:40 eswu Exp $ */

/***********************************************************

Copyright (c) 1987, 1988, 1994  X Consortium

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
X CONSORTIUM BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN
AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

Except as contained in this notice, the name of the X Consortium shall not be
used in advertising or otherwise to promote the sale, use or other dealings
in this Software without prior written authorization from the X Consortium.


Copyright 1987, 1988 by Digital Equipment Corporation, Maynard, Massachusetts.

                        All Rights Reserved

Permission to use, copy, modify, and distribute this software and its
documentation for any purpose and without fee is hereby granted,
provided that the above copyright notice appear in all copies and that
both that copyright notice and this permission notice appear in
supporting documentation, and that the name of Digital not be
used in advertising or publicity pertaining to distribution of the
software without specific, written prior permission.

DIGITAL DISCLAIMS ALL WARRANTIES WITH REGARD TO THIS SOFTWARE, INCLUDING
ALL IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS, IN NO EVENT SHALL
DIGITAL BE LIABLE FOR ANY SPECIAL, INDIRECT OR CONSEQUENTIAL DAMAGES OR
ANY DAMAGES WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS,
WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS ACTION,
ARISING OUT OF OR IN CONNECTION WITH THE USE OR PERFORMANCE OF THIS
SOFTWARE.

******************************************************************/

/* Slider.c */
/* created by weissman, Mon Jul  7 13:20:03 1986 */
/* converted by swick, Thu Aug 27 1987 */

#include <X11/IntrinsicP.h>
#include <X11/StringDefs.h>

#include <X11/Xaw3d/XawInit.h>
#include "vSliderP_.h"

#include <X11/Xmu/Drawing.h>

/* Private definitions. */

static char defaultTranslations[] =
    "<BtnDown>:	   StartScroll(Continuous) MoveThumb() NotifyThumb()	\n\
     <BtnMotion>:  MoveThumb() NotifyThumb()				\n\
     <BtnUp>:	   NotifyScroll(Proportional) EndScroll()";

static float floatZero = 0.0;

#define Offset(field) XtOffsetOf(SliderRec, field)

static XtResource resources[] = {
  {XtNscrollVCursor, XtCCursor, XtRCursor, sizeof(Cursor),
       Offset(slider.verCursor), XtRString, "sb_v_double_arrow"},
  {XtNscrollHCursor, XtCCursor, XtRCursor, sizeof(Cursor),
       Offset(slider.horCursor), XtRString, "sb_h_double_arrow"},
  {XtNlength, XtCLength, XtRDimension, sizeof(Dimension),
       Offset(slider.length), XtRImmediate, (XtPointer) 1},
  {XtNthickness, XtCThickness, XtRDimension, sizeof(Dimension),
       Offset(slider.thickness), XtRImmediate, (XtPointer) 14},
  {XtNorientation, XtCOrientation, XtROrientation, sizeof(XtOrientation),
      Offset(slider.orientation), XtRImmediate, (XtPointer) XtorientVertical},
  {XtNscrollProc, XtCCallback, XtRCallback, sizeof(XtPointer),
       Offset(slider.scrollProc), XtRCallback, NULL},
  {XtNthumbProc, XtCCallback, XtRCallback, sizeof(XtPointer),
       Offset(slider.thumbProc), XtRCallback, NULL},
  {XtNjumpProc, XtCCallback, XtRCallback, sizeof(XtPointer),
       Offset(slider.jumpProc), XtRCallback, NULL},
  {XtNthumb, XtCThumb, XtRBitmap, sizeof(Pixmap),
       Offset(slider.thumb), XtRImmediate, (XtPointer) XtUnspecifiedPixmap},
  {XtNforeground, XtCForeground, XtRPixel, sizeof(Pixel),
       Offset(slider.foreground), XtRString, XtDefaultForeground},
  {XtNshown, XtCShown, XtRFloat, sizeof(float),
       Offset(slider.shown), XtRFloat, (XtPointer)&floatZero},
  {XtNtopOfThumb, XtCTopOfThumb, XtRFloat, sizeof(float),
       Offset(slider.top), XtRFloat, (XtPointer)&floatZero},
  {XtNpickTop, XtCPickTop, XtRBoolean, sizeof(Boolean),
       Offset(slider.pick_top), XtRBoolean, (XtPointer) False},
  {XtNminimumThumb, XtCMinimumThumb, XtRDimension, sizeof(Dimension),
       Offset(slider.min_thumb), XtRImmediate, (XtPointer) 7}
};
#undef Offset

static void ClassInitialize();
static void Initialize();
static void Destroy();
static void Realize();
static void Resize();
static void Redisplay();
static Boolean SetValues();

static void StartScroll();
static void MoveThumb();
static void NotifyThumb();
static void NotifyScroll();
static void EndScroll();

static XtActionsRec actions[] = {
    {"StartScroll",     StartScroll},
    {"MoveThumb",	MoveThumb},
    {"NotifyThumb",	NotifyThumb},
    {"NotifyScroll",	NotifyScroll},
    {"EndScroll",	EndScroll}
};


SliderClassRec sliderClassRec = {
  { /* core fields */
    /* superclass       */	(WidgetClass) &threeDClassRec,
    /* class_name       */	"Slider",
    /* size             */	sizeof(SliderRec),
    /* class_initialize	*/	ClassInitialize,
    /* class_part_init  */	NULL,
    /* class_inited	*/	FALSE,
    /* initialize       */	Initialize,
    /* initialize_hook  */	NULL,
    /* realize          */	Realize,
    /* actions          */	actions,
    /* num_actions	*/	XtNumber(actions),
    /* resources        */	resources,
    /* num_resources    */	XtNumber(resources),
    /* xrm_class        */	NULLQUARK,
    /* compress_motion	*/	TRUE,
    /* compress_exposure*/	TRUE,
    /* compress_enterleave*/	TRUE,
    /* visible_interest */	FALSE,
    /* destroy          */	Destroy,
    /* resize           */	Resize,
    /* expose           */	Redisplay,
    /* set_values       */	SetValues,
    /* set_values_hook  */	NULL,
    /* set_values_almost */	XtInheritSetValuesAlmost,
    /* get_values_hook  */	NULL,
    /* accept_focus     */	NULL,
    /* version          */	XtVersion,
    /* callback_private */	NULL,
    /* tm_table         */	defaultTranslations,
    /* query_geometry	*/	XtInheritQueryGeometry,
    /* display_accelerator*/	XtInheritDisplayAccelerator,
    /* extension        */	NULL
  },
  { /* simple fields */
    /* change_sensitive	*/	XtInheritChangeSensitive
  },
  { /* threeD fields */
    /* shadowdraw	*/	XtInheritXaw3dShadowDraw /*,*/
    /* shadowboxdraw	*/	/*XtInheritXaw3dShadowBoxDraw*/
  },
  { /* slider fields */
    /* ignore		*/	0
  }

};

WidgetClass sliderWidgetClass = (WidgetClass)&sliderClassRec;

#define NoButton -1
#define PICKLENGTH(widget, x, y) \
    ((widget->slider.orientation == XtorientHorizontal) ? x : y)
#define MIN(x,y)	((x) < (y) ? (x) : (y))
#define MAX(x,y)	((x) > (y) ? (x) : (y))

static void ClassInitialize()
{
    XawInitializeWidgetSet();
    XtAddConverter( XtRString, XtROrientation, XmuCvtStringToOrientation,
		    (XtConvertArgList)NULL, (Cardinal)0 );
}

#define MARGIN(sbw) (sbw)->threeD.shadow_width

/* 
 The original Xaw Slider's FillArea *really* relied on the fact that the 
 server was going to clip at the window boundaries; so the logic was really
 rather sloppy.  To avoid drawing over the shadows and the arrows requires 
 some extra care...  Hope I didn't make any mistakes.
*/
static void FillArea (sbw, top, bottom, fill)
    SliderWidget sbw;
    Position top, bottom;
    int fill;
{
    int tlen = bottom - top;	/* length of thumb in pixels */
    int sw, margin, floor;
    int lx, ly, lw, lh;

    if (bottom <= 0 || bottom <= top) 
	return;
    if ((sw = sbw->threeD.shadow_width) < 0) 
	sw = 0;
    margin = MARGIN (sbw);
    floor = sbw->slider.length - margin;

    if (sbw->slider.orientation == XtorientHorizontal) {
	lx = ((top < margin) ? margin : top);
	ly = sw;
	lw = ((bottom > floor) ? floor - top : tlen);
/* CHECKIT	lw = (((top + tlen) > floor) ? floor - top : tlen); */
	lh = sbw->core.height - 2 * sw;
    } else {
	lx = sw;
	ly = ((top < margin) ? margin : top);
	lw = sbw->core.width - 2 * sw;
/* CHECKIT	lh = (((top + tlen) > floor) ? floor - top : tlen); */
	lh = ((bottom > floor) ? floor - top : tlen);
    }
    if (lh <= 0 || lw <= 0) return;
    if (fill) {
	XFillRectangle(XtDisplay((Widget) sbw), XtWindow((Widget) sbw), 
			sbw->slider.gc, 
			lx, ly, (unsigned int) lw, (unsigned int) lh);
    } else {
	XClearArea (XtDisplay((Widget) sbw), XtWindow((Widget) sbw), 
			lx, ly, (unsigned int) lw, (unsigned int) lh, 
			FALSE);
    }
}

/* Paint the thumb in the area specified by sbw->top and
   sbw->shown.  The old area is erased.  The painting and
   erasing is done cleverly so that no flickering will occur. */

static void PaintThumb (sbw, event)
    SliderWidget sbw;
    XEvent *event; 
{
    Dimension s                   = sbw->threeD.shadow_width;
    Position  oldtop              = sbw->slider.topLoc;
    Position  oldbot              = oldtop + sbw->slider.shownLength;
    Dimension margin              = MARGIN (sbw);
    Dimension tzl                 = sbw->slider.length - margin - margin;
    Position  newtop, newbot;    
    Position  floor               = sbw->slider.length - margin;

    newtop = margin + (int)(tzl * sbw->slider.top);
    newbot = newtop + (int)(tzl * sbw->slider.shown);
    if (sbw->slider.shown < 1.) newbot++;
    if (newbot < newtop + (int)sbw->slider.min_thumb +
                        2 * (int)sbw->threeD.shadow_width) 
      newbot = newtop + sbw->slider.min_thumb +
                        2 * sbw->threeD.shadow_width;
    if ( newbot >= floor ) {
	newtop = floor-(newbot-newtop)+1;
	newbot = floor;
    }

    sbw->slider.topLoc = newtop;
    sbw->slider.shownLength = newbot - newtop;
    if (XtIsRealized ((Widget) sbw)) {
      /*  3D thumb wanted ?
       */
      if (s) 
	  {
	  Position	oldmid = (oldtop + oldbot) / 2;
	  Position	newmid = (newtop + newbot) / 2;
	      
	  if (newtop < oldtop) FillArea(sbw, oldtop, oldtop + s, 0);
          if (newtop > oldtop) FillArea(sbw, oldtop, MIN(newtop, oldmid), 0);
	  if (newmid < oldmid)
	  {
	      FillArea(sbw, oldmid, oldmid + s, 0);
	      FillArea(sbw, MAX(newmid, oldtop), oldmid, 0);
	  }
	  if (newmid > oldmid)
	  {
	      FillArea(sbw, oldmid, MIN(newmid, oldbot), 0);
	      FillArea(sbw, oldmid - s, oldmid, 0);
	  }
          if (newbot < oldbot) FillArea(sbw, MAX(newbot, oldmid), oldbot, 0);
	  if (newbot > oldbot) FillArea(sbw, oldbot - s, oldbot, 0);

          if (sbw->slider.orientation == XtorientHorizontal) 
	      {
	      _ShadowSurroundedBox((Widget)sbw, (ThreeDWidget)sbw,
		  newtop, s, newmid, sbw->core.height - s,
		  sbw->threeD.relief, TRUE);
	      _ShadowSurroundedBox((Widget)sbw, (ThreeDWidget)sbw,
		  newmid, s, newbot, sbw->core.height - s,
		  sbw->threeD.relief, TRUE);
	      }
	  else 
	      {
	      _ShadowSurroundedBox((Widget)sbw, (ThreeDWidget)sbw,
		  s, newtop, sbw->core.width - s, newmid,
		  sbw->threeD.relief, TRUE);
	      _ShadowSurroundedBox((Widget)sbw, (ThreeDWidget)sbw,
		  s, newmid, sbw->core.width - s, newbot,
		  sbw->threeD.relief, TRUE);
	      }
	  }
      else 
	  {
	  /* 
	    Note to Mitch: FillArea is (now) correctly implemented to 
	    not draw over shadows or the arrows. Therefore setting clipmasks 
	    doesn't seem to be necessary.  Correct me if I'm wrong!
	  */
          if (newtop < oldtop) FillArea(sbw, newtop, MIN(newbot, oldtop), 1);
          if (newtop > oldtop) FillArea(sbw, oldtop, MIN(newtop, oldbot), 0);
          if (newbot < oldbot) FillArea(sbw, MAX(newbot, oldtop), oldbot, 0);
          if (newbot > oldbot) FillArea(sbw, MAX(newtop, oldbot), newbot, 1);
	  }
    }
}


/*	Function Name: Destroy
 *	Description: Called as the slider is going away...
 *	Arguments: w - the slider.
 *	Returns: nonw
 */
static void Destroy (w)
    Widget w;
{
    SliderWidget sbw = (SliderWidget) w;
    XtReleaseGC (w, sbw->slider.gc);
}

/*	Function Name: CreateGC
 *	Description: Creates the GC.
 *	Arguments: w - the slider widget.
 *	Returns: none. 
 */

static void CreateGC (w)
    Widget w;
{
    SliderWidget sbw = (SliderWidget) w;
    XGCValues gcValues;
    XtGCMask mask;
    unsigned int depth = 1;

    if (sbw->slider.thumb == XtUnspecifiedPixmap) {
        sbw->slider.thumb = XmuCreateStippledPixmap (XtScreen(w), 
					(Pixel) 1, (Pixel) 0, depth);
    } else if (sbw->slider.thumb != None) {
	Window root;
	int x, y;
	unsigned int width, height, bw;
	if (XGetGeometry (XtDisplay(w), sbw->slider.thumb, &root, &x, &y,
			 &width, &height, &bw, &depth) == 0) {
	    XtAppError (XtWidgetToApplicationContext (w),
	       "Slider Widget: Could not get geometry of thumb pixmap.");
	}
    }

    gcValues.foreground = sbw->slider.foreground;
    gcValues.background = sbw->core.background_pixel;
    mask = GCForeground | GCBackground;

    if (sbw->slider.thumb != None) {
	if (depth == 1) {
	    gcValues.fill_style = FillOpaqueStippled;
	    gcValues.stipple = sbw->slider.thumb;
	    mask |= GCFillStyle | GCStipple;
	}
	else {
	    gcValues.fill_style = FillTiled;
	    gcValues.tile = sbw->slider.thumb;
	    mask |= GCFillStyle | GCTile;
	}
    }
    /* the creation should be non-caching, because */
    /* we now set and clear clip masks on the gc returned */
    sbw->slider.gc = XtGetGC (w, mask, &gcValues);
}

static void SetDimensions (sbw)
    SliderWidget sbw;
{
    if (sbw->slider.orientation == XtorientVertical) {
	sbw->slider.length = sbw->core.height;
	sbw->slider.thickness = sbw->core.width;
    } else {
	sbw->slider.length = sbw->core.width;
	sbw->slider.thickness = sbw->core.height;
    }
}

/* ARGSUSED */
static void Initialize( request, new, args, num_args )
    Widget request;		/* what the client asked for */
    Widget new;			/* what we're going to give him */
    ArgList args;
    Cardinal *num_args;
{
    SliderWidget sbw = (SliderWidget) new;

    CreateGC (new);

    if (sbw->core.width == 0)
	sbw->core.width = (sbw->slider.orientation == XtorientVertical)
	    ? sbw->slider.thickness : sbw->slider.length;

    if (sbw->core.height == 0)
	sbw->core.height = (sbw->slider.orientation == XtorientHorizontal)
	    ? sbw->slider.thickness : sbw->slider.length;

  //sbw->threeD.shadow_width = 1;
    
    SetDimensions (sbw);
    sbw->slider.direction = 0;
    sbw->slider.topLoc = 0;
    sbw->slider.shownLength = sbw->slider.min_thumb;
}

static void Realize (w, valueMask, attributes)
    Widget w;
    Mask *valueMask;
    XSetWindowAttributes *attributes;
{
    SliderWidget sbw = (SliderWidget) w;
    sbw->slider.inactiveCursor =
      (sbw->slider.orientation == XtorientVertical)
	? sbw->slider.verCursor
	: sbw->slider.horCursor;

    XtVaSetValues (w, XtNcursor, sbw->slider.inactiveCursor, NULL);
    /* 
     * The Simple widget actually stuffs the value in the valuemask. 
     */
    
    (*sliderWidgetClass->core_class.superclass->core_class.realize)
	(w, valueMask, attributes);
}

/* ARGSUSED */
static Boolean SetValues (current, request, desired, args, num_args)
    Widget  current,		/* what I am */
	    request,		/* what he wants me to be */
	    desired;		/* what I will become */
    ArgList args;
    Cardinal *num_args;
{
    SliderWidget sbw = (SliderWidget) current;
    SliderWidget dsbw = (SliderWidget) desired;
    Boolean redraw = FALSE;

/*
 * If these values are outside the acceptable range ignore them...
 */

    if (dsbw->slider.top < 0.0 || dsbw->slider.top > 1.0)
        dsbw->slider.top = sbw->slider.top;

    if (dsbw->slider.shown < 0.0 || dsbw->slider.shown > 1.0)
        dsbw->slider.shown = sbw->slider.shown;

/*
 * Change colors and stuff...
 */
    if (XtIsRealized (desired)) {
	if (sbw->slider.foreground != dsbw->slider.foreground ||
	    sbw->core.background_pixel != dsbw->core.background_pixel ||
	    sbw->slider.thumb != dsbw->slider.thumb) {
	    XtReleaseGC (desired, sbw->slider.gc);
	    CreateGC (desired);
	    redraw = TRUE;
	}
	if (sbw->slider.top != dsbw->slider.top ||
	    sbw->slider.shown != dsbw->slider.shown)
	    redraw = TRUE;
    }
    return redraw;
}

static void Resize (w)
    Widget w;
{
    /* ForgetGravity has taken care of background, but thumb may
     * have to move as a result of the new size. */
    SetDimensions ((SliderWidget) w);
    Redisplay (w, (XEvent*) NULL, (Region)NULL);
}


/* ARGSUSED */
static void Redisplay (w, event, region)
    Widget w;
    XEvent *event;
    Region region;
{
    SliderWidget sbw = (SliderWidget) w;
    SliderWidgetClass swclass = (SliderWidgetClass) XtClass (w);
    int x, y;
    unsigned int width, height;

    (*swclass->threeD_class.shadowdraw) (w, event, region, sbw->threeD.relief, FALSE);

    if (sbw->slider.orientation == XtorientHorizontal) {
	x = sbw->slider.topLoc;
	y = 1;
	width = sbw->slider.shownLength;
	height = sbw->core.height - 2;
    } else {
	x = 1;
	y = sbw->slider.topLoc;
	width = sbw->core.width - 2;
	height = sbw->slider.shownLength;
    }
    if (region == NULL ||
	XRectInRegion (region, x, y, width, height) != RectangleOut) {
	/* Forces entire thumb to be painted. */
	sbw->slider.topLoc = -(sbw->slider.length + 1);
	PaintThumb (sbw, event); 
    }
}


static Boolean CompareEvents (oldEvent, newEvent)
    XEvent *oldEvent, *newEvent;
{
#define Check(field) if (newEvent->field != oldEvent->field) return False;

    Check(xany.display);
    Check(xany.type);
    Check(xany.window);

    switch (newEvent->type) {
    case MotionNotify:
	Check(xmotion.state); 
	break;
    case ButtonPress:
    case ButtonRelease:
	Check(xbutton.state);
	Check(xbutton.button); 
	break;
    case KeyPress:
    case KeyRelease:
	Check(xkey.state);
	Check(xkey.keycode); 
	break;
    case EnterNotify:
    case LeaveNotify:
	Check(xcrossing.mode);
	Check(xcrossing.detail);
	Check(xcrossing.state); 
	break;
    }
#undef Check

    return True;
}

struct EventData {
    XEvent *oldEvent;
    int count;
};

static Bool PeekNotifyEvent (dpy, event, args)
    Display *dpy;
    XEvent *event;
    char *args;
{
    struct EventData *eventData = (struct EventData*)args;

    return ((++eventData->count == QLength(dpy)) /* since PeekIf blocks */
	    || CompareEvents(event, eventData->oldEvent));
}


static Boolean LookAhead (w, event)
    Widget w;
    XEvent *event;
{
    XEvent newEvent;
    struct EventData eventData;

    if (QLength (XtDisplay (w)) == 0) return False;

    eventData.count = 0;
    eventData.oldEvent = event;

    XPeekIfEvent (XtDisplay (w), &newEvent, PeekNotifyEvent, (char*)&eventData);

    return CompareEvents (event, &newEvent);
}


static void ExtractPosition (event, x, y)
    XEvent *event;
    Position *x, *y;		/* RETURN */
{
    switch( event->type ) {
    case MotionNotify:
	*x = event->xmotion.x;	 
	*y = event->xmotion.y;	  
	break;
    case ButtonPress:
    case ButtonRelease:
	*x = event->xbutton.x;   
	*y = event->xbutton.y;   
	break;
    case KeyPress:
    case KeyRelease:
	*x = event->xkey.x;      
	*y = event->xkey.y;	  
	break;
    case EnterNotify:
    case LeaveNotify:
	*x = event->xcrossing.x; 
	*y = event->xcrossing.y; 
	break;
    default:
	*x = 0; *y = 0;
    }
}

/* ARGSUSED */
static void StartScroll (w, event, params, num_params )
    Widget w;
    XEvent *event;
    String *params;		/* direction: Back|Forward|Smooth */
    Cardinal *num_params;	/* we only support 1 */
{
    SliderWidget sbw = (SliderWidget) w;
    Cursor cursor;
    char direction;

    if (sbw->slider.direction != 0) return; /* if we're already scrolling */
    if (*num_params > 0) 
	direction = *params[0];
    else
	direction = 'C';

    sbw->slider.direction = direction;

    cursor = (sbw->slider.orientation == XtorientVertical ?
	      sbw->slider.verCursor : sbw->slider.horCursor);

    XtVaSetValues (w, XtNcursor, cursor, NULL);
    XFlush (XtDisplay (w));
}

/*
 * Make sure the first number is within the range specified by the other
 * two numbers.
 */

static int InRange(num, small, big)
    int num, small, big;
{
    return (num < small) ? small : ((num > big) ? big : num);
}

/*
 * Same as above, but for floating numbers. 
 */

static float FloatInRange(num, small, big)
    float num, small, big;
{
    return (num < small) ? small : ((num > big) ? big : num);
}


static void NotifyScroll (w, event, params, num_params)
    Widget w;
    XEvent *event;
    String *params;		/* style: Proportional|FullLength */
    Cardinal *num_params;	/* we only support 1 */
{
    SliderWidget sbw = (SliderWidget) w;
    long call_data;
    char style;
    Position x, y;

    if (sbw->slider.direction == 0) return; /* if no StartScroll */
    if (LookAhead (w, event)) return;
    if (*num_params > 0) 
	style = *params[0];
    else
	style = 'P';

    switch (style) {
    case 'P':    /* Proportional */
    case 'p':
	ExtractPosition (event, &x, &y);
	call_data = 
	    InRange (PICKLENGTH (sbw, x, y), 0, (int) sbw->slider.length); 
	break;

    case 'F':    /* FullLength */
    case 'f':    
	call_data = sbw->slider.length; 
	break;
    }
    switch (sbw->slider.direction) {
    case 'B':
    case 'b':    
	call_data = -call_data;
	/* fall through */

    case 'F':
    case 'f':
	XtCallCallbacks (w, XtNscrollProc, (XtPointer)call_data);
	break;

    case 'C':
    case 'c':
	/* NotifyThumb has already called the thumbProc(s) */
	break;
    }
}

/* ARGSUSED */
static void EndScroll(w, event, params, num_params )
    Widget w;
    XEvent *event;		/* unused */
    String *params;		/* unused */
    Cardinal *num_params;	/* unused */
{
    SliderWidget sbw = (SliderWidget) w;

    XtVaSetValues (w, XtNcursor, sbw->slider.inactiveCursor, NULL);
    XFlush (XtDisplay (w));
    sbw->slider.direction = 0;
}

static float FractionLoc (sbw, x, y)
    SliderWidget sbw;
    int x, y;
{
    float   result;
    int margin;
    float   height, width;

    margin = MARGIN (sbw);
    x -= margin;
    y -= margin;
    height = sbw->core.height - 2 * margin;
    width = sbw->core.width - 2 * margin;
    result = PICKLENGTH (sbw, x / width, y / height);
    return FloatInRange(result, 0.0, 1.0);
}


static void MoveThumb (w, event, params, num_params)
    Widget w;
    XEvent *event;
    String *params;		/* unused */
    Cardinal *num_params;	/* unused */
{
    SliderWidget sbw = (SliderWidget) w;
    Position x, y;
    float loc, t, s;

    if (sbw->slider.direction == 0) return; /* if no StartScroll */
    
    if (LookAhead (w, event)) return;

    if (!event->xmotion.same_screen) return;

    ExtractPosition (event, &x, &y);
    
    loc = FractionLoc (sbw, x, y);
    t = sbw->slider.top;
    s = sbw->slider.shown;
    sbw->slider.picked = 0.5 * s;

    if (sbw->slider.pick_top)
      sbw->slider.top = loc;
    else {
      sbw->slider.top = loc - sbw->slider.picked;
      if (sbw->slider.top < 0.0) sbw->slider.top = 0.0;
    }

#if 0
    /* this breaks many text-line scrolls */
    if (sbw->slider.top + sbw->slider.shown > 1.0)
      sbw->slider.top = 1.0 - sbw->slider.shown;
#endif
    PaintThumb (sbw, event);
    XFlush (XtDisplay (w));	/* re-draw it before Notifying */
}


/* ARGSUSED */
static void NotifyThumb (w, event, params, num_params )
    Widget w;
    XEvent *event;
    String *params;		/* unused */
    Cardinal *num_params;	/* unused */
{
    register SliderWidget	sbw = (SliderWidget) w;
    Dimension			margin = MARGIN(sbw);
    float			tzl = sbw->slider.length - 2*margin;
    float			top = sbw->slider.top
				    * tzl / (tzl - sbw->slider.shownLength);

    top = FloatInRange(top, 0.0, 1.0);
    
    if (sbw->slider.direction == 0) return; /* if no StartScroll */
    
    if (LookAhead (w, event)) return;

    /* thumbProc is not pretty, but is necessary for backwards
       compatibility on those architectures for which it work{s,ed};
       the intent is to pass a (truncated) float by value. */
/* #ifdef XAW_ARROW_SLIDERS */
    /* This corrects for rounding errors: If the thumb is moved to the end of
       the scrollable area sometimes the last line/column is not displayed.
       This can happen when the integer number of the top line or leftmost
       column to be be displayed is calculated from the float value
       sbw->slider.top. The numerical error of this rounding problem is
       very small. We therefore add a small value which then forces the
       next line/column (the correct one) to be used. Since we can expect
       that the resolution of display screens will not be higher then
       10000 text lines/columns we add 1/10000 to the top position. The
       intermediate variable `top' is used to avoid erroneous summing up
       corrections (can this happen at all?). If the arrows are not displayed
       there is no problem since in this case there is always a constant
       integer number of pixels the thumb must be moved in order to scroll
       to the next line/column. */
    /* Removed the dependancy on slider arrows. Xterm as distributed in
       X11R6.6 by The XFree86 Project wants this correction, with or without
       the arrows. */
    top += 0.0001;
/* #endif */
    XtCallCallbacks (w, XtNthumbProc, *(XtPointer*)&top);
    XtCallCallbacks (w, XtNjumpProc, (XtPointer)&top);
}



/************************************************************
 *
 *  Public routines. 
 *
 ************************************************************/

/* Set the scroll bar to the given location. */

#if NeedFunctionPrototypes
void XawSliderSetThumb (Widget w, 
#if NeedWidePrototypes
			double top, double shown)
#else
			float top, float shown)
#endif
#else
void XawSliderSetThumb (w, top, shown)
    Widget w;
    float top, shown;
#endif
{
    SliderWidget sbw = (SliderWidget) w;
    Dimension margin = MARGIN(sbw);
    float tzl = sbw->slider.length - 2*margin;

    top *= (tzl - sbw->slider.shownLength) / tzl;
    
#ifdef WIERD
    fprintf(stderr,"< vSliderSetThumb w=%p, top=%f, shown=%f\n", 
	    w,top,shown);
#endif

    if (sbw->slider.direction == 'c') return; /* if still thumbing */

    sbw->slider.top = (top > 1.0) ? 1.0 :
				(top >= 0.0) ? top : sbw->slider.top;

    sbw->slider.shown = (shown > 1.0) ? 1.0 :
				(shown >= 0.0) ? shown : sbw->slider.shown;

    PaintThumb (sbw, (XEvent*)NULL);
}

