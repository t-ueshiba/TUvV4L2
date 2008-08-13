/*
 *  平成9-19年（独）産業技術総合研究所 著作権所有
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
 *  Copyright 1997-2007.
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
 *  $Id: vCanvas.c,v 1.3 2008-08-13 06:06:35 ueshiba Exp $
 */
#include <X11/IntrinsicP.h>
#include <X11/StringDefs.h>
#include <X11/Xmu/Converters.h>
#include <X11/Xaw3d/XawInit.h>
#include "vCanvasP_.h"

/****************************************************************
*  vCanvas Resources						*
****************************************************************/

static XtResource resources[] =
{
    {XtNbackingStore, XtCBackingStore, XtRBackingStore, sizeof(int),
     XtOffsetOf(VCanvasRec, vcanvas.backing_store), XtRString, XtEalways},
    {XtNginitCallback, XtCGinitCallback, XtRCallback, sizeof(XtPointer),
     XtOffsetOf(VCanvasRec, vcanvas.ginit_callbacks),
     XtRImmediate, (XtPointer)NULL}
};

/****************************************************************
*  Full class record constant					*
****************************************************************/
static void	ClassInitialize(void);
static void	Realize(Widget, XtValueMask*, XSetWindowAttributes*);
static Boolean	SetValues(Widget, Widget, Widget, ArgList, Cardinal*);

VCanvasClassRec vCanvasClassRec = {
{
/* core_class fields      */
    /* superclass         */    (WidgetClass) &coreClassRec,
    /* class_name         */    "vcanvas",
    /* widget_size        */    sizeof(VCanvasRec),
    /* class_initialize   */    ClassInitialize,
    /* class_part_init    */	NULL,
    /* class_inited       */	FALSE,
    /* initialize         */    NULL,
    /* initialize_hook    */	NULL,
    /* realize            */    Realize,
    /* actions            */    NULL,
    /* num_actions	  */	0,
    /* resources          */    resources,
    /* num_resources      */    XtNumber(resources),
    /* xrm_class          */    NULLQUARK,
    /* compress_motion	  */	TRUE,
    /* compress_exposure  */	TRUE,
    /* compress_enterleave*/	TRUE,
    /* visible_interest   */    FALSE,
    /* destroy            */    NULL,
    /* resize             */    XtInheritResize,
    /* expose             */    NULL,
    /* set_values         */    SetValues,
    /* set_values_hook    */	NULL,
    /* set_values_almost  */    XtInheritSetValuesAlmost,
    /* get_values_hook    */	NULL,
    /* accept_focus       */    NULL,
    /* version            */	XtVersion,
    /* callback_private   */    NULL,
    /* tm_table           */    NULL,
    /* query_geometry     */	XtInheritQueryGeometry,
    /* display_accelerator*/	XtInheritDisplayAccelerator,
    /* extension          */	NULL
  },{
/* vcanvas class fields */
    /* empty		  */	0,
  }
};

WidgetClass vCanvasWidgetClass = (WidgetClass)&vCanvasClassRec;

/****************************************************************
*   Private Routines						*
****************************************************************/
static void
ClassInitialize(void)
{
    XawInitializeWidgetSet();
    XtAddConverter(XtRString, XtRBackingStore, XmuCvtStringToBackingStore,
		   (XtConvertArgList)NULL, (Cardinal)0);
}

static void
Realize(Widget w, XtValueMask* valueMask, XSetWindowAttributes* attributes)
{
    VCanvasWidget	vcw = (VCanvasWidget)w;
    
    *valueMask |= CWBackingStore;
    attributes->backing_store = vcw->vcanvas.backing_store;
    (*vCanvasWidgetClass->core_class.superclass->core_class.realize)
	(w, valueMask, attributes);
    if (vcw->vcanvas.ginit_callbacks)
	XtCallCallbackList(w, vcw->vcanvas.ginit_callbacks, (XtPointer)NULL);
}

static Boolean
SetValues(Widget current, Widget request, Widget new,
	  ArgList args, Cardinal* num_args)
{
    VCanvasWidget	vcwcurrent = (VCanvasWidget)current;
    VCanvasWidget	vcwnew	   = (VCanvasWidget)new;

  /* need to relayout if h_space or v_space change */
    if (vcwnew->vcanvas.backing_store != vcwcurrent->vcanvas.backing_store)
    {
	if (XtIsRealized(current))
	{
	    unsigned long		valuemask = CWBackingStore;
	    XSetWindowAttributes	attributes;

	    attributes.backing_store = vcwnew->vcanvas.backing_store;
	    XChangeWindowAttributes(XtDisplay(current), XtWindow(current),
				    valuemask, &attributes);
	}
    }

    return False;
}
