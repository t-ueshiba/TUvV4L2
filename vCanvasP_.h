/* ===============================================================
 vCanvasP_.h - Private definitions for vCanvas widget

 Copyright (C) 1997,  Toshio Ueshiba

=============================================================== */

#ifndef vCanvasP_h
#define vCanvasP_h

#include "vCanvas_.h"

/* New fields for the vCanvas widget class record */
typedef struct _VCanvasClass
{
    int			empty;		/* need something */
} VCanvasClassPart;

/* Full class record declaration */
typedef struct _VCanvasClassRec
{
    CoreClassPart	core_class;
    VCanvasClassPart	vcanvas_class;
} VCanvasClassRec;

extern VCanvasClassRec vCanvasClassRec;

/* New fields for the vCanvas widget record */
typedef struct
{
    int			backing_store;	 /* Whether we allow backing store */
    XtCallbackList	ginit_callbacks; /* called once when realized	   */
} VCanvasPart;


/****************************************************************

	Full instance record declaration

 ****************************************************************/

typedef struct _VCanvasRec
{
    CorePart		core;
    VCanvasPart		vcanvas;
} VCanvasRec;

#endif
