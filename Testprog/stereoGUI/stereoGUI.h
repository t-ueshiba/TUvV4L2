/*
 *  $Id$
 */
#ifndef	__TU_V_STEREOGUI_H
#define	__TU_V_STEREOGUI_H

namespace TU
{
namespace v
{
/************************************************************************
*  global data and definitions						*
************************************************************************/
enum
{
    c_Frame,

  // File menu
    c_OpenDisparityMap,
    c_RestoreConfig,
    c_SaveConfig,
    c_SaveMatrices,
    c_SaveRectifiedImages,
    c_SaveThreeD,
    c_SaveThreeDImage,
    
  // Camera control.
    c_ContinuousShot,
    c_OneShot,
    c_Cursor,
    c_DisparityLabel,
    c_Disparity,
    c_Trigger,

  // Stereo matching parameters
    c_Algorithm,
    c_SAD,
    c_GuidedFilter,
    c_TreeFilter,
    c_Binocular,
    c_DoHorizontalBackMatch,
    c_DoVerticalBackMatch,
    c_DisparitySearchWidth,
    c_DisparityMax,
    c_DisparityInconsistency,
    c_WindowSize,
    c_IntensityDiffMax,
    c_DerivativeDiffMax,
    c_Blend,
    c_Regularization,
    c_DepthRange,
    c_WMF,
    c_WMFWindowSize,
    c_WMFSigma,
    c_RefineDisparity,
    
  // Viewing control.
    c_DrawMode,
    c_Texture,
    c_Polygon,
    c_Mesh,
    c_MoveViewpoint,
    c_GazeDistance,
    c_SwingView,
    c_StereoView,
    c_Refresh,
};

}	// namespace v
}	// namespace TU
#endif	// !__TU_V_STEREOGUI_H

