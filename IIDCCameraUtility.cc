/*
 *  $Id$
 */
#include "TU/IIDCCameraUtility.h"

namespace TU
{
/************************************************************************
*  class IIDCCameraArray						*
************************************************************************/
constexpr const char*	IIDCCameraArray::DEFAULT_CAMERA_NAME;
constexpr const char*	IIDCCameraArray::DEFAULT_CONFIG_DIRS;
    
//! 設定ファイルを読み込んでIIDCデジタルカメラの配列を初期化する.
/*!
  \param name		カメラ名
  \param dirs		カメラ設定ファイルの探索ディレクトリ名の並び
  \param speed		FireWireバスの速度
*/
inline void
IIDCCameraArray::restore(const char* name, const char* dirs,
			 IIDCCamera::Speed speed)
{
  // 設定ファイルのfull path名を生成し, ファイルをオープンする.
    std::ifstream	in;
    _fullName = openFile(in,
			 std::string(name != 0 ? name : DEFAULT_CAMERA_NAME),
			 std::string(dirs != 0 ? dirs : DEFAULT_CONFIG_DIRS),
			 ".conf");
    in >> *this;
    for (auto& camera : *this)
	camera.setSpeed(speed);
}

}
