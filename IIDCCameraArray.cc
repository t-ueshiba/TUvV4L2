/*
 *  $Id$
 */
#include "TU/IIDCCameraUtility.h"
#include "TU/io.h"

namespace TU
{
/************************************************************************
*  class IIDCCameraArray						*
************************************************************************/
constexpr const char*	IIDCCameraArray::DEFAULT_CAMERA_NAME;
constexpr const char*	IIDCCameraArray::DEFAULT_CONFIG_DIRS;
    
//! IIDCデジタルカメラの配列を生成する.
IIDCCameraArray::IIDCCameraArray(size_t ncameras)
    :Array<IIDCCamera>(ncameras), _fullName()
{
}
    
//! 設定ファイルを読み込んでIIDCデジタルカメラの配列を初期化する.
/*!
  \param name		カメラ名
  \param dirs		カメラ設定ファイルの探索ディレクトリ名の並び
  \param speed		FireWireバスの速度
*/
void
IIDCCameraArray::restore(const char* name, const char* dirs,
			 IIDCCamera::Speed speed)
{
  // 設定ファイルのfull path名を生成し, ファイルをオープンする.
    std::ifstream	in;
    _fullName = openFile(in,
			 std::string(name ? name : DEFAULT_CAMERA_NAME),
			 std::string(dirs ? dirs : DEFAULT_CONFIG_DIRS),
			 ".conf");
    in >> *this;
    for (auto& camera : *this)
	camera.setSpeed(speed);
}

//! IIDCデジタルカメラの配列の設定を設定ファイルに書き込む.
void
IIDCCameraArray::save() const
{
    std::ofstream	out(configFile().c_str());
    if (!out)
	throw std::runtime_error("IIDCCameraArray::save(): cannot open " +
				 configFile());
    out << *this;
}

//! カメラのfull path名を返す.
/*!
  \return	カメラのfull path名
*/
const std::string&
IIDCCameraArray::fullName() const
{
    return _fullName;
}
    
//! カメラ設定ファイル名を返す.
/*!
  \return	カメラ設定ファイル名
*/
std::string
IIDCCameraArray::configFile() const
{
    return _fullName + ".conf";
}
    
//! キャリブレーションファイル名を返す.
/*!
  \return	キャリブレーションファイル名
*/
std::string
IIDCCameraArray::calibFile() const
{
    return _fullName + ".calib";
}

}
