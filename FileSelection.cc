/*
 *  平成14-19年（独）産業技術総合研究所 著作権所有
 *  
 *  創作者：植芝俊夫
 *
 *  本プログラムは（独）産業技術総合研究所の職員である植芝俊夫が創作し，
 *  （独）産業技術総合研究所が著作権を所有する秘密情報です．著作権所有
 *  者による許可なしに本プログラムを使用，複製，改変，第三者へ開示する
 *  等の行為を禁止します．
 *  
 *  このプログラムによって生じるいかなる損害に対しても，著作権所有者お
 *  よび創作者は責任を負いません。
 *
 *  Copyright 2002-2007.
 *  National Institute of Advanced Industrial Science and Technology (AIST)
 *
 *  Creator: Toshio UESHIBA
 *
 *  [AIST Confidential and all rights reserved.]
 *  This program is confidential. Any using, copying, changing or
 *  giving any information concerning with this program to others
 *  without permission by the copyright holder are strictly prohibited.
 *
 *  [No Warranty.]
 *  The copyright holder or the creator are not responsible for any
 *  damages caused by using this program.
 *
 *  $Id: FileSelection.cc,v 1.10 2012-08-29 21:17:18 ueshiba Exp $  
 */
#include <unistd.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <errno.h>
#include <algorithm>
#include "TU/v/FileSelection.h"
#include "TU/v/Notify.h"
#include "TU/v/Confirm.h"

namespace TU
{
namespace v
{
/************************************************************************
*  static data								*
************************************************************************/
enum	{c_Directory, c_FileList, c_Cancel, c_FileName};

static CmdDef Cmds[] =
{
    {C_Label,  c_Directory,0, "",	noProp, CA_NoBorder, 0, 0, 2, 1, 0},
    {C_List,   c_FileList, 0, "",	noProp, CA_None, 0, 1, 1, 1, 15},
    {C_Button, c_Cancel,   0, "Cancel", noProp, CA_None, 1, 1, 1, 1, 0},
    {C_TextIn, c_FileName, 0, "",	noProp, CA_None, 0, 2, 1, 1, 0},
    EndOfCmds
};

/************************************************************************
*  static functions							*
************************************************************************/
static mode_t
fileMode(const std::string& filename)
{
    struct stat	statbuf;
    if (::stat(filename.c_str(), &statbuf))	// error ?
	return 0;
    else
	return statbuf.st_mode;
}

/************************************************************************
*  class FileSelection							*
************************************************************************/
FileSelection::FileSelection(Window& parentWindow)
    :ModalDialog(parentWindow, "File selection", Cmds),
     _fullname(), _dirname(), _filenames(1)
{
  // Initialize _dirname to current working directory.
    char	s[1024];
    ::getcwd(s, sizeof(s)/sizeof(s[0]));
    _dirname = s;
    if (_dirname[_dirname.length()-1] != '/')
	_dirname += '/';

  // Set null pointer to the tail of the file name list.
    _filenames[0] = 0;
}

FileSelection::~FileSelection()
{
    for (int i = 0; _filenames[i] != 0; ++i)
	delete [] _filenames[i];
}

bool
FileSelection::open(std::ifstream& in)
{
    changeDirectory(_dirname);

    for (;;)
    {
	show();
	if (_fullname.empty())		// ファイル名が選択されていない？
	    return false;
	in.open(_fullname.c_str());
	if (in)				// 正常にオープンされた?
	    break;
	Notify	notify(*this);
	notify << "Cannot open " << _fullname << ": " << strerror(errno);
	notify.show();
    }
    return true;
}

bool
FileSelection::open(std::ofstream& out)
{
    changeDirectory(_dirname);

    for (;;)
    {
	show();
	if (_fullname.empty())		// ファイル名が選択されていない？
	    return false;
	if (fileMode(_fullname))	// 既存ファイル？
	{
	    Confirm	confirm(*this);
	    confirm << _fullname << " already exists. Override?";
	    if (!confirm.ok())
		continue;
	}
	out.open(_fullname.c_str());
	if (out)			// 正常にオープンされた？
	    break;
	Notify	notify(*this);
	notify << "Cannot open " << _fullname << ": " << strerror(errno);
	notify.show();
    }
    return true;
}

void
FileSelection::callback(CmdId id, CmdVal val)
{
    switch (id)
    {
      case c_Cancel:
	_fullname.erase();
	hide();
	break;
	
      case c_FileList:
      case c_FileName:
      {
	const std::string&
		fullname = fullPathName(id == c_FileList ? _filenames[val] :
					pane().getString(id));
	mode_t	filemode = fileMode(fullname);
	if (id == c_FileList && filemode & S_IFDIR)	   // directory ?
	    changeDirectory(fullname);
	else if ((filemode & S_IFREG) || (filemode == 0))  // normal/new file?
	{
	    _fullname = fullname;
	    hide();
	}
      }
	break;
    }
}

// 指定したdirectoryに移動し，その中のファイルを_filenamesにセットする．
/*!
  \param dirname	移動先のdirectory名．末尾は'\'でなければならない．
*/
void
FileSelection::changeDirectory(const std::string& dirname)
{
  // Get file names in the new working directory.
    DIR*	dirp = ::opendir(dirname.c_str());
    if (dirp == NULL)
    {
	std::cerr << "Failed to open direcotry [" << dirname << "]: "
		  << strerror(errno) << std::endl;
	return;
    }
    getFileNames(dirp, 0);
    ::closedir(dirp);

  // Sort file names.
    std::sort(&_filenames[0], &_filenames[_filenames.dim() - 1], cmp());

  // Append '/' to directory names.
    for (int i = 0; _filenames[i] != 0; ++i)
	if (fileMode(dirname + _filenames[i]) & S_IFDIR)
	{
	    int	len = strlen(_filenames[i]);
	    _filenames[i][len] = '/';
	    _filenames[i][len+1] = '\0';
	}

  // Set file names in scrolling list.
    pane().setProp(c_FileList, (char**)_filenames);

  // Change directory.
    _dirname = dirname;
    pane().setString(c_Directory, _dirname.c_str());
}

void
FileSelection::getFileNames(DIR* dirp, int n)
{
    const dirent* dp = ::readdir(dirp);
    if (dp == NULL)
    {
	for (int i = 0; _filenames[i] != 0; ++i)
	    delete [] _filenames[i];
	_filenames.resize(n+1);
	_filenames[n] = 0;
    }
    else
    {
	char* const	name = new char[strlen(dp->d_name) + 2];
	strcpy(name, dp->d_name);
	getFileNames(dirp, n+1);
	_filenames[n] = name;
    }
}

std::string
FileSelection::fullPathName(const char* name) const
{
    if (!strcmp(name, "./"))
	return _dirname;
    else if (!strcmp(name, "../"))
    {  // _dirnameの末尾が'/'で終わっていると仮定している．
	std::string	fullname = _dirname;
	fullname.erase(fullname.rfind('/'));	// 末尾の'/'を除去．
	std::string::size_type	slash = fullname.rfind('/');
	if (slash != std::string::npos)
	    fullname.erase(slash + 1);		// 最後の'/'以降を除去．
	else
	    fullname = '/';	// 末尾以外に'/'が残らなかったらroot．
	return fullname;
    }
    else
    {  // _dirnameの末尾が'/'で終わっていると仮定している．
	std::string	fullname = _dirname + name;;
	return fullname;
    }
}

}
}
