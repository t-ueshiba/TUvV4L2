/*
 *  $Id: FileSelection.cc,v 1.4 2003-01-10 00:33:11 ueshiba Exp $
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

  // Set file names in current working directory to _filenames.
    _filenames[0] = 0;
    changeDirectory(_dirname);
}

FileSelection::~FileSelection()
{
    for (int i = 0; _filenames[i] != 0; ++i)
	delete [] _filenames[i];
}

bool
FileSelection::open(std::ifstream& in)
{
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

#if defined(__GNUG__) || defined(__INTEL_COMPILER)
#  include "TU/Array++.cc"
#endif
