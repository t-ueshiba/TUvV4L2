/*
 *  $Id: IIDCNode.cc 1655 2014-10-03 01:37:50Z ueshiba $
 */
#include "TU/IIDC++.h"
#include <libraw1394/csr.h>
#include <algorithm>		// for std::find_if()

namespace TU
{
/************************************************************************
*  class IIDCNode							*
************************************************************************/
IIDCNode::IIDCNode()
{
}
    
IIDCNode::~IIDCNode()
{
    _nodes.remove(this);
}
    
//! このノードに結び付けられている機器固有の64bit IDを返す
uint64_t
IIDCNode::globalUniqueId() const
{
    uint64_t	hi = readQuadletFromConfigROM(0x0c),
		lo = readQuadletFromConfigROM(0x10);
    return (hi << 32) | lo;
}

//! このノードのunit_spec_IDを返す
uint32_t
IIDCNode::unitSpecId() const
{
    return readValueFromUnitDirectory(0x12);
}

//! このノードのコマンドレジスタのベースアドレスを返す
IIDCNode::nodeaddr_t
IIDCNode::commandRegisterBase() const
{
    return CSR_REGISTER_BASE + 4 * readValueFromUnitDependentDirectory(0x40);
}

//! この機器のglobal unique IDが既に登録されているか調べる
/*
  \return	登録されていればtrue, そうでなければfalse
*/
bool
IIDCNode::isUnique() const
{
    return std::find_if(_nodes.begin(), _nodes.end(),
			[=](const IIDCNode* node)
			{
			    return node->globalUniqueId() == globalUniqueId();
			}) == _nodes.end();
}

//! この機器を新たに登録する
void
IIDCNode::addToList() const
{
    _nodes.push_back(this);
}
    
//! 与えられたkeyに対する値をUnit Dependent Directoryから読み出す
/*!
  \param key	keyすなわち4byteの並びのMSB側8bit
 */
uint32_t
IIDCNode::readValueFromUnitDependentDirectory(uint8_t key) const
{
    auto	offset = readValueFromUnitDirectory(0xd4);
    
    return readValueFromDirectory(key, offset);
}

//! 与えられたkeyに対する値をUnit Directoryから読み出す
/*!
  \param key	keyすなわち4byteの並びのMSB側8bit
 */
uint32_t
IIDCNode::readValueFromUnitDirectory(uint8_t key) const
{
  // Read length of Bus Info Block and skip it.
    uint32_t	offset = 0;
    auto	quad = readQuadletFromConfigROM(offset);
    offset += 4 * (1 + (quad >> 24));

  // Read unit_directory_offset.
    uint32_t	tmpOffset = readValueFromDirectory(0xd1, offset);
    offset += 4 * tmpOffset;

  // Read unit_spec_ID and return it.
    return readValueFromDirectory(key, offset);
}

uint32_t
IIDCNode::readValueFromDirectory(uint8_t key, uint32_t& offset) const
{
  // Read length of the directory in quadlets.
    auto	quad = readQuadletFromConfigROM(offset);
    uint32_t	length = quad >> 16;
    offset += 4;

  // Read each field of the directory.
    for (uint32_t i = 0; i < length; ++i)
    {
	quad = readQuadletFromConfigROM(offset);
	if (uint8_t(quad >> 24) == key)
	    return (quad & 0xffffff);
	offset += 4;
    }

    throw std::runtime_error("TU::IIDCNode::readValueFromDirectory: field with specified key not found!!");

    return ~0;
}

IIDCNode::quadlet_t
IIDCNode::readQuadletFromConfigROM(uint32_t offset) const
{
    return readQuadlet(CSR_REGISTER_BASE + CSR_CONFIG_ROM + offset);
}

std::list<const IIDCNode*>	IIDCNode::_nodes;
}
