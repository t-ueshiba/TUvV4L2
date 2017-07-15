/*!
  \file		simd.h
  \author	Toshio UESHIBA
  \brief	CPUのSIMD命令に関連するクラスと関数の定義と実装
*/
#if !defined(TU_SIMD_SIMD_H)
#define TU_SIMD_SIMD_H

#include "TU/simd/config.h"

#if defined(SIMD)
#  include "TU/simd/vec.h"
#  include "TU/simd/allocator.h"

#  include "TU/simd/load_store.h"
#  include "TU/simd/zero.h"
#  include "TU/simd/cast.h"
#  include "TU/simd/insert_extract.h"
#  include "TU/simd/shift.h"
#  include "TU/simd/bit_shift.h"
#  include "TU/simd/dup.h"
#  include "TU/simd/cvt.h"
#  include "TU/simd/logical.h"
#  include "TU/simd/compare.h"
#  include "TU/simd/select.h"
#  include "TU/simd/arithmetic.h"
#  include "TU/simd/misc.h"
#  include "TU/simd/transform.h"
#  include "TU/simd/lookup.h"

#  include "TU/simd/load_store_iterator.h"
#  include "TU/simd/cvtdown_iterator.h"
#  include "TU/simd/cvtup_iterator.h"
#  include "TU/simd/shift_iterator.h"

#  include "TU/simd/algorithm.h"
#  include "TU/simd/BufTraits.h"
#endif

#endif	// !TU_SIMD_SIMD_H
