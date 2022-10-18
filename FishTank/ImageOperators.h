#ifndef _IMAGE_OPERATORS_H
#define _IMAGE_OPERATORS_H

#include "defs.h"
extern void blitBlend( UCImg &src, 
					  UCImg &dst, 
					  unsigned int dstXOffset, unsigned int dstYOffset, 
					  SimdMode simdMode = SIMD_NONE);

#endif
