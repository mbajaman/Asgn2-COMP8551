#ifndef __DEFS__H
#define __DEFS__H

#include "CImg.h"

typedef cimg_library::CImg<unsigned char> UCImg;

typedef enum { SIMD_NONE = 0, SIMD_EMMX, SIMD_EMMX_INTRINSICS, SIMD_LAST} SimdMode;

#endif