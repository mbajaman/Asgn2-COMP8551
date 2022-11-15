#include "ImageOperators.h"
#include <emmintrin.h>

#pragma warning(disable:4018) // signed/unsigned mismatch

#define EMMX_BLEND(comp) \
	t = _mm_loadu_si128((__m128i *) pDst[(comp)]);                      \
	d0 = _mm_unpacklo_epi8(t, zero);                                    \
	d1 = _mm_unpackhi_epi8(t, zero);                                    \
	t = _mm_loadu_si128((__m128i *) pSrc[(comp)]);                      \
	s0 = _mm_unpacklo_epi8(t, zero);									\
	s1 = _mm_unpackhi_epi8(t, zero);									\
	diff0 = _mm_sub_epi16(ff, a0);										\
	diff1 = _mm_sub_epi16(ff, a1);										\
	r0 = _mm_add_epi16(_mm_mullo_epi16(s0, a0), _mm_mullo_epi16(diff0, d0)); \
	r1 = _mm_add_epi16(_mm_mullo_epi16(s1, a1), _mm_mullo_epi16(diff1, d1)); \
	_mm_storeu_si128((__m128i *) pDst[(comp)], _mm_packus_epi16(_mm_srli_epi16(r0, 8), _mm_srli_epi16(r1, 8)))                      

// 
void blitBlend( UCImg &src, UCImg &dst, unsigned int dstXOffset, unsigned int dstYOffset, SimdMode simdMode)
{
	if (src.spectrum() != 4) throw cimg_library::CImgException("blitBlend: Src image is missing ALPHA channel");

	// calcualte our SIMD blend area (defined by X0, Y0 to X1, Y1). Take into account alignment restrictions;

	unsigned int X1 = src.width() + dstXOffset; // right edge of src image
	if (X1 > dst.width()) X1 = dst.width();
	unsigned int X0 = dstXOffset; // left edge of src image

	unsigned int Y1 = src.height() + dstYOffset; // bottom edge of src image
	if (Y1 > dst.height()) Y1 = dst.height();

	unsigned int Y0 = dstYOffset; // top edge of src image
	// TODO: Y0 & Y1 need to be aligned.

	// loop over the area and blend the pixels 
	for (unsigned int y = Y0, srcLine = 0; y < Y1; y++, srcLine++) { // For each row of 1px in the Src image, from top to bottom:
		unsigned char* pSrc[4]; // pointers to the 4 channels of the Src image

		// pSrc holds the color for each row, all reds in [0], all greens in [1], etc.
		// pSrc at an index is a pointer to the first pixel in that row. (which is why it goes up by 16 each loop of the inner loop, when it grabs the next 16px)
		pSrc[0] = src.data(0, srcLine, 0, 0); // 4 bits
		pSrc[1] = src.data(0, srcLine, 0, 1); // 4 bits
		pSrc[2] = src.data(0, srcLine, 0, 2); // 4 bits = 12 at this point
		pSrc[3] = src.data(0, srcLine, 0, 3);

		unsigned char *pDst[4];
		pDst[0] = dst.data(X0, y, 0, 0);
		pDst[1] = dst.data(X0, y, 0, 1);
		pDst[2] = dst.data(X0, y, 0, 2);
		pDst[3] = dst.data(X0, y, 0, 3); // For part 2

		// Fill ffconst with all 255 color values
		// ffconst[8] = {r,g,b,a,r,g,b,0}
		short ffconst[8] = {0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff};

		//*** __m128i maps to 128-bit XMM registers
		// Loads the ffconst above from memory into destination (__mm128i ff)
		__m128i ff = _mm_loadu_si128((__m128i *)ffconst);

		/*
		XMM0 = Used as all 0s (DOESN'T CHANGE)
		XMM1 = For pSrc and pDst content
		XMM2 & XMM3 = Those half alpha value blending shenanigans
		XMM4 = ffconst, which is black, maybe has to do with removing the black from the bubble (DOESN'T CHANGE)
		XMM5 = This modifies ffconst and uses that modified value to calculate XMM6
		XMM6 = Those half alpha value belnding shenanigans BUT for destination and ffconst?
		XMM7 = Keeps the color of the Bubble (a.k.a the outline of it)
		*/
#pragma region EMMX
		if (simdMode == SIMD_EMMX) { // RISC Architecture, much more efficient
			for (unsigned x = X0; x < X1; x += 16) { // For each column of 16px in the Src image, from left to right:
				__asm {
					// Save half of pSrc[3] in xmm2 and other half in xmm3 //

					// Set register xmm0 to 0
					pxor xmm0, xmm0			
					
					// Copy the 32-bit address of [pSrc + 12] (Which is pSrc[3]) into eax
					// Part 2: modify this to use Dst instead of Src
					mov eax, dword ptr[pDst + 12]			

					// XMM1 <- *pSrc[3] Copy the 128-bit contents of that [EAX] points to into xmm1
					movdqu xmm1, [eax];
					
					// Copy the contents of XMM1 over to XMM2
					movdqa xmm2, xmm1;
					
					// Unpack the lower order bits of XMM0 which is the blend factor and put it in XMM2    
					punpcklbw xmm2, xmm0; // XMM2 <- a0, 16bit out of 32
					
					// Copy the contents of XMM1 over to XMM3
					movdqa xmm3, xmm1;

					// Unpack the higher order bits of XMM0 which is the blend factor and put it in XMM3 
					//XMM3 <- a1, 16bit
					punpckhbw xmm3, xmm0;

					/* ========================== RED ========================== */
					/* Blending the red;
					 *Save half of pDst[0] in xmm6 and other half in xmm7
					 *load d0
					 */ 
					
					// Copy the contents of pDst[0] to EAX register
					// Store the address of pDst[0] to EAX register (pDst is the background)
					mov eax, dword ptr[pDst + 0]; // GET THE RED VALUE OF THE BACKGROUND

					// Copy the 32-bit contents of [eax] into xmm1
					movdqu xmm1, [eax]; // XMM1 = pDst[0]

					// Copy the contents of xmm1 to xmm6
					movdqa xmm6, xmm1;
					
					// Unpack the lower 8 bytes of xmm1 into xmm6
					punpcklbw xmm6, xmm0; // XMM6 <- pDst[0] low 16bit //SPREAD LOWER BITS OF THAT RED VALUE ACROSS XMM6 (INTERSPERCED WITH 0s)
					
					// Unpack the higher 8 bytes of xmm1 into xmm7
					// Copy the contents of xmm1 over to xmm7
					movdqa xmm7, xmm1;
					
					// Unpack the higher
					punpckhbw xmm7, xmm0; // XMM7 <- pDst[0] high, 16 bit //SPREAD HIGHER BITS OF THAT RED VALUE ACROSS XMM7 (INTERSPERCED WITH 0s)

					// Copy the contents of ffconst into XMM4
					// ffconst acts as the "1" when multiplying the pDst with (1 - blendFac)  
					movdqu xmm4, [ffconst]; // XMM4 <- ff

					// Copy XMM4 (ffconst) to register XMM5
					movdqa xmm5, xmm4; 

					// Subtract XMM2 (Alpha channel) from XMM5 (ffconst) TO GET 1 - a0
					psubw  xmm5, xmm2; // XMM5 = ff - a0 = (1-a0)

					// Destination image[low order] * (1 - blendFactor[low order])
					pmullw xmm6, xmm5; // XMM6 = (ff - a0) * d0; 

					// Now for the upper bits
					movdqa xmm5, xmm4; // put 4 in 5
					psubw  xmm5, xmm3; // XMM5 = ff - a1 = (1-a1)

					// Destination image[high order] * (1 - blendFactor[high order])
					pmullw xmm7, xmm5; // XMM7 = (ff - a1) * d1; 

					// load the source;
					// Store the address of pSrc[0] to EAX register (pSrc is the bubble)
					mov eax, dword ptr[pSrc + 0];

					//Copy the contents of EAX into xmm1
					movdqu xmm1, [eax]; // XMM1 = pSrc[0]

					// Low bits of pSrc[0]
					// Copy the contents of xmm1 into xmm5
					movdqa xmm5, xmm1;

					// Unpack low order bits of pSrc[0]
					punpcklbw xmm5, xmm0; // XMM5 = pSrc[0], low, 16 bit;

					// Multiply low order bits from pSrc[0] and srcAlpha[0]
					pmullw xmm5, xmm2; // XMM5 = s0 * a0;

					// Add the above result with Alpha blend of destination 
					paddw xmm6, xmm5; // XMM6 = s0 * a0 + (ff - a0) * d0;

					// High bits of pSrc[0]
					// Repeat above steps but for higher order bits.
					movdqa xmm5, xmm1;
					punpckhbw xmm5, xmm0;
					pmullw xmm5, xmm3; // XMM5 = s1 * a1
					paddw xmm7, xmm5; // XMM7 = s1 * a1 + (ff - a1) * d1;

					// Shift the results;
					psrlw xmm6, 8;
					psrlw xmm7, 8;
					
					// Pack back
					// Pack both low and high order bits back
					packuswb xmm6, xmm7; // xmm6 <- xmm6{}xmm7 low bits;

					// EAX will hold pBlendImg, resulting blended image
					mov eax, dword ptr [pDst + 0];


					movdqu [eax], xmm6; // Done for this component;

					/* ======================= GREEN ======================= */
					/* The above steps are followed for Green as well */
					// blending the green;
					// load d0
					mov eax, dword ptr[pDst + 4]; 
					movdqu xmm1, [eax]; // xmm1 = pDst[1]
					movdqa xmm6, xmm1;
					punpcklbw xmm6, xmm0; // xmm6 <- pDst[1] low 16bit
					movdqa xmm7, xmm1;
					punpckhbw xmm7, xmm0; // xmm7 <- pDst[1] high, 16 bit
					// load the ff constant
					movdqu xmm4, [ffconst]; // xmm4 <- ff
					movdqa xmm5, xmm4; 
					psubw  xmm5, xmm2; // xmm5 = ff - a0
					pmullw xmm6, xmm5; // xmm6 = (ff - a0) * d0;
					// now for the upper bits
					movdqa xmm5, xmm4;
					psubw  xmm5, xmm3; // xmm5 = ff - a1
					pmullw xmm7, xmm5; // xmm7 = (ff - a1) * d1;
					// load the source;
					mov eax, dword ptr[pSrc + 4];
					movdqu xmm1, [eax]; // xmm1 = pSrc[1]
					// low bits of pSrc[1]
					movdqa xmm5, xmm1;
					punpcklbw xmm5, xmm0; // xmm5 = pSrc[1], low, 16 bit;
					pmullw xmm5, xmm2; // xmm5 = s0 * a0;
					paddw xmm6, xmm5; // xmm6 = s0 * a0 + (ff - a0) * d0;
					// high bits of pSrc[0]
					movdqa xmm5, xmm1;
					punpckhbw xmm5, xmm0;
					pmullw xmm5, xmm3; // xmm5 = s1 * a1
					paddw xmm7, xmm5; // xmm7 = s1 * a1 + (ff - a1) * d1;
					// shift the results;
					psrlw xmm6, 8;
					psrlw xmm7, 8;
					// pack back
					packuswb xmm6, xmm7; // xmm6 <- xmm6{}xmm7 low bits;
					mov eax, dword ptr [pDst + 4];
					movdqu [eax], xmm6; // done for this component;

					/* ======================= BLUE ======================= */
					/* The steps commented in Red are followed for Blue as well */
					// blending the blue;
					// load d0
					mov eax, dword ptr[pDst + 8]; 
					movdqu xmm1, [eax]; // xmm1 = pDst[0]
					movdqa xmm6, xmm1;
					punpcklbw xmm6, xmm0; // xmm6 <- pDst[0] low 16bit
					movdqa xmm7, xmm1;
					punpckhbw xmm7, xmm0; // xmm7 <- pDst[0] high, 16 bit
					// load the ff constant
					movdqu xmm4, [ffconst]; // xmm4 <- ff
					movdqa xmm5, xmm4; 
					psubw  xmm5, xmm2; // xmm5 = ff - a0
					pmullw xmm6, xmm5; // xmm6 = (ff - a0) * d0;
					// now for the upper bits
					movdqa xmm5, xmm4;
					psubw  xmm5, xmm3; // xmm5 = ff - a1
					pmullw xmm7, xmm5; // xmm7 = (ff - a1) * d1;
					// load the source;
					mov eax, dword ptr[pSrc + 8];
					movdqu xmm1, [eax]; // xmm1 = pSrc[2]
					// low bits of pSrc[0]
					movdqa xmm5, xmm1;
					punpcklbw xmm5, xmm0; // xmm5 = pSrc[2], low, 16 bit;
					pmullw xmm5, xmm2; // xmm5 = s0 * a0;
					paddw xmm6, xmm5; // xmm6 = s0 * a0 + (ff - a0) * d0;
					// high bits of pSrc[0]
					movdqa xmm5, xmm1;
					punpckhbw xmm5, xmm0;
					pmullw xmm5, xmm3; // xmm5 = s1 * a1
					paddw xmm7, xmm5; // xmm7 = s1 * a1 + (ff - a1) * d1;
					// shift the results;
					psrlw xmm6, 8;
					psrlw xmm7, 8;
					// pack back
					packuswb xmm6, xmm7; // xmm6 <- xmm6{}xmm7 low bits;
					mov eax, dword ptr [pDst + 8];
					movdqu [eax], xmm6; // done for this component;
				};
				pSrc[0] += 16;
				pSrc[1] += 16;
				pSrc[2] += 16;
				pSrc[3] += 16;

				pDst[0] += 16;
				pDst[1] += 16;
				pDst[2] += 16;
				pDst[3] += 16; // Modified for part 2, not that this really does anything since Dst has no alpha values
			}
		}
#pragma endregion
#pragma region SERIAL
		else if (simdMode == SIMD_NONE) {
			for (unsigned int x = X0; x < X1; x++) {
				short diff;
				short tmp;

				diff = *pSrc[0] - *pDst[0];
				tmp = short(*pDst[3] * diff) >> 8; // Part 2
				*pDst[0] = tmp + *pDst[0];

				diff = *pSrc[1] - *pDst[1];
				tmp = short(*pDst[3] * diff) >> 8; // Part 2
				*pDst[1] = tmp + *pDst[1];

				diff = *pSrc[2] - *pDst[2];
				tmp = short(*pDst[3] * diff) >> 8; // Part 2
				*pDst[2] = tmp + *pDst[2];

				pSrc[0] += 1;
				pSrc[1] += 1;
				pSrc[2] += 1;
				pSrc[3] += 1;

				pDst[0] += 1;
				pDst[1] += 1;
				pDst[2] += 1;
				pDst[3] += 1; // Modified for part 2 even though this does nothing
			}
		}
#pragma endregion
#pragma region EMMX_INTRINSICS
		else if (simdMode == SIMD_EMMX_INTRINSICS) { // CISC Architecture, a bit of overhead

			for (unsigned x = X0; x < X1; x += 16) {					// Loop through columns of 16px
				register __m128i s0, s1, d0, d1, a0, a1, r0, r1, zero;	// Assign some variables to 128 bit registers
				register __m128i diff0, tmp0, diff1, tmp1, t;			// Unused registers
				
				zero = _mm_setzero_si128(); // This acts as XMM0 and is set to all 0s, used for unpacking high-order and low-order bits

				// load alpha, will be used for blend factor
				t = _mm_loadu_si128((__m128i *) pDst[3]); // Modified for part 2

				// Unpack the low order alphas from the pSrc. Used for blend factor. 
				a0 = _mm_unpacklo_epi8(t, zero);

				// Unpack the high order alphas from pSrc. Used for blend factor.
				a1 = _mm_unpackhi_epi8(t, zero);
				
				/*  EMMX_BLEND does the calculations for Rs, Gs and Bs taking in their index as the argument. 
				*	It then sets the above defined registers with pSrc and pDst values, unpacking them into high and low order
				*	bits for further calculation. It then performs the blending calculation:
				*	pBlendImg[i][j] = pOrigImg[i][j] * blendFac + pKernelImg[i][j] * (1 - blendFac);
				*   The code was modified to make use of s0, s1, r0 and r1 __m128i registers.
				*   
				*/
				
			    EMMX_BLEND(0); 
				EMMX_BLEND(1);
				EMMX_BLEND(2);

				// Similar to MMX, it performs this on blocks of 16 pxls
				pSrc[0] += 16;
				pSrc[1] += 16;
				pSrc[2] += 16;
				pSrc[3] += 16;

				pDst[0] += 16;
				pDst[1] += 16;
				pDst[2] += 16;
				pDst[3] += 16; // Modified for part 2 even though this does nothing
			}
		}
#pragma endregion
	}
}
