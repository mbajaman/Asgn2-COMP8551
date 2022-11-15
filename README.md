## Assignment 2 for COMP 8551
Using assembly for image manipulation

## Setup
After cloning the repo make sure you set the VS2022 configuration to `Debug` and `x86`.

In addition to this, make sure do the following:

```
Click on FishTank (project) > Properties > Debugging
```

Set command arguments to `..\data\reef_small.bmp ..\data\small_pink_bubble.bmp`


## QUESTIONS

1. The SIMD_EMMX mode, written almost entirely in assembly, is fastest. This mode Blends images 16 pixels at a time, giving it an advantage over the serial mode. It does this blending by iterating over the source and destination images, and unpacking their color values for R, G, and B into separate registers. Then, it does the blending formula in the document, using the alpha channel of the source image as the blending factor to simulate transparency. To finally bring it all together, it packs the color values from the source and destination images back into each other, and since they were interleaved with 0s, their color data for the 16 pixels each loop can be combined and drawn onto the destination image.

The SIMD_NONE serial mode is the slowest. It also follows the same formula as the other modes, but uses regular C++, and importantly, only iterates across the source image one pixel at a time. As such, it is noticeably slower than the other two modes.

The SIMD_EMMX_INTRINSICS serial mode is the second fastest. It works in a very similar way to the fastest EMMX mode, also iterating by 16 pixels, except it is written in C++ and uses some 'expensive' function calls that have unneeded overhead, at least when compared to the EMMX mode. It also defines several variables and their register spaces that aren’t needed or used.  Because of this, it is slightly slower than the mode written in pure assembly that performs only the needed operations.

2. This appears to cause the bubbles to lose their blending, which makes sense. Using the alpha channel of the destination image to blend the source image doesn't really work because the destination image is supposed to be opaque, (at least with the tank example) so it's not going to have alpha values that make the code do much blending.

Code for parts 3 and 4 can be found beyone line 100 of main.cpp.
3. In the first function call, the assembly casts the contents of x as a dword ptr into register EAX, but with the pass by reference, it can use lea to simply pass the existing address of the variable into EAX.


 