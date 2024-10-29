//#define IN_SHADERTOY

#ifndef IN_SHADERTOY

#include "Common.glsl"

#iChannel0 "BufferB.glsl"

#endif

void mainImage(out vec4 fragColor, in vec2 fragCoord) {
    fragColor = vec4(texelFetch(iChannel0, ivec2(fragCoord), 0).rgb, 1.0);
}
