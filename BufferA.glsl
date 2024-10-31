//#define IN_SHADERTOY

#ifdef IN_SHADERTOY

//bool isKeyDown(k) {
//    return (texelFetch(iChannel1, ivec2(k, 0), 0).x > 0.0);
//}

const int KeyLeft = 65;  //'A'
const int KeyRight = 68;
const int KeyUp = 87;
const int KeyDown = 83;
const int KeyE = 69;
const int KeyQ = 81;
const int KeyT = 84;

#else

#include "Common.glsl"

#iChannel0 "self"
#iChannel0::MinFilter "Linear"
#iChannel0::MagFilter "Linear"
#iChannel0::WrapMode "Clamp"

#iKeyboard  // exposes isKeyPressed, isKeyReleased, isKeyDown, isKeyToggled functions

const int KeyLeft = Key_A;  //'A'
const int KeyRight = Key_D;
const int KeyUp = Key_W;
const int KeyDown = Key_S;
const int KeyE = Key_E;
const int KeyQ = Key_Q;
const int KeyT = Key_T;


#endif




// Handles camera only


vec4 UpdateCamera(vec2 fragCoord) {
    /*  
    We will use the first 2 pixels of the buffer to store the information we need.
    Every pixel contains 4 channels (floats), for RGBA. We can exploit this in the following way:
        pixel0 = (cameraX, cameraY, cameraZ, U)
        pixel1 = (wX, wY, wZ, V)
    where
        cameraX, cameraY and cameraZ describe the position of the camera respectively
        U,V give the current rotation of the camera in spherical coordinates
        wX, wY, wZ is the forward vector
    */

    vec4 data1 = texelFetch(iChannel0, ivec2(0, 0), 0);
    vec4 data2 = texelFetch(iChannel0, ivec2(1, 0), 0);
    vec3 eye = data1.xyz + EyeStartPosition;
    vec2 uv = abs(vec2(data1.w, data2.w));

    if (iMouse.z > 0. || data1.w >= 0.) {  // mouse held or was held last frame
        float du = (abs(iMouse.z) - abs(iMouse.x)) * 0.01;
        float dv = (abs(iMouse.y) - abs(iMouse.w)) * 0.01;
        if (data1.w >= 0.0) {
            uv = vec2((uv.x + du), clamp((uv.y + dv), 1.571, 4.712)); // 0.5 pi and 1.5 pi
        } else {
            uv = vec2((uv.x + du), clamp((uv.y + dv), -4.713, -1.570)); // -1.5 pi and -0.5 pi
        }
    }

    vec3 w = vec3(cos(uv.x) * cos(-uv.y), sin(-uv.y), sin(uv.x) * cos(-uv.y));
    vec3 u = normalize(cross(vec3(0, 1, 0), w));

    // Keyboard and mouse handling:

    float speed = 0.2;
    if (isKeyDown(KeyLeft)) {
        eye -= u * speed;
    }
    if (isKeyDown(KeyRight)) {
        eye += u * speed;
    }
    if (isKeyDown(KeyUp)) {
        eye += w * speed;
    }
    if (isKeyDown(KeyDown)) {
        eye -= w * speed;
    }
    if (isKeyDown(KeyE)) {
        eye.y += speed;
    }
    if (isKeyDown(KeyQ)) {
        eye.y -= speed;
    }

    vec4 data3 = texelFetch(iChannel0, ivec2(0, 1), 0);
    vec4 data4 = texelFetch(iChannel0, ivec2(1, 1), 0);
    vec3 ray_p = data3.xyz;
    vec3 ray_v = data4.xyz;
    if (isKeyDown(KeyT)) {
        ray_p = eye;
        ray_v = normalize(w);
    }

    vec2 outdata = vec2(data1.w, data2.w);
    if (iMouse.z >= 0.0) {
        outdata = abs(vec2(outdata));  // mouse held
    } else if (data1.w >= 0.0) {
        outdata = -mod(uv, 2.0 * pi);  // mouse released
    }

    vec4 fragColor;
    if (fragCoord.x == 0.5 && fragCoord.y == 0.5) { // pixel (0,0)
        fragColor = vec4(eye - EyeStartPosition, outdata.x);
    }
    if (fragCoord.x == 1.5 && fragCoord.y == 0.5) { // pixel (1,0)
        fragColor = vec4(w, outdata.y);
    }
    if (fragCoord.x == 0.5 && fragCoord.y == 1.5) { // pixel (0,1)
        fragColor = vec4(ray_p, 0.0);
    }
    if (fragCoord.x == 1.5 && fragCoord.y == 1.5) { // pixel (1,1)
        fragColor = vec4(ray_v, 0.0);
    }

    return fragColor;
}

void mainImage(out vec4 fragColor, in vec2 fragCoord) {
    //Generate ray from pixel
    if (fragCoord.x > 1.5 || fragCoord.y > 1.5) { 
        discard;
    }

    fragColor = UpdateCamera(fragCoord);
}
