//#define IN_SHADERTOY

#ifndef IN_SHADERTOY

#include "Common.glsl"

#iChannel0 "BufferA.glsl"

#iChannel1 "self"
#iChannel1::MinFilter "Linear"
#iChannel1::MagFilter "Linear"
#iChannel1::WrapMode "Clamp"

#iChannel2 "../skybox_{}.png"
#iChannel2::Type "CubeMap"

#endif

const Material materials[6] = Material[](
    Material(vec3(0.2, 0.0, 0.0), 0.2, 0.2, 0.02),     // Plastic
    Material(vec3(0.95, 0.93, 0.88), 0.1, 0.0, 0.95),  // Silver
    Material(vec3(1.0, 0.71, 0.29), 0.05, 0.0, 0.9),    // Polished Gold
    Material(vec3(0.8, 0.85, 0.88), 0.5, 0.0, 0.7),     // Brushed Aluminum
    Material(vec3(0.95, 0.64, 0.54), 0.3, 0.0, 0.75),  // Copper
    Material(vec3(0.56, 0.57, 0.58), 1.0, 0.0, 0.75)   // Rough Iron
);

const Light lights[3] = Light[](
    Light(vec3(10.0, 10.0, 0.0), vec3(1.0, 0.0, 0.0)), 
    Light(vec3(0.0, 10.0, 0.0), vec3(0.0, 1.0, 0.0)), 
    Light(vec3(0.0, 10.0, 10.0), vec3(0.0, 0.0, 1.0)));

// stolen from: https://learnopengl.com/PBR/Lighting
float DistributionGGX(vec3 N, vec3 H, float roughness) {
    float a = roughness * roughness;
    float a2 = a * a;
    float NdotH = max(dot(N, H), 0.0);
    float NdotH2 = NdotH * NdotH;

    float num = a2;
    float denom = (NdotH2 * (a2 - 1.0) + 1.0);
    denom = pi * denom * denom;

    return num / denom;
}

float GeometrySchlickGGX(float NdotV, float roughness) {
    float r = (roughness + 1.0);
    float k = (r * r) / 8.0;

    float num = NdotV;
    float denom = NdotV * (1.0 - k) + k;

    return num / denom;
}

float GeometrySmith(vec3 N, vec3 V, vec3 L, float roughness) {
    float NdotV = max(dot(N, V), 0.0);
    float NdotL = max(dot(N, L), 0.0);
    float ggx2 = GeometrySchlickGGX(NdotV, roughness);
    float ggx1 = GeometrySchlickGGX(NdotL, roughness);

    return ggx1 * ggx2;
}

vec3 fresnelSchlick(float cosTheta, vec3 F0) {
    return F0 + (1.0 - F0) * pow(clamp(1.0 - cosTheta, 0.0, 1.0), 5.0);
}

vec3 BRDF(vec3 camPos, vec3 WorldPos, vec3 Normal, int mat_id, inout vec3 refl) {
    vec3 albedo = materials[mat_id].color;
    float roughness = materials[mat_id].roughness;
    float emission_strength = materials[mat_id].emission_strength;
    float metallic = materials[mat_id].metalness;


    const float ao = 0.1;
    const float reflection_attenuation = 1.0;

    vec3 N = normalize(Normal);
    vec3 V = normalize(camPos - WorldPos);

    vec3 F0 = vec3(0.04);
    F0 = mix(F0, albedo, metallic);

    // reflectance equation
    vec3 Lo = vec3(0.0);
    for (int i = 0; i < 3; ++i) {
        vec3 lightPosition = lights[i].position;
        vec3 lightColor = lights[i].color;
        // calculate per-light radiance
        vec3 L = normalize(lightPosition - WorldPos);
        vec3 H = normalize(V + L);
        float dist = length(lightPosition - WorldPos);
        float attenuation = 1.0 / (dist * dist);
        vec3 radiance = lightColor * attenuation;

        // cook-torrance brdf
        float NDF = DistributionGGX(N, H, roughness);
        float G = GeometrySmith(N, V, L, roughness);
        vec3 F = fresnelSchlick(max(dot(H, V), 0.0), F0);

        vec3 kS = F;
        vec3 kD = vec3(1.0) - kS;
        kD *= (1.0 - metallic);

        vec3 numerator = NDF * G * F;
        float denominator = 4.0 * max(dot(N, V), 0.0) * max(dot(N, L), 0.0) + 0.0001;
        vec3 specular = numerator / denominator;

        // add to outgoing radiance Lo
        float NdotL = max(dot(N, L), 0.0);
        Lo += (kD * albedo / pi + specular) * radiance * NdotL;
    }

    vec3 ambient = vec3(0.03) * albedo * ao;
    vec3 color = ambient + Lo + albedo * emission_strength * max(dot(V, N), 0.0);

    color = color / (color + vec3(1.0));
    color = pow(color, vec3(1.0 / 2.2));

    color *= refl;
    refl *= fresnelSchlick(max(dot(V, N), 0.0), F0) * reflection_attenuation;

    return color;
}

// Buffer A handels camera movement. Here we only read it
Ray ReadCamera(vec2 fragCoord, out float diagInv) {
    vec4 data1 = texelFetch(iChannel0, ivec2(0, 0), 0);
    vec4 data2 = texelFetch(iChannel0, ivec2(1, 0), 0);
    vec3 eye = data1.xyz + EyeStartPosition;

    vec3 w = data2.xyz;
    vec3 u = normalize(cross(vec3(0, 1, 0), w));
    vec3 v = cross(w, u);

    // vec2 px = (fragCoord/iResolution.xy*2.-1.)*1.*normalize(iResolution.xy);
    diagInv = 1.0 / length(iResolution.xy);
    vec2 px = (2.0 * fragCoord - iResolution.xy) * diagInv * 1.0;

    return Ray(eye, 0.0, normalize(w + px.x * u + px.y * v), 1000.0);
}
// SDF Primitives (more on https://iquilezles.org/articles/distfunctions/)

float sdRoundBox(vec3 p, vec3 b, float r) {
    vec3 q = abs(p) - b + r;
    return length(max(q, 0.0)) + min(max(q.x, max(q.y, q.z)), 0.0) - r;
}

float sdBoxFrame(vec3 p, vec3 b, float e) {
    p = abs(p) - b;
    vec3 q = abs(p + e) - e;
    return min(min(length(max(vec3(p.x, q.y, q.z), 0.0)) + min(max(p.x, max(q.y, q.z)), 0.0), length(max(vec3(q.x, p.y, q.z), 0.0)) + min(max(q.x, max(p.y, q.z)), 0.0)), length(max(vec3(q.x, q.y, p.z), 0.0)) + min(max(q.x, max(q.y, p.z)), 0.0));
}

float sdCapsule(vec3 p, vec3 a, vec3 b, float r) {
    vec3 pa = p - a, ba = b - a;
    float h = clamp(dot(pa, ba) / dot(ba, ba), 0.0, 1.0);
    return length(pa - ba * h) - r;
}

float sdSphere(vec3 p, float s) {
    return length(p) - s;
}

float sdBox(vec3 p, vec3 b) {
    vec3 d = abs(p) - b;
    return length(max(d, 0.0)) + min(max3(d.x, d.y, d.z), 0.0);
}

// SDF Operations

vec3 opRepLim(vec3 p, float c, in vec3 l) {
    return p - c * clamp(round(p / c), -l, l);
}
vec3 opRep(vec3 p, float s) {
    return mod(p + s * 0.5, s) - s * 0.5;
}

vec3 opSymX(vec3 p) {
    return vec3(abs(p.x), p.y, p.z);
}

vec3 opRepZ(vec3 p, float sz, float lo, float up) {
    return vec3(p.x, p.y, p.z - sz * clamp(round(p.z / sz), lo, up));
}

vec3 opRepXZ(vec3 p, float sx, float sz) {
    return vec3((mod(p.x + sx * 0.5, sx) - sx * 0.5), p.y, (mod(p.z + sz * 0.5, sz) - sz * 0.5));
}

float intersectPlane(Ray ray, vec3 q, vec3 n) {
    return dot(q - ray.P, n) / dot(ray.V, n);
}

float pendulum(float t) {
    // const float q = 0.0;                   // initial angle: 0
    // const float q = 0.0004765699168668419; // initial angle: 10
    // const float q = 0.0019135945901703004; // initial angle: 20
    // const float q = 0.004333420509983138;  // initial angle: 30
    // const float q = 0.007774680416441802;  // initial angle: 40
    // const float q = 0.01229456052718145;   // initial angle: 50
    const float q = 0.017972387008967222;  // initial angle: 60
    // const float q = 0.02491506252398093;   // initial angle: 70
    // const float q = 0.03326525669557733;   // initial angle: 80
    // const float q = 0.04321391826377224;   // initial angle: 90

    float theta = 8.0 * (pow(q, 0.5) / (1.0 + pow(q, 1.0))) * cos(1.0 * t) 
                + -2.6666666666666665 * (pow(q, 1.5) / (1.0 + pow(q, 3.0))) * cos(3.0 * t) 
                + 1.6 * (pow(q, 2.5) / (1.0 + pow(q, 5.0))) * cos(5.0 * t);
    //            + -1.1428571428571428 * (pow(q, 3.5) / (1.0 + pow(q, 7.0))) * cos(7.0 * t)
    //            + 0.8888888888888888  * (pow(q, 4.5) / (1.0 + pow(q, 9.0))) * cos(9.0 * t)
    //            + -0.7272727272727273 * (pow(q, 5.5) / (1.0 + pow(q, 11.0))) * cos(11.0 * t);

    return theta;
}

// Signed Distance Function
//float sdf(vec3 p) { return 0.0; }
Value sdf(vec3 p) {
    const float freq = 2.0;

    float first_sphere_angle = pendulum(clamp(mod(freq * iTime, 2.0 * pi), 0.5 * pi, 1.5 * pi) + pi);
    float last_sphere_angle = pendulum(clamp(mod(freq * iTime + pi, 2.0 * pi), 0.5 * pi, 1.5 * pi));

    vec3 first_sphere_pos =    vec3(0.0, -13.0 + 7.0 * cos(first_sphere_angle), -4.0 - 7.0 * sin(first_sphere_angle));
    vec3 last_sphere_pos =     vec3(0.0, -13.0 + 7.0 * cos(last_sphere_angle),   4.0 - 7.0 * sin(last_sphere_angle));
    vec3 first_cable_end_pos = vec3(0.0, -13.0 + 6.0 * cos(first_sphere_angle), -4.0 - 6.0 * sin(first_sphere_angle));
    vec3 last_cable_end_pos =  vec3(0.0, -13.0 + 6.0 * cos(last_sphere_angle),   4.0 - 6.0 * sin(last_sphere_angle));

    p = vec3(p.x, p.y + 10.0, p.z);

    float d = 10000.0;
    Value v = Value(10000.0, 0);
    v = Unite(v, Value(sdBoxFrame(p + vec3(0.0, -8.0, 0.0), vec3(3.0, 5.0, 6.0), 0.1), 1));
    v = Unite(v, Value(max(sdRoundBox(p + vec3(0.0, -3.0, 0.0), vec3(4.0, 1.0, 7.0), 0.5), -(p.y - 3.0)), 0));
    v = Unite(v, Value(sdSphere(p + first_sphere_pos, 1.0), 2));
    v = Unite(v, Value(sdSphere(p + last_sphere_pos, 1.0), 2));
    v = Unite(v, Value(sdCapsule(vec3(0.0, 0.0, 0.0), p + first_cable_end_pos, p + vec3(3.0 - 0.1, -13.0 + 0.1, -4.0), 0.05), 4));
    v = Unite(v, Value(sdCapsule(vec3(0.0, 0.0, 0.0), p + first_cable_end_pos, p + vec3(-3.0 + 0.1, -13.0 + 0.1, -4.0), 0.05), 4));
    v = Unite(v, Value(sdCapsule(vec3(0.0, 0.0, 0.0), p + last_cable_end_pos, p + vec3(3.0 - 0.1, -13.0 + 0.1, 4.0), 0.05), 4));
    v = Unite(v, Value(sdCapsule(vec3(0.0, 0.0, 0.0), p + last_cable_end_pos, p + vec3(-3.0 + 0.1, -13.0 + 0.1, 4.0), 0.05), 4));
    vec3 p_rep_z = opRepZ(p, 2.0, -1.0, 1.0);
    vec3 p_rep_sym_x = opSymX(p_rep_z);
    v = Unite(v, Value(sdSphere(p_rep_z + vec3(0.0, -6.0, 0.0), 1.0), 2));
    v = Unite(v, Value(sdCapsule(vec3(0.0, 0.0, 0.0), p_rep_z + vec3(0.0, -7.0, 0.0), p_rep_z + vec3(3.0 - 0.1, -13.0 + 0.1, 0.0), 0.05), 4));
    v = Unite(v, Value(sdCapsule(vec3(0.0, 0.0, 0.0), p_rep_z + vec3(0.0, -7.0, 0.0), p_rep_z + vec3(-3.0 + 0.1, -13.0 + 0.1, 0.0), 0.05), 4));

    return v;
}

// symmetric differential
vec3 normal(vec3 p) {
    const vec2 eps0 = vec2(0.01, 0);
    vec3 m0 = vec3(sdf(p - eps0.xyy).d, sdf(p - eps0.yxy).d, sdf(p - eps0.yyx).d);
    vec3 m1 = vec3(sdf(p + eps0.xyy).d, sdf(p + eps0.yxy).d, sdf(p + eps0.yyx).d);
    return normalize(m1 - m0);
}


TraceResult sphere_trace(Ray ray, SphereTraceDesc params) {
    TraceResult ret = TraceResult(ray.Tmin, 0, 0);
    float d;
    do {
        //d = abs(sdf(ray.P + ret.T * ray.V));
        ret.T += d;
        ret.steps++;
    } while (ret.T < ray.Tmax &&            // Stay within bound box
             d > params.epsilon * ret.T &&  // Stop if cone is close to surface
             ret.steps < params.maxiters    // Stop if too many iterations
    );
    ret.flags = int(ret.T >= ray.Tmax) | (int(d <= params.epsilon * ret.T) << 1) | (int(ret.steps >= params.maxiters) << 2);
    return ret;
}


void sphere_trace(Ray ray, SphereTraceDesc params, inout TraceResult tr) {
    float d;
    Value v;
    do {
        //d = abs(sdf(ray.P + tr.T * ray.V));
        v = sdf(ray.P + tr.T * ray.V);
        d = abs(v.d);
        tr.T += d;
        tr.steps++;
    } while (tr.T < ray.Tmax &&            // Stay within bound box
             d > params.epsilon * tr.T &&  // Stop if cone is close to surface
             tr.steps < params.maxiters    // Stop if too many iterations
    );
    tr.flags = int(tr.T >= ray.Tmax) | (int(d <= params.epsilon * tr.T) << 1) | (int(tr.steps >= params.maxiters) << 2) | (int(v.id << 3));
}

vec3 Render(Ray ray, SphereTraceDesc params) {
    vec3 col = vec3(0.0);
    vec3 refl = vec3(1.0);

    TraceResult tr = TraceResult(ray.Tmin, 0, 0);
    for (int i = 0; i < 7; i++) {
        sphere_trace(ray, params, tr);

        if (bool(tr.flags & 1)) {
            col += refl * texture(iChannel2, ray.V).rgb;
            break;
        } else if (bool(tr.flags & 2)) {
            int mat_id = (tr.flags >> 3) & 0xFF;
            vec3 pos = ray.P + ray.V * tr.T;
            vec3 norm = normal(pos);
            col += BRDF(ray.P, pos, norm, mat_id, refl);

            ray.P = pos + 0.0001 * norm; // small offset so next sdf doesn't return 0
            ray.V = reflect(ray.V, norm);

        } else if (bool(tr.flags & 4)) {
            //col = vec3(1.0, 0.0, 0.0);
            break;
        } else {
            // this shouldn't happen
            //col = vec3(0.0, 0.0, 1.0);
            break;
        }

        tr.T = 0.0;
        tr.flags = 0;
    }



    return col;
}

void mainImage(out vec4 fragColor, in vec2 fragCoord) {
    float epsilon;
    Ray ray = ReadCamera(fragCoord, epsilon);

    SphereTraceDesc params = SphereTraceDesc(epsilon, 1064);
    // TraceResult ret = sphere_trace(ray, params);

    vec3 color = Render(ray, params);
    fragColor = vec4(Render(ray, params), 1.0);

}
