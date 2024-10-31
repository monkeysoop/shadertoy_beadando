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

#define NUM_OF_REFLECTIONS 5

#define IMAGE_MAX_ITERS 150
#define SHADOW_MAX_ITERS 40


#define NUM_OF_MATERIALS 6
#define NUM_OF_LIGHTS 3


const Material materials[NUM_OF_MATERIALS] = Material[](
    Material(vec3(10.2, 0.0, 0.0), 0.2, 0.0, 0.02),     // Plastic
    Material(vec3(0.95, 0.93, 0.88), 0.1, 0.0, 0.95),  // Silver
    Material(vec3(1.0, 0.71, 0.29), 0.05, 0.0, 0.9),    // Polished Gold
    Material(vec3(0.8, 0.85, 0.88), 0.5, 0.0, 0.7),     // Brushed Aluminum
    Material(vec3(0.95, 0.64, 0.54), 0.3, 0.0, 0.75),  // Copper
    Material(vec3(0.56, 0.57, 0.58), 1.0, 0.0, 0.75)   // Rough Iron
);

const Light lights[NUM_OF_LIGHTS] = Light[](
    Light(vec3(10.0, 10.0, 0.0), vec3(10.0, 0.0, 0.0)), 
    Light(vec3(0.0, 10.0, 0.0), vec3(0.0, 10.0, 0.0)), 
    Light(vec3(0.0, 10.0, 10.0), vec3(0.0, 0.0, 1.0))
);





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

    return Ray(eye, 0.0, normalize(w + px.x * u + px.y * v), MAX_DIST);
}

Ray ReadVisRay() {
    vec4 data1 = texelFetch(iChannel0, ivec2(0, 1), 0);
    vec4 data2 = texelFetch(iChannel0, ivec2(1, 1), 0);
    Ray ray = Ray(data1.xyz, 0.0, data2.xyz, MAX_DIST);
    return ray;
}


//float RaySphereIntersect(Ray ray, vec3 o, float r) {
//    // rax.V should be normalized
//    float b = dot(ray.V, (ray.P - o));
//    float c = dot((ray.P - o), (ray.P - o)) - r * r;
//    float d = b * b - c;
//
//    float t1 = -b - sqrt(max(d, 0.0));
//
//    return (d < 0.0 || t1 < 0.0) ? MAX_DIST : t1; 
//}
//
//Value sdfSpheres(Ray ray, vec4 vis_spheres[VIS_RAY_MAX_ITERS]) {
//    Value v = Value(MAX_DIST, 0);
//    for (int i = 0; i < VIS_RAY_MAX_ITERS; i++) {
//        vec3 pos = vis_spheres[i].xyz;
//        float r = vis_spheres[i].w;
//        v = Unite(v, Value(RaySphereIntersect(ray, pos, r), (i % NUM_OF_MATERIALS)));
//    }
//    return v;
//}
//
//float RaySphereMinDist(Ray ray, vec3 o , float r)  {
//    return max(0.0, length(o - ray.P + dot((o - ray.P), ray.V) * ray.V) - r);
//}
//
//float SphereMinDist(Ray ray, vec4 vis_spheres[VIS_RAY_MAX_ITERS], float penumbra) {
//    float min_d = 1.0;
//    for (int i = 0; i < VIS_RAY_MAX_ITERS; i++) {
//        vec3 pos = vis_spheres[i].xyz;
//        float r = vis_spheres[i].w;
//        float d = RaySphereMinDist(ray, pos , r);
//        min_d = min(min_d, (d * penumbra) / (dot((pos - ray.P), ray.V)))
//    }
//    return min_d;
//}

// Signed Distance Function
Value sdf(vec3 p, vec4 vis_spheres[VIS_RAY_MAX_ITERS]) {
    Value v = SceneSDF(p);
    for (int i = 0; i < VIS_RAY_MAX_ITERS; i++) {
        vec3 pos = vis_spheres[i].xyz;
        float r = vis_spheres[i].w;
        v = Unite(v, Value(sdSphere(p - pos, r), 2));
    }
    return v;
}

// symmetric differential
vec3 normal(vec3 p, vec4 vis_spheres[VIS_RAY_MAX_ITERS]) {
    const vec2 eps0 = vec2(0.01, 0);
    vec3 m0 = vec3(sdf(p - eps0.xyy, vis_spheres).d, sdf(p - eps0.yxy, vis_spheres).d, sdf(p - eps0.yyx, vis_spheres).d);
    vec3 m1 = vec3(sdf(p + eps0.xyy, vis_spheres).d, sdf(p + eps0.yxy, vis_spheres).d, sdf(p + eps0.yyx, vis_spheres).d);
    return normalize(m1 - m0);
}



void relaxed_sphere_trace_vis_ray(Ray ray, SphereTraceDesc params, inout vec4 vis_spheres[VIS_RAY_MAX_ITERS]) {
    TraceResult tr = TraceResult(ray.Tmin, 0, 0);
    const float w = 1.6;

    float d;
    Value v;

    float next_d;
    Value next_v;
    float next_t;

    v = SceneSDF(ray.P + tr.T * ray.V);
    d = abs(v.d);
    while (tr.T < ray.Tmax &&        // Stay within bound box
           d > params.epsilon * tr.T &&  // Stop if cone is close to surface
           tr.steps < params.maxiters    // Stop if too many iterations)
    ) {
        next_t = tr.T + w * d;
        next_v = SceneSDF(ray.P + next_t * ray.V);
        next_d = abs(next_v.d);
        if ((d + next_d) < w * d) {
            next_t = tr.T + d; 
            next_v = SceneSDF(ray.P + next_t * ray.V);
            next_d = abs(next_v.d);       
        }
        vis_spheres[tr.steps] = vec4((ray.P + tr.T * ray.V), d);
        tr.T = next_t;
        v = next_v;
        d = next_d;
        tr.steps++;
    }    
}

void relaxed_sphere_trace(Ray ray, SphereTraceDesc params, vec4 vis_spheres[VIS_RAY_MAX_ITERS], inout TraceResult tr) {
    const float w = 1.6;

    float d;
    Value v;

    float next_d;
    Value next_v;
    float next_t;

    v = sdf(ray.P + tr.T * ray.V, vis_spheres);
    d = abs(v.d);
    while (tr.T < ray.Tmax &&            // Stay within bound box
           d > params.epsilon * tr.T &&  // Stop if cone is close to surface
           tr.steps < params.maxiters    // Stop if too many iterations)
    ) {
        next_t = tr.T + w * d;
        next_v = sdf(ray.P + next_t * ray.V, vis_spheres);
        next_d = abs(next_v.d);
        if ((d + next_d) < w * d) {
            next_t = tr.T + d; 
            next_v = sdf(ray.P + next_t * ray.V, vis_spheres);
            next_d = abs(next_v.d);       
        }
        tr.T = next_t;
        v = next_v;
        d = next_d;
        tr.steps++;
    }    
    tr.flags = int(tr.T >= ray.Tmax) | (int(d <= params.epsilon * tr.T) << 1) | (int(tr.steps >= params.maxiters) << 2) | (int(v.id << 3));
}

float relaxed_sphere_trace_shadow(Ray ray, SphereTraceDesc params, float penumbra, vec4 vis_spheres[VIS_RAY_MAX_ITERS]) {
    TraceResult tr = TraceResult(ray.Tmin, 0, 0);
    const float w = 1.6;

    float min_d = 1.0;
    float d;
    Value v;

    float next_d;
    Value next_v;
    float next_t;

    v = sdf(ray.P + tr.T * ray.V, vis_spheres);
    d = abs(v.d);
    while (tr.T < ray.Tmax &&        // Stay within bound box
           d > params.epsilon * tr.T &&  // Stop if cone is close to surface
           tr.steps < params.maxiters    // Stop if too many iterations)
    ) {
        next_t = tr.T + w * d;
        next_v = sdf(ray.P + next_t * ray.V, vis_spheres);
        next_d = abs(next_v.d);
        if ((d + next_d) < w * d) {
            next_t = tr.T + d; 
            next_v = sdf(ray.P + next_t * ray.V, vis_spheres);
            next_d = abs(next_v.d);       
        }
        min_d = min(min_d, ((d * penumbra) / tr.T));
        tr.T = next_t;
        v = next_v;
        d = next_d;
        tr.steps++;
    }    
    return min_d;
}

void sphere_trace_vis_ray(Ray ray, SphereTraceDesc params, inout vec4 vis_spheres[VIS_RAY_MAX_ITERS]) {
    TraceResult tr = TraceResult(ray.Tmin, 0, 0);
    float d;
    Value v;
    do {
        v = SceneSDF(ray.P + tr.T * ray.V);
        d = abs(v.d);
        vis_spheres[tr.steps] = vec4((ray.P + tr.T * ray.V), d);
        tr.T += d;
        tr.steps++;
    } while (tr.T < ray.Tmax &&         // Stay within bound box
             d > params.epsilon * tr.T &&   // Stop if cone is close to surface
             tr.steps < params.maxiters     // Stop if too many iterations
    );
}

void sphere_trace(Ray ray, SphereTraceDesc params, vec4 vis_spheres[VIS_RAY_MAX_ITERS], inout TraceResult tr) {
    float d;
    Value v;
    do {
        v = sdf(ray.P + tr.T * ray.V, vis_spheres);
        d = abs(v.d);
        tr.T += d;
        tr.steps++;
    } while (tr.T < ray.Tmax &&            // Stay within bound box
             d > params.epsilon * tr.T &&  // Stop if cone is close to surface
             tr.steps < params.maxiters    // Stop if too many iterations
    );
    tr.flags = int(tr.T >= ray.Tmax) | (int(d <= params.epsilon * tr.T) << 1) | (int(tr.steps >= params.maxiters) << 2) | (int(v.id << 3));
}

float sphere_trace_shadow(Ray ray, SphereTraceDesc params, float penumbra, vec4 vis_spheres[VIS_RAY_MAX_ITERS]) {
    TraceResult tr = TraceResult(ray.Tmin, 0, 0);
    float min_d = 1.0;
    float d;
    Value v;
    do {
        v = sdf(ray.P + tr.T * ray.V, vis_spheres);
        d = abs(v.d);
        min_d = min(min_d, ((d * penumbra) / tr.T));
        tr.T += d;
        tr.steps++;
    } while (tr.T < ray.Tmax &&            // Stay within bound box
             d > params.epsilon * tr.T &&  // Stop if cone is close to surface
             tr.steps < params.maxiters    // Stop if too many iterations
    );
    return min_d;
}

float SoftShadow(vec3 pos, vec3 light_pos, vec4 vis_spheres[VIS_RAY_MAX_ITERS]) {
    const float penumbra = 10.0;
    const float intensity = 1.0;
    
    SphereTraceDesc params = SphereTraceDesc(0.01, SHADOW_MAX_ITERS);
    Ray ray = Ray(pos, 0.0, normalize(light_pos - pos), min(MAX_DIST, length(light_pos - pos)));

    float min_d = relaxed_sphere_trace_shadow(ray, params, penumbra, vis_spheres);

    
    return pow(min_d, intensity);
}

// stolen from: https://learnopengl.com/PBR/Lighting
vec3 BRDF(vec3 camPos, vec3 WorldPos, vec3 Normal, int mat_id, bool with_shadow, vec4 vis_spheres[VIS_RAY_MAX_ITERS], inout vec3 refl) {
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
    for (int i = 0; i < NUM_OF_LIGHTS; ++i) {
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

        if (with_shadow) {
            float shadow = SoftShadow((WorldPos + Normal * 0.0001), lightPosition, vis_spheres);
            Lo += (kD * albedo / pi + specular) * radiance * NdotL * shadow;
        } else {
            Lo += (kD * albedo / pi + specular) * radiance * NdotL;
        }

    }

    vec3 ambient = vec3(0.03) * albedo * ao;
    vec3 color = ambient + Lo + albedo * emission_strength * max(dot(V, N), 0.0);

    color = color / (color + vec3(1.0));
    color = pow(color, vec3(1.0 / 2.2));

    color *= refl;
    refl *= fresnelSchlick(max(dot(V, N), 0.0), F0) * reflection_attenuation;

    return color;
}



vec3 Render(Ray ray, Ray vis_ray, SphereTraceDesc params) {
    vec3 col = vec3(0.0);
    vec3 refl = vec3(1.0);

    vec4 vis_spheres[VIS_RAY_MAX_ITERS];
    for (int i = 0; i < VIS_RAY_MAX_ITERS; i++) {
        vis_spheres[i] = vec4(0.0);
    }
    SphereTraceDesc vis_params = SphereTraceDesc(0.001, VIS_RAY_MAX_ITERS);
    relaxed_sphere_trace_vis_ray(vis_ray, vis_params, vis_spheres);
 

    TraceResult tr = TraceResult(ray.Tmin, 0, 0);
    for (int i = 0; i < (NUM_OF_REFLECTIONS + 1); i++) {
        relaxed_sphere_trace(ray, params, vis_spheres, tr);
        

        if (bool(tr.flags & 1)) {
            col += refl * texture(iChannel2, ray.V).rgb;
            break;
        } else if (bool(tr.flags & 2)) {
            int mat_id = (tr.flags >> 3) & 0xFF;
            vec3 pos = ray.P + ray.V * tr.T;
            vec3 norm = normal(pos, vis_spheres);
            col += BRDF(ray.P, pos, norm, mat_id, (i == 0), vis_spheres, refl);

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

    Ray vis_ray = ReadVisRay();

    SphereTraceDesc params = SphereTraceDesc(epsilon, IMAGE_MAX_ITERS);

    vec3 color = Render(ray, vis_ray, params);

    fragColor = vec4(color, 1.0);
}
