// A common shader included (as is) in all Buffers

struct Value {
    float d; 
    int id;
};

struct Material {
    vec3 color;         // [0,1/pi]
    float roughness;    // [0,~7]
    float emission_strength;      // [0, inf]
    float metalness;    // 0.02-0.05 for non-metals, 0.6-0.9 for metals
};

struct Light {
    vec3 position;
    vec3 color;
};

struct Ray {
    vec3 P;
    float Tmin;
    vec3 V;
    float Tmax;
};

struct SphereTraceDesc {
    float epsilon;  // Stopping distance to surface
    int maxiters;   // Maximum iteration count
};

struct TraceResult {
    float T;    // Distance taken on ray
    int flags;  // flags bit 0:   distance condition: true if travelled to far t > t_max
                // flags bit 1:   surface condition:  true if distance to surface is small < error threshold
                // flags bit 2:   max. iter. condition:  true if none of the above, exited the loop because i > maxsteps
                // bit 2+: material id bits
    int steps;
};  

const float pi = 3.1415926535897932384626433832795;

const vec3 EyeStartPosition = vec3(2.0, 2.0, 2.0);

#define VIS_RAY_MAX_ITERS 10
#define MAX_DIST 1000.0


// common useful functions
float min3(float a, float b, float c) {
    return min(min(a, b), c);
}
float max3(float a, float b, float c) {
    return max(max(a, b), c);
}
float max3(vec3 a) {
    return max(max(a.x, a.y), a.z);
}
float min3(vec3 a) {
    return min(min(a.x, a.y), a.z);
}
float dot2(vec3 a) {
    return dot(a, a);
}

// union on the original sdf representation
float Unite(float a, float b) { 
    return min(a, b);
}
// same union but also carrying mat id
Value Unite(Value a, Value b) {
    if (b.d < a.d) {
        return b;
    } 
    return a;
}
// creates a Value from the (float,int)
Value Unite(Value a, float b_d, int b_id) {
    if (b_d < a.d) {
        return Value(b_d, b_id);
    } 
    return a;
}
// pairs and calls Unite(Value,Value)
Value Unite(float a_d, Value b, int b_id) {
    return Unite(Value(a_d, b_id), b);
}


Value Intersect(Value a, Value b) {
    if (b.d > a.d) {
        return b;
    }
    return a;
}

Value Substraction(Value a, Value b) {
    if (b.d > -a.d) {
        return b;
    }
    return a;
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
    //s            + -1.1428571428571428 * (pow(q, 3.5) / (1.0 + pow(q, 7.0))) * cos(7.0 * t)
    //            + 0.8888888888888888  * (pow(q, 4.5) / (1.0 + pow(q, 9.0))) * cos(9.0 * t)
    //            + -0.7272727272727273 * (pow(q, 5.5) / (1.0 + pow(q, 11.0))) * cos(11.0 * t);

    return theta;
}

Value SceneSDF(vec3 p) {
    const float freq = 2.0;

    float first_sphere_angle = pendulum(clamp(mod(freq * iTime, 2.0 * pi), 0.5 * pi, 1.5 * pi) + pi);
    float last_sphere_angle = pendulum(clamp(mod(freq * iTime + pi, 2.0 * pi), 0.5 * pi, 1.5 * pi));

    vec3 first_sphere_pos =    vec3(0.0, -13.0 + 7.0 * cos(first_sphere_angle), -4.0 - 7.0 * sin(first_sphere_angle));
    vec3 last_sphere_pos =     vec3(0.0, -13.0 + 7.0 * cos(last_sphere_angle),   4.0 - 7.0 * sin(last_sphere_angle));
    vec3 first_cable_end_pos = vec3(0.0, -13.0 + 6.0 * cos(first_sphere_angle), -4.0 - 6.0 * sin(first_sphere_angle));
    vec3 last_cable_end_pos =  vec3(0.0, -13.0 + 6.0 * cos(last_sphere_angle),   4.0 - 6.0 * sin(last_sphere_angle));

    p = vec3(p.x, p.y + 10.0, p.z);

    float d = MAX_DIST;
    Value v = Value(MAX_DIST, 0);
    v = Unite(v, Value(sdBoxFrame(p + vec3(0.0, -8.0, 0.0), vec3(3.0, 5.0, 6.0), 0.1), 1));
    v = Unite(v, Value(max(sdRoundBox(p + vec3(0.0, -3.0, 0.0), vec3(4.0, 1.0, 7.0), 0.5), -(p.y - 2.99)), 0));
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
