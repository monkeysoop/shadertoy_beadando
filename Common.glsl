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
