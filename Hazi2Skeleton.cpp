//=============================================================================================
// Mintaprogram: Zold haromszog. Ervenyes 2019. osztol.
//
// A beadott program csak ebben a fajlban lehet, a fajl 1 byte-os ASCII karaktereket tartalmazhat, BOM kihuzando.
// Tilos:
// - mast "beincludolni", illetve mas konyvtarat hasznalni
// - faljmuveleteket vegezni a printf-et kiveve
// - Mashonnan atvett programresszleteket forrasmegjeloles nelkul felhasznalni es
// - felesleges programsorokat a beadott programban hagyni!!!!!!!
// - felesleges kommenteket a beadott programba irni a forrasmegjelolest kommentjeit kiveve
// ---------------------------------------------------------------------------------------------
// A feladatot ANSI C++ nyelvu forditoprogrammal ellenorizzuk, a Visual Studio-hoz kepesti elteresekrol
// es a leggyakoribb hibakrol (pl. ideiglenes objektumot nem lehet referencia tipusnak ertekul adni)
// a hazibeado portal ad egy osszefoglalot.
// ---------------------------------------------------------------------------------------------
// A feladatmegoldasokban csak olyan OpenGL fuggvenyek hasznalhatok, amelyek az oran a feladatkiadasig elhangzottak
// A keretben nem szereplo GLUT fuggvenyek tiltottak.
//
// NYILATKOZAT
// ---------------------------------------------------------------------------------------------
// Nev    : Szeplaki Daniel
// Neptun : 
// ---------------------------------------------------------------------------------------------
// ezennel kijelentem, hogy a feladatot magam keszitettem, es ha barmilyen segitseget igenybe vettem vagy
// mas szellemi termeket felhasznaltam, akkor a forrast es az atvett reszt kommentekben egyertelmuen jeloltem.
// A forrasmegjeloles kotelme vonatkozik az eloadas foliakat es a targy oktatoi, illetve a
// grafhazi doktor tanacsait kiveve barmilyen csatornan (szoban, irasban, Interneten, stb.) erkezo minden egyeb
// informaciora (keplet, program, algoritmus, stb.). Kijelentem, hogy a forrasmegjelolessel atvett reszeket is ertem,
// azok helyessegere matematikai bizonyitast tudok adni. Tisztaban vagyok azzal, hogy az atvett reszek nem szamitanak
// a sajat kontribucioba, igy a feladat elfogadasarol a tobbi resz mennyisege es minosege alapjan szuletik dontes.
// Tudomasul veszem, hogy a forrasmegjeloles kotelmenek megsertese eseten a hazifeladatra adhato pontokat
// negativ elojellel szamoljak el es ezzel parhuzamosan eljaras is indul velem szemben.
//=============================================================================================

//=============================================================================================
// Computer Graphics Sample Program: Ray-tracing-let
//=============================================================================================
#include "framework.h"


const float tube_r = 0.6f;



vec3 operator/(const vec3& enumerator, const vec3& denominator){
    return vec3(enumerator.x/denominator.x, enumerator.y/denominator.y, enumerator.z/denominator.z);
}
namespace Material {

    enum MaterialType{ ROUGH, REFLECTIVE };
    struct Material {
        vec3 ka, kd, ks;
        float  shininess;
        vec3 F0;
        MaterialType type;
        Material(MaterialType type) : type(type) {}
    };
    struct ReflectiveMaterial : public Material {
        ReflectiveMaterial(const vec3 &n, const vec3 &kappa) : Material(REFLECTIVE) {
            vec3 one {1, 1, 1};
            F0 = ((n - one) * (n - one) + kappa * kappa) / ((n + one) * (n + one) + kappa * kappa);
        }
    };
    struct RoughMaterial : public Material {
    RoughMaterial(vec3 _kd, vec3 _ks, float _shininess) : Material(ROUGH) {
        ka = _kd * M_PI;
        kd = _kd;
        ks = _ks;
        shininess = _shininess;
    }
    };

static Material* gold = new ReflectiveMaterial(vec3(0.17f, 0.35f, 1.5f), vec3(3.1f, 2.7f, 1.9f));
static Material* silver = new ReflectiveMaterial(vec3(0.14f, 0.16f, 0.13f), vec3(4.1f, 2.3f, 3.1f));
static Material* rough_pink = new RoughMaterial(vec3(0.3f, 0.1f, 0.2f), vec3(1,1,1), 25.0f);
static Material* rough_brownish = new RoughMaterial(vec3(0.5f, 0.3f, 0.1f), vec3(1,1,1), 20.0f);
static Material* rough_blue = new RoughMaterial(vec3(0.1f, 0.1f, 0.8f), vec3(2,2,2), 40.0f);
    
}

namespace Raycasting {
    struct Hit {
        float t;
        vec3 position, normal;
        Material::Material * material;
        Hit() { t = -1; }
    };

    struct Ray {
        vec3 start, dir;
        Ray(vec3 _start, vec3 _dir) { start = _start; dir = normalize(_dir); }
    };

    struct Light {
        vec3 direction;
        vec3 Le;
        Light(vec3 _direction, vec3 _Le) {
            direction = normalize(_direction);
            Le = _Le;
        }
    };
}
mat4 transpose(mat4 m){
    return mat4(vec4(m.rows[0].x,m.rows[1].x,m.rows[2].x,m.rows[3].x),
                vec4(m.rows[0].y,m.rows[1].y,m.rows[2].y,m.rows[3].y),
                vec4(m.rows[0].z,m.rows[1].z,m.rows[2].z,m.rows[3].z),
                vec4(m.rows[0].w,m.rows[1].w,m.rows[2].w,m.rows[3].w));
}

namespace Surface {
class Intersectable {
protected:
    Material::Material * material;
    vec2 vertical_scale;
public:
    virtual Raycasting::Hit intersect(const Raycasting::Ray& ray) = 0;
    Intersectable(vec2 vertical){vertical_scale = vertical;}
};
class Quadric: public Intersectable{
protected:
    
    vec3 gradf(vec4 r) {
        vec4 g = r * Q * 2.0f;
        return {g.x, g.y, g.z};
    }
    float f(vec3 r){
        vec4 _r(r.x, r.y, r.z, 1.0f);
        return dot(_r * Q, _r);
    }
    mat4 Q;
public:
    Quadric() = delete;
    
    Quadric(mat4 _Q, const vec3& translate, const vec3& scale, Material::Material* mat, vec2 vertical = vec2(2.0f, -2.0f))
    :Intersectable(vec2(vertical)){
        Q = _Q;
        material = mat;
        mat4 inv_scale = ScaleMatrix(vec3(1/scale.x, 1/scale.y, 1/scale.z));
        mat4 inv_trans = TranslateMatrix((-1)*translate);
        Q = inv_scale * Q * transpose(inv_scale);
        Q = inv_trans * Q * transpose(inv_trans);
    }
    virtual Raycasting::Hit intersect(const Raycasting::Ray& ray)override{
        Raycasting::Hit hit;

        vec4 p = vec4(ray.start.x, ray.start.y, ray.start.z, 1);
        vec4 u = vec4(ray.dir.x, ray.dir.y, ray.dir.z, 0);

        float a = dot(u * Q, u);
        float b = dot(u * Q, p);
        float c = dot(p * Q, p);
        float discr = b * b - a * c;
        if (discr < 0) return hit;
        float sqrt_discr = sqrtf(discr);
        float t1 = (-b + sqrt_discr) / a;    // t1 >= t2 for sure
        float t2 = (-b - sqrt_discr) / a;
        vec3 p1 = ray.start + ray.dir * t1;
        vec3 p2 = ray.start + ray.dir * t2;
        if (t1 <= 0) return hit;
        if (p1.y > vertical_scale.x || p1.y < vertical_scale.y)
            t1 = -1;
        if (p2.y > vertical_scale.x || p2.y < vertical_scale.y)
            t2 = -1;
        hit.t = (t2 > 0) ? t2 : t1;
        hit.position = ray.start + ray.dir * hit.t;
        vec4 pos = vec4(hit.position.x, hit.position.y, hit.position.z, 1);
        hit.normal = normalize(gradf(pos));
        hit.material = material;
        return hit;
    }
};
}

namespace ObjectType {
class Ellipsoid: public Surface::Quadric{
public:
    Ellipsoid(const vec3& center, const vec3& scale, Material::Material* material, const float& top = 5.0f)
    :Surface::Quadric(mat4(1, 0, 0, 0,
                           0, 1, 0, 0,
                           0, 0, 1, 0,
                           0, 0, 0, -1), center, scale , material, vec2(top, -1000.0f)){}
};
class Cylinder : public Surface::Quadric{
    float height;
public:
    Cylinder(const vec3& center, float height, float radius, Material::Material* material, const float& top, const float& bot):
        Surface::Quadric(mat4(1, 0, 0, 0,
                              0, 0, 0, 0,
                              0, 0, 1, 0,
                              0, 0, 0, -1), center, vec3(radius,radius, radius), material, vec2(top, bot)){
            this->height = height;
            
        }
};
class Hyperboloid: public Surface::Quadric{
public:
    Hyperboloid(const vec3& center, const vec3& scale, Material::Material* material, float top, float bot):
        Surface::Quadric(mat4(1, 0, 0, 0,
        0, -1, 0, 0,
        0, 0, 1, 0,
        0, 0, 0, -1), center, scale, material, vec2(top, bot)){
        }
};
class Paraboloid: public Surface::Quadric{
public:
    Paraboloid(const vec3& center, float radius, Material::Material* material, float top, float bot)
        :Surface::Quadric(mat4(1, 0, 0, 0,
        0, 0, 0, 1,
        0, 0, 1, 0,
       0, 1, 0, 0), center, vec3(radius,radius, radius), material, vec2(top, bot) ){
            
    }
};
struct Sphere : public Ellipsoid {
    vec3 center;
    float radius;

    Sphere(const vec3& _center, float _radius, Material::Material* _material)
    :Ellipsoid(center, vec3(_radius, _radius, _radius), _material, 50.0f) {
    }
};
}

class Camera {
    vec3 eye, lookat, right, up;
public:
    void set(vec3 _eye, vec3 _lookat, vec3 vup, float fov) {
        eye = _eye;
        lookat = _lookat;
        vec3 w = eye - lookat;
        float focus = length(w);
        right = normalize(cross(vup, w)) * focus * tanf(fov / 2);
        up = normalize(cross(w, right)) * focus * tanf(fov / 2);
    }
    Raycasting::Ray getRay(int X, int Y) {
        vec3 dir = lookat + right * (2.0f * (X + 0.5f) / windowWidth - 1) + up * (2.0f * (Y + 0.5f) / windowHeight - 1) - eye;
        return Raycasting::Ray(eye, dir);
    }
};


float rnd() { return (float)rand() / RAND_MAX; }

const float epsilon = 0.0001f;

class Scene {
    std::vector<Surface::Intersectable *> objects;
    std::vector<Raycasting::Light *> lights;
    std::vector<vec3> light_points;
    Camera camera;
    vec3 La;
    
    float radius(float k, float n, float b){
        if(k> n-b)
            return 1.0f;
        else
            return sqrtf(k-1.0f/2.0f)/sqrt(n-(b+1.0f)/2.0f);
    }//StackOverflow-rol kolcsonoztem az algoritmust.
    void setLightPoints(std::vector<vec3> &light_source ,size_t n = 40, float alpha = 1.0f){
        
        float b = roundf(alpha*sqrt(n));;
        float phi = (sqrtf(5.0f)+1.0f)/2.0f;
        for (int i = 1; i <= n; i++) {
            float r = radius(i, n, b);
            float theta = 2.0f * M_PI * i/(phi*phi);
            light_source.push_back(vec3(r*cosf(theta), 0.98f, r* sinf(theta)));
        }
    }
        
        
       
    
public:
    void build() {
        setLightPoints(light_points);
        
        vec3 eye = vec3(0, 0, 1.99f), vup = vec3(0, 1, 0), lookat = vec3(0, 0.0f, 0);
        float fov = 80 * M_PI / 180;
        camera.set(eye, lookat, vup, fov);

        La = vec3(0.5f,0.5f,0.55f);
        Raycasting::Light* sun = new Raycasting::Light(vec3(0.5, 5, 1), vec3(30,30,40));
        lights.push_back(sun);
        
        objects.push_back(new ObjectType::Hyperboloid(vec3(0, 0.98f,0), tube_r*vec3(1.0f, 1.0f, 1.0f), Material::silver, 2.0f, 0.98f));
        objects.push_back(new ObjectType::Ellipsoid(vec3(0,0,0), vec3(2.0f, 1.0f, 2.0f), Material::rough_pink, 0.98f));

        objects.push_back(new ObjectType::Cylinder(vec3(-1.0f, 0, 0.0f), 1.0f, 0.4, Material::rough_brownish, 0.75f, -1.0f));
        objects.push_back(new ObjectType::Ellipsoid(vec3(0.2f, -0.0f, -0.9f), vec3(0.3, 0.9, 0.3), Material::rough_blue));
        objects.push_back(new ObjectType::Paraboloid(vec3(0.75f, 0.2f, 0.7f), 0.15f, Material::gold, 1.0f, -1.0f));
        
        
    }

    void render(std::vector<vec4>& image) {
        for (int Y = 0; Y < windowHeight; Y++) {
#pragma omp parallel for
            for (int X = 0; X < windowWidth; X++) {
                
                vec3 color = trace(camera.getRay(X, Y));
                image[Y * windowWidth + X] = vec4(color.x, color.y, color.z, 1);
            }
        }
    }

    Raycasting::Hit firstIntersect(Raycasting::Ray ray) {
        Raycasting::Hit bestHit;
        for (Surface::Intersectable * object : objects) {
            Raycasting::Hit hit = object->intersect(ray);
            if (hit.t > 0 && (bestHit.t < 0 || hit.t < bestHit.t))  bestHit = hit;
        }
        if (dot(ray.dir, bestHit.normal) > 0) bestHit.normal = bestHit.normal * (-1);
        return bestHit;
    }

    bool shadowIntersect(Raycasting::Ray ray) {
        for (Surface::Intersectable * object : objects){
            Raycasting::Hit hit = object->intersect(ray);
            if (hit.t > 0)
                return true;
        }
        return false;
    }

    vec3 trace(Raycasting::Ray ray, int depth = 0) {
        
        
        if (depth > 5) {
            return La;
        }
        if (ray.start.y >= 0.98f){
           return La + lights[0]->Le * powf(dot(ray.dir, lights[0]->direction), 10);
        }
        
        Raycasting::Hit hit = firstIntersect(ray);
        if (hit.t < 0)
            return La + lights[0]->Le * powf(dot(ray.dir, lights[0]->direction), 10);
        vec3 outRadiance(0, 0, 0);
        if (hit.material->type == Material::ROUGH) {
            outRadiance = hit.material->ka * La;
            
            for (vec3 point : light_points) {
                vec3 partial_outRadiance = vec3(0, 0, 0);
                vec3 start = hit.position + hit.normal * epsilon;
                vec3 dir = normalize(point - start);
                Raycasting::Ray shadowRay(start, dir);
                float cosTheta = dot(hit.normal, dir);
                if (cosTheta > 0 && !shadowIntersect(shadowRay)) {
                    vec3 Le = trace(Raycasting::Ray(start, dir), depth + 1);
                    partial_outRadiance = partial_outRadiance + Le * hit.material->kd * cosTheta;
                    vec3 halfway = normalize(-ray.dir + dir);
                    float cosDelta = dot(hit.normal, halfway);
                    if (cosDelta > 0)
                        partial_outRadiance = partial_outRadiance + Le * hit.material->ks * powf(cosDelta, hit.material->shininess);
                }
                float dist = length(point-start);
                float delta_omega =(M_PI * tube_r*tube_r/light_points.size()) * cosTheta/(dist*dist);
                outRadiance = outRadiance + partial_outRadiance * delta_omega;
            }
        }
        else if (hit.material->type == Material::REFLECTIVE) {
            vec3 reflectedDir = ray.dir - hit.normal * dot(hit.normal, ray.dir) * 2.0f;
            float cosA = -dot(ray.dir, hit.normal);
            vec3 one(1, 1, 1);
            vec3 F = hit.material->F0 + (one - hit.material->F0) * pow(1 - cosA, 5);
            outRadiance = outRadiance + trace(Raycasting::Ray(hit.position + hit.normal * epsilon, reflectedDir), depth + 1) * F;
        }
        return outRadiance;
    }
};

GPUProgram gpuProgram; // vertex and fragment shaders
Scene scene;

// vertex shader in GLSL
const char *vertexSource = R"(
    #version 330
    precision highp float;

    layout(location = 0) in vec2 cVertexPosition;    // Attrib Array 0
    out vec2 texcoord;

    void main() {
        texcoord = (cVertexPosition + vec2(1, 1))/2;                            // -1,1 to 0,1
        gl_Position = vec4(cVertexPosition.x, cVertexPosition.y, 0, 1);         // transform to clipping space
    }
)";

// fragment shader in GLSL
const char *fragmentSource = R"(
    #version 330
    precision highp float;

    uniform sampler2D textureUnit;
    in  vec2 texcoord;            // interpolated texture coordinates
    out vec4 fragmentColor;        // output that goes to the raster memory as told by glBindFragDataLocation

    void main() {
        fragmentColor = texture(textureUnit, texcoord);
    }
)";

class FullScreenTexturedQuad {
    unsigned int vao;    // vertex array object id and texture id
    Texture texture;
public:
    FullScreenTexturedQuad(int windowWidth, int windowHeight, std::vector<vec4>& image)
        : texture(windowWidth, windowHeight, image)
    {
        glGenVertexArrays(1, &vao);    // create 1 vertex array object
        glBindVertexArray(vao);        // make it active

        unsigned int vbo;        // vertex buffer objects
        glGenBuffers(1, &vbo);    // Generate 1 vertex buffer objects

        // vertex coordinates: vbo0 -> Attrib Array 0 -> vertexPosition of the vertex shader
        glBindBuffer(GL_ARRAY_BUFFER, vbo); // make it active, it is an array
        float vertexCoords[] = { -1, -1,  1, -1,  1, 1,  -1, 1 };    // two triangles forming a quad
        glBufferData(GL_ARRAY_BUFFER, sizeof(vertexCoords), vertexCoords, GL_STATIC_DRAW);       // copy to that part of the memory which is not modified
        glEnableVertexAttribArray(0);
        glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, NULL);     // stride and offset: it is tightly packed
    }

    void Draw() {
        glBindVertexArray(vao);    // make the vao and its vbos active playing the role of the data source
        gpuProgram.setUniform(texture, "textureUnit");
        glDrawArrays(GL_TRIANGLE_FAN, 0, 4);    // draw two triangles forming a quad
    }
};

FullScreenTexturedQuad * fullScreenTexturedQuad;

// Initialization, create an OpenGL context
void onInitialization() {
    glViewport(0, 0, windowWidth, windowHeight);
    scene.build();

    std::vector<vec4> image(windowWidth * windowHeight);
    long timeStart = glutGet(GLUT_ELAPSED_TIME);
    scene.render(image);
    long timeEnd = glutGet(GLUT_ELAPSED_TIME);
    printf("Rendering time: %d milliseconds\n", (timeEnd - timeStart));

    // copy image to GPU as a texture
    fullScreenTexturedQuad = new FullScreenTexturedQuad(windowWidth, windowHeight, image);

    // create program for the GPU
    gpuProgram.create(vertexSource, fragmentSource, "fragmentColor");
}

// Window has become invalid: Redraw
void onDisplay() {
    fullScreenTexturedQuad->Draw();
    glutSwapBuffers();                                    // exchange the two buffers
}

// Key of ASCII code pressed
void onKeyboard(unsigned char key, int pX, int pY) {
}

// Key of ASCII code released
void onKeyboardUp(unsigned char key, int pX, int pY) {

}

// Mouse click event
void onMouse(int button, int state, int pX, int pY) {
}

// Move mouse with key pressed
void onMouseMotion(int pX, int pY) {
}

// Idle event indicating that some time elapsed: do animation here
void onIdle() {
}
