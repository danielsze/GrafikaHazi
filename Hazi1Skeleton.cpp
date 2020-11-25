//=============================================================================================
// Mintaprogram: Zöld háromszög. Ervenyes 2019. osztol.
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

#pragma GCC diagnostic push

// turn off the specific warning. Can also use "-Wall"
#pragma GCC diagnostic ignored "-Wall"


#include "framework.h"

// turn the warnings back on
#pragma GCC diagnostic pop

// vertex shader in GLSL: It is a Raw string (C++11) since it contains new line characters
const char * const vertexSource = R"(
	#version 330				// Shader 3.3
	precision highp float;		// normal floats, makes no difference on desktop computers

	uniform mat4 MVP;			// uniform variable, the Model-View-Projection transformation matrix
	layout(location = 0) in vec2 vp;	// Varying input: vp = vertex position is expected in attrib array 0

	void main() {
		gl_Position = vec4(vp.x, vp.y, 0, 1) * MVP;		// transform vp from modeling space to normalized device space
	}
)";

// fragment shader in GLSL
const char * const fragmentSource = R"(
	#version 330			// Shader 3.3
	precision highp float;	// normal floats, makes no difference on desktop computers
	
	uniform vec3 color;		// uniform variable, the color of the primitive
	out vec4 outColor;		// computed color of the current pixel

	void main() {
		outColor = vec4(color, 1);	// computed color is the color of the primitive
	}
)";



template<class T> struct Node{

    Node(T data):data(data){}
    Node<T>* previous;
    T data;
    Node<T>* next;
    
    Node<T>* remove(){
        previous->next = next;
        next->previous = previous;
        return this;
    }
    Node<T>* insert(const Node<T>* param){
        Node<T>* tmp = next;
        this->next = param;
        param->previous=this;
        param->next = tmp;
    }
};

float hyperDistance(vec2 a, vec2 b){
    return sqrt(b.x*b.x + b.y*b.y)/(1-a.x*a.x-a.y*a.y);
}

float angleBetweenVectors(vec2 v1, vec2 v2){
    float angle =acosf(dot(v1,v2)/(length(v1)*length(v2)));
    return angle;
}
class Drawable{

protected:
    GPUProgram* gpuProgram;
    unsigned int vao;       // virtual world on the GPU
    
    bool draw = false;
    
    virtual void _Create()=0;
    virtual void _Draw() = 0;
public:
    void Create(GPUProgram* prog){
        if(prog == NULL)throw "GPU program nem jott letre!";
        gpuProgram = prog;
        glGenVertexArrays(1, &vao);
        glBindVertexArray(vao);
        draw = true;
        _Create();
    }
    void Draw(){
        if(!draw)return;
        _Draw();
    }
    virtual ~Drawable()=default;
};


class TriangleFill:public Drawable{
    std::vector<vec2> *border;
    std::vector<vec2> vertices;
    std::vector<vec2> controlPoints;
    
    float distance_a, distance_b, distance_c;
    
    void calcVertices(){
//        vec2 center = vec2(0.0f, 0.0f);
//        vertices.push_back(center);
//        for(float phi = 0.0f; phi<M_PI*2+0.1f; phi+=0.1){
//            vertices.push_back(0.5f*vec2(sinf(phi), cosf(phi)));
//        }
        Node<vec2> *previous = NULL;
        for (vec2 vertex:*border) {
            Node<vec2> *current = new Node<vec2>(vertex);
            current->previous = previous;
            if(previous)
                previous->next = current;
            previous = current;
        }
        Node<vec2> *lastNode =  previous;
        
        Node<vec2> * currentNode = lastNode;
        Node<vec2>* first;
        while(currentNode->previous){
            first = currentNode->previous;
            currentNode= first;
        }
        first->previous = lastNode;
        lastNode->next = first;
        
        
        size_t n = border->size();
        
        vec2 origin(0.0f,0.0f);
        Node<vec2> *a, *b, *c;
        b = first;
        a = b->previous;
        c = b->next;
        
        while(n>3){
            
            float angle = angleBetweenVectors(c->data - b->data, a->data - b->data);
            if( angle < M_PI-0.2f){
                vertices.push_back(a->data);
                vertices.push_back(b->data);
                vertices.push_back(c->data);
                Node<vec2>* tmp = b->next;
                delete b->remove();
                b = tmp;
                n--;
            }
            int n = 0;
            b = b->next;
            a = b->previous;
            c = b->next;
            
            

        }
        
        vertices.push_back(a->data);
        vertices.push_back(b->data);
        vertices.push_back(c->data);
        
        
}
public:
    void setBorder(std::vector<vec2>* borderVertices){
        border=borderVertices;
    }
    TriangleFill(){}
    void addPoint(vec2 cp){
        controlPoints.push_back(cp);
    }
    void _Create() override{
        
        unsigned int vbo[2];        // vertex buffer objects
        glGenBuffers(2, &vbo[0]);    // Generate 2 vertex buffer objects

        // vertex coordinates: vbo[0] -> Attrib Array 0 -> vertexPosition of the vertex shader
        glBindBuffer(GL_ARRAY_BUFFER, vbo[0]); // make it active, it is an array
        
        calcVertices();
        float vertexCoords[vertices.size()*2];
        int idx = 0;
        for(vec2 v : vertices){
            vertexCoords[idx]=v.x;
            vertexCoords[idx+1]=v.y;
            idx+=2;
        }
        
        glBufferData(GL_ARRAY_BUFFER,      // copy to the GPU
                     vertices.size()*sizeof(float)*2, // number of the vbo in bytes
                     vertexCoords,           // address of the data array on the CPU
                     GL_STATIC_DRAW);       // copy to that part of the memory which is not modified
        // Map Attribute Array 0 to the current bound vertex buffer (vbo[0])
        glEnableVertexAttribArray(0);
        // Data organization of Attribute Array 0
        glVertexAttribPointer(0,            // Attribute Array 0
                              2, GL_FLOAT,  // components/attribute, component type
                              GL_FALSE,        // not in fixed point format, do not normalized
                              0, NULL);     // stride and offset: it is tightly packed

        // vertex colors: vbo[1] -> Attrib Array 1 -> vertexColor of the vertex shader
        glBindBuffer(GL_ARRAY_BUFFER, vbo[1]); // make it active, it is an array
        static float vertexColors[] = { 1, 1, 1,  0, 1, 0,  0, 0, 1 };    // vertex data on the CPU
        glBufferData(GL_ARRAY_BUFFER, sizeof(vertexColors), vertexColors, GL_STATIC_DRAW);    // copy to the GPU

        // Map Attribute Array 1 to the current bound vertex buffer (vbo[1])
        glEnableVertexAttribArray(1);  // Vertex position
        // Data organization of Attribute Array 1
        glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 0, NULL); // Attribute Array 1, components/attribute, component type, normalize?, tightly packed
        
    }
    
    void _Draw() override{
        int location = glGetUniformLocation(gpuProgram->getId(), "color");
        
        glUniform3f(location, 0.0f, 0.55f, 0.80f); // 3 floats

        float MVPtransf[4][4] = { 1, 0, 0, 0,    // MVP matrix,
                                  0, 1, 0, 0,    // row-major!
                                  0, 0, 1, 0,
                                  0, 0, 0, 1 };

        location = glGetUniformLocation(gpuProgram->getId(), "MVP");    // Get the GPU location of uniform variable MVP
        glUniformMatrix4fv(location, 1, GL_TRUE, &MVPtransf[0][0]);    // Load a 4x4 row-major float matrix to the specified location
    
        glBindVertexArray(vao);  // Draw call
        glDrawArrays(GL_TRIANGLES, 0 /*startIdx*/,  vertices.size()/*# Elements*/);
    }
    
};




class SiriusTriangle:public Drawable{
    std::vector<vec2>* points;
    std::vector<vec2> vertices;
    std::vector<vec2> segment_a;
    std::vector<vec2> segment_b;
    std::vector<vec2> segment_c;
    float triangle_length = 0;
    float angle_a, angle_b, angle_c;
    
   void calculateVertices(){
        
        calculateSegment(points->at(0), points->at(1), &segment_a);
        calculateSegment(points->at(1), points->at(2), &segment_b);
        calculateSegment(points->at(2), points->at(0), &segment_c);
        
        
        vertices.reserve(segment_a.size()+segment_b.size()+segment_c.size() + 1);
        vertices.insert(vertices.end(),segment_a.begin(), segment_a.end());
        vertices.insert(vertices.end(),segment_b.begin(), segment_b.end());
        vertices.insert(vertices.end(),segment_c.begin(), segment_c.end());
        vertices.push_back(points->at(0));

       
       
        vec2 A = points->at(0);
        vec2 B = points->at(1);
        vec2 C = points->at(2);
               
        angle_a = angleBetweenVectors(segment_a[1]-A, -(*(segment_c.end()-1))-A);
        angle_b = angleBetweenVectors(segment_b[1]-B, (*(segment_a.end()-1))-B);
        angle_c = angleBetweenVectors(segment_a[1]-C, (*(segment_b.end()-1))-C);
        
       /*
        vec2 previous = vertices.at(0);
        for(vec2 vertex:vertices){
            triangle_length+= length(vertex-previous);
            previous = vertex;
        }
        */
        float x, y, dx, dy;
        float ds=0;
        for(int i = 0; i < 3;i++){
            x = points->at(i).x;
            y = points->at(i).y;
            dx = points->at((i+1)%3).x;
            dy = points->at((i+1)%3).y;
            ds += sqrt(dx*dx + dy*dy)/(1-x*x-y*y);
        }
       triangle_length=ds;
       
       
    }
    void calculateSegment(vec2 p1, vec2 p2, std::vector<vec2> *vertices){
        //Find center //TODO
        vec2 center;
        float segment_length = 0.0f;
        float ax, ay, bx, by;
        ax= p1.x; ay = p1.y; bx= p2.x; by=p2.y;
        
        
        
        //Azert ilyen csunya mert matlabbal szamoltam, de onalloan csinaltam, meg tudom mutatni.
        float y = (- pow(ax, 2)*bx + ax*pow(bx, 2) + ax*pow(by, 2) + ax - pow(ay, 2)*bx - bx)/(2*ax*by - 2*ay*bx);
        float x = -(- pow(ax, 2) - pow(ay, 2) + 2*y*ay + pow(bx, 2) + pow(by, 2) - 2*y*by)/(2*ax - 2*bx);
        
        float part1=pow(ay - (- pow(ax, 2)*bx + ax*pow(bx, 2) + ax*pow(by, 2) + ax - pow(ay, 2)*bx - bx)/(2*ax*by - 2*ay*bx),2);
        float part2 = pow(ax - (pow(ax, 2) + pow(ay, 2) - pow(bx, 2) - pow(by, 2) - (2*ay*(- pow(ax, 2)*bx + ax*pow(bx, 2) + ax*pow(by, 2) + ax - pow(ay, 2)*bx - bx))/(2*ax*by - 2*ay*bx) +
                             (2*by*(- pow(ax, 2)*bx + ax*pow(bx, 2) + ax*pow(by, 2) + ax - pow(ay, 2)*bx - bx))/(2*ax*by - 2*ay*bx))/(2*ax - 2*bx),2);
        float radius = sqrt( part1 + part2 );
        
        //Ellenorzo osszegek, nulla korul kene legyenek
        center.x = x;
        center.y = y;
        
        float s1, s2, s3;
        s1 = abs(pow(radius, 2) - dot(p1-center, p1-center));
        s2 = abs(pow(radius, 2) + 1.0f - dot(center,center));
        s3 = abs(dot(center-(p1+p2)/2,(p1-p2)));
        
        
        vec2 center_to_p1 = p1-center;
        vec2 center_to_p2 = p2-center;
        
        float start_angle = atan2(center_to_p1.y, center_to_p1.x);
        float end_angle = atan2(center_to_p2.y, center_to_p2.x);
        
        if(end_angle<start_angle){
            float tmp = end_angle;
            end_angle=start_angle;
            start_angle=tmp;
        }
        
        float delta = end_angle-start_angle;
        
        
        for(float phi = start_angle; phi<end_angle; phi+=delta/10){
            vec2 point = center + radius* vec2(cosf(phi), sinf(phi));
            vertices->push_back(point);
        }
        
    }
    
    bool isWithinRectangle(vec2 A,vec2 B,vec2 C,vec2 D,vec2 point){
        float area1 = areaOfTriangle(A, B, point)+areaOfTriangle(A, C, point) + areaOfTriangle(B, point, D)+areaOfTriangle(C, point, D);
        float area2 = areaOfTriangle(A, B, C)*2;
        return area2>=area1;
    }
    float areaOfTriangle(vec2 A, vec2 B, vec2 C){
        float a = length(A-B);
        float b = length(A-C);
        float c = length(B-C);
        float s = (a*a + b*b+ c*c)/2;
        return sqrt(s*(s-a)*(s-b)*(s-c));
        
    }
public:
    SiriusTriangle(std::vector<vec2>* vertices_)
        :points(vertices_){
        
    }
    std::vector<vec2>* getVertices(){
        return &this->vertices;
    }
    void _Create() override{
        
        unsigned int vbo[2];        // vertex buffer objects
        glGenBuffers(2, &vbo[0]);    // Generate 2 vertex buffer objects
        calculateVertices();
        glBindBuffer(GL_ARRAY_BUFFER, vbo[0]); // make it active, it is an array
        
        
        float vertexCoords[vertices.size()*2];
        int idx=0;
        for(vec2 vertex : vertices){
          vertexCoords[idx]=vertex.x;
          vertexCoords[idx+1]=vertex.y;
          idx+=2;
        }   // vertex data on the CPU
              
              
              
        glBufferData(GL_ARRAY_BUFFER,      // copy to the GPU
                     sizeof(vertexCoords), // number of the vbo in bytes
                     vertexCoords,           // address of the data array on the CPU
                     GL_STATIC_DRAW);       // copy to that part of the memory which is not modified
        // Map Attribute Array 0 to the current bound vertex buffer (vbo[0])
        glEnableVertexAttribArray(0);
        // Data organization of Attribute Array 0
        glVertexAttribPointer(0,            // Attribute Array 0
                              2, GL_FLOAT,  // components/attribute, component type
                              GL_FALSE,        // not in fixed point format, do not normalized
                              0, NULL);     // stride and offset: it is tightly packed
        
        static float vertexColors[] = { 1, 0, 0,  0, 1, 0,  0, 0, 1 };    // vertex data on the CPU
        // vertex colors: vbo[1] -> Attrib Array 1 -> vertexColor of the vertex shader
        glBindBuffer(GL_ARRAY_BUFFER, vbo[1]); // make it active, it is an array
        glBufferData(GL_ARRAY_BUFFER, sizeof(vertexColors), vertexColors, GL_STATIC_DRAW);    // copy to the GPU

        // Map Attribute Array 1 to the current bound vertex buffer (vbo[1])
        glEnableVertexAttribArray(1);  // Vertex position
        // Data organization of Attribute Array 1
        glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 0, NULL); // Attribute Array 1, components/attribute, component type, normalize?, tightly packed
      
        
        
        
        
        
        float rad_to_deg = 180/M_PI;
        printf("Az oldalak hossza osszesen: %f \nA szogei: %f %f %f fok.", triangle_length, angle_a*rad_to_deg, angle_b*rad_to_deg, angle_c*rad_to_deg);
    }
    
    void _Draw() override{
        int location = glGetUniformLocation(gpuProgram->getId(), "color");
        
        glUniform3f(location, 0.0f, 1.0f, 0.0f); // 3 floats

        float MVPtransf[4][4] = { 1, 0, 0, 0,    // MVP matrix,
                                  0, 1, 0, 0,    // row-major!
                                  0, 0, 1, 0,
                                  0, 0, 0, 1 };

        location = glGetUniformLocation(gpuProgram->getId(), "MVP");    // Get the GPU location of uniform variable MVP
        glUniformMatrix4fv(location, 1, GL_TRUE, &MVPtransf[0][0]);    // Load a 4x4 row-major float matrix to the specified location

        
        glBindVertexArray(vao);  // Draw call
        glDrawArrays(GL_LINE_STRIP, 0 /*startIdx*/, vertices.size() /*# Elements*/);
    }
};
class CerealPlane:public Drawable{
    std::vector<vec2> vertices;
    
    void calcVertices(){
        vec2 center = vec2(0.0f, 0.0f);
        vertices.push_back(center);
        for(float phi = 0.0f; phi<M_PI*2+0.1f; phi+=0.1){
            vertices.push_back(vec2(sinf(phi), cosf(phi)));
        }
    }
    
public:
    CerealPlane(){}
    void _Create() override{
        
        unsigned int vbo[2];        // vertex buffer objects
        glGenBuffers(2, &vbo[0]);    // Generate 2 vertex buffer objects

        // vertex coordinates: vbo[0] -> Attrib Array 0 -> vertexPosition of the vertex shader
        glBindBuffer(GL_ARRAY_BUFFER, vbo[0]); // make it active, it is an array
        
        calcVertices();
        float vertexCoords[vertices.size()*2];
        int idx = 0;
        for(vec2 v : vertices){
            vertexCoords[idx]=v.x;
            vertexCoords[idx+1]=v.y;
            idx+=2;
        }
        
        glBufferData(GL_ARRAY_BUFFER,      // copy to the GPU
                     vertices.size()*sizeof(float)*2, // number of the vbo in bytes
                     vertexCoords,           // address of the data array on the CPU
                     GL_STATIC_DRAW);       // copy to that part of the memory which is not modified
        // Map Attribute Array 0 to the current bound vertex buffer (vbo[0])
        glEnableVertexAttribArray(0);
        // Data organization of Attribute Array 0
        glVertexAttribPointer(0,            // Attribute Array 0
                              2, GL_FLOAT,  // components/attribute, component type
                              GL_FALSE,        // not in fixed point format, do not normalized
                              0, NULL);     // stride and offset: it is tightly packed

        // vertex colors: vbo[1] -> Attrib Array 1 -> vertexColor of the vertex shader
        glBindBuffer(GL_ARRAY_BUFFER, vbo[1]); // make it active, it is an array
        static float vertexColors[] = { 1, 0, 0,  0, 1, 0,  0, 0, 1 };    // vertex data on the CPU
        glBufferData(GL_ARRAY_BUFFER, sizeof(vertexColors), vertexColors, GL_STATIC_DRAW);    // copy to the GPU

        // Map Attribute Array 1 to the current bound vertex buffer (vbo[1])
        glEnableVertexAttribArray(1);  // Vertex position
        // Data organization of Attribute Array 1
        glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 0, NULL); // Attribute Array 1, components/attribute, component type, normalize?, tightly packed
        
    }
    
    void _Draw() override{
        int location = glGetUniformLocation(gpuProgram->getId(), "color");
        
        glUniform3f(location, 0.05f, 0.05f, 0.05f); // 3 floats

        float MVPtransf[4][4] = { 1, 0, 0, 0,    // MVP matrix,
                                  0, 1, 0, 0,    // row-major!
                                  0, 0, 1, 0,
                                  0, 0, 0, 1 };

        location = glGetUniformLocation(gpuProgram->getId(), "MVP");    // Get the GPU location of uniform variable MVP
        glUniformMatrix4fv(location, 1, GL_TRUE, &MVPtransf[0][0]);    // Load a 4x4 row-major float matrix to the specified location

        glBindVertexArray(vao);  // Draw call
        glDrawArrays(GL_TRIANGLE_FAN, 0 /*startIdx*/,  vertices.size()/*# Elements*/);
    }
    
};




std::vector<vec2> triangle_vertices;
bool drawing(false);
SiriusTriangle triangle(&triangle_vertices);
CerealPlane plane;
TriangleFill triangle_fill;
GPUProgram gpuProgram; // vertex and fragment shaders
unsigned int vao;
// Initialization, create an OpenGL context
void onInitialization() {
	glViewport(0, 0, windowWidth, windowHeight);
	// create program for the GPU
	gpuProgram.create(vertexSource, fragmentSource, "outColor");
    plane.Create(&gpuProgram);
    
}

// Window has become invalid: Redraw
void onDisplay() {
	glClearColor(0, 0, 0, 0);     // background color
	glClear(GL_COLOR_BUFFER_BIT); // clear frame buffer
    
    plane.Draw(); //Draw the plane first
   
    triangle_fill.Draw();
    triangle.Draw();
    
	glutSwapBuffers(); // exchange buffers for double buffering
}

// Key of ASCII code pressed
void onKeyboard(unsigned char key, int pX, int pY) {
	if (key == 'd') glutPostRedisplay();         // if d, invalidate display, i.e. redraw
}

// Key of ASCII code released
void onKeyboardUp(unsigned char key, int pX, int pY) {
}

// Move mouse with key pressed
void onMouseMotion(int pX, int pY) {	// pX, pY are the pixel coordinates of the cursor in the coordinate system of the operation system
	// Convert to normalized device space
	float cX = 2.0f * pX / windowWidth - 1;	// flip y axis
	float cY = 1.0f - 2.0f * pY / windowHeight;
    
	printf("Mouse moved to (%3.2f, %3.2f)\n", cX, cY);
}

// Mouse click event
void onMouse(int button, int state, int pX, int pY) { // pX, pY are the pixel coordinates of the cursor in the coordinate system of the operation system
	// Convert to normalized device space
	float cX = 2.0f * pX / windowWidth - 1;	// flip y axis
	float cY = 1.0f - 2.0f * pY / windowHeight;

	char * buttonStat;
	switch (state) {
	case GLUT_DOWN: buttonStat = "pressed"; break;
	case GLUT_UP:   buttonStat = "released"; break;
    default:buttonStat="";
	}

    
	switch (button) {
	case GLUT_LEFT_BUTTON:
            printf("Left button %s at (%3.2f, %3.2f)\n", buttonStat, cX, cY);
            if(buttonStat== "released" && !drawing){
                vec2 click(cX, cY);
                if(sqrt(pow(click.x,2) + pow(click.y,2)) <= 1.0f){
                    triangle_vertices.push_back(click);
                    triangle_fill.addPoint(click);
                    if(triangle_vertices.size()==3){
                        drawing = true;
                        triangle.Create(&gpuProgram);
                        std::vector<vec2> *vertices = triangle.getVertices();
                        triangle_fill.setBorder(vertices);
                        triangle_fill.Create(&gpuProgram);
                        
                        glutPostRedisplay();
                    }
                }
            }
            
            break;
	case GLUT_MIDDLE_BUTTON: printf("Middle button %s at (%3.2f, %3.2f)\n", buttonStat, cX, cY); break;
	case GLUT_RIGHT_BUTTON:  printf("Right button %s at (%3.2f, %3.2f)\n", buttonStat, cX, cY);  break;
	}
     
}

// Idle event indicating that some time elapsed: do animation here
void onIdle() {
	//long time = glutGet(GLUT_ELAPSED_TIME); // elapsed time since the start of the program
}
