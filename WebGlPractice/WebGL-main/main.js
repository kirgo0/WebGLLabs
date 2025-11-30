'use strict';

let gl;                 // WebGL context
let surface;            // Surface model (parabolic humming-top)
let shProgram;          // Shader program
let spaceball;          // Trackball rotator
let currentTime = 0.0;  // For rotating light

function deg2rad(angle) {
    return angle * Math.PI / 180;
}

// Helper: transform point by 4x4 matrix
function transformPoint(m, p) {
    const x = p[0], y = p[1], z = p[2];
    return [
        m[0] * x + m[4] * y + m[8]  * z + m[12],
        m[1] * x + m[5] * y + m[9]  * z + m[13],
        m[2] * x + m[6] * y + m[10] * z + m[14]
    ];
}

/*======================  MODEL – TRIANGLES WITH INDICES  ======================*/

function Model(name) {
    this.name = name;

    this.vbo = gl.createBuffer(); // vertex positions
    this.nbo = gl.createBuffer(); // vertex normals
    this.ibo = gl.createBuffer(); // indices
    this.indexCount = 0;

    /**
     * vertices: flat [x,y,z,...]
     * normals:  flat [nx,ny,nz,...]
     * indices:  flat [i0,i1,i2,...] (Uint16)
     */
    this.BufferData = function (vertices, normals, indices) {
        this.indexCount = indices.length;

        // Positions
        gl.bindBuffer(gl.ARRAY_BUFFER, this.vbo);
        gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(vertices), gl.STATIC_DRAW);

        // Normals
        gl.bindBuffer(gl.ARRAY_BUFFER, this.nbo);
        gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(normals), gl.STATIC_DRAW);

        // Indices
        gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, this.ibo);
        gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, new Uint16Array(indices), gl.STATIC_DRAW);
    };

    this.Draw = function () {
        // Bind positions
        gl.bindBuffer(gl.ARRAY_BUFFER, this.vbo);
        gl.vertexAttribPointer(shProgram.iAttribVertex, 3, gl.FLOAT, false, 0, 0);
        gl.enableVertexAttribArray(shProgram.iAttribVertex);

        // Bind normals
        gl.bindBuffer(gl.ARRAY_BUFFER, this.nbo);
        gl.vertexAttribPointer(shProgram.iAttribNormal, 3, gl.FLOAT, false, 0, 0);
        gl.enableVertexAttribArray(shProgram.iAttribNormal);

        // Bind indices and draw
        gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, this.ibo);
        gl.drawElements(gl.TRIANGLES, this.indexCount, gl.UNSIGNED_SHORT, 0);
    };
}

/*======================  SHADER PROGRAM WRAPPER  ======================*/

function ShaderProgram(name, program) {

    this.name = name;
    this.prog = program;

    this.iAttribVertex = -1;
    this.iAttribNormal = -1;

    this.iModelViewMatrix = -1;
    this.iProjectionMatrix = -1;

    this.iLightPos = -1;
    this.iAmbientColor = -1;
    this.iDiffuseColor = -1;
    this.iSpecularColor = -1;
    this.iShininess = -1;

    this.Use = function () {
        gl.useProgram(this.prog);
    };
}

/*======================  PARABOLIC HUMMING-TOP GEOMETRY  ======================*/

// parametric surface: (y is vertical axis)
// y ∈ [-h, h],  beta ∈ [0, 2π]
function parabolicHummingTopVertex(y, beta, h, p) {
    let rBase = Math.abs(y) - h;          // |y| - h
    let r = (rBase * rBase) / (2 * p);    // (|y| - h)^2 / (2p)

    let x = r * Math.cos(beta);
    let z = r * Math.sin(beta);

    return [x, y, z];
}

/**
 * Create surface mesh data for given U/V granularity.
 * uSeg: number of segments along angle (U)
 * vSeg: number of segments along vertical (V)
 *
 * Returns { positions, normals, indices }
 */
function CreateSurfaceData(uSeg, vSeg) {
    // defaults if not provided
    uSeg = uSeg || 40;
    vSeg = vSeg || 40;

    const h = 1.0;
    const p = 0.5;

    let positions = [];
    let normals = [];
    let indices = [];

    // Build grid of vertices
    for (let j = 0; j <= vSeg; j++) {
        let v = j / vSeg;
        let y = -h + 2.0 * h * v; // from -h to h

        for (let i = 0; i <= uSeg; i++) {
            let u = i / uSeg;
            let beta = 2.0 * Math.PI * u;

            let [x, yy, z] = parabolicHummingTopVertex(y, beta, h, p);
            positions.push(x, yy, z);

            // init normals to zero – will accumulate facet normals
            normals.push(0.0, 0.0, 0.0);
        }
    }

    const vertsPerRow = uSeg + 1;

    // Helper: add one triangle and accumulate facet-average normal
    function addFace(i0, i1, i2) {
        const ax = positions[3 * i0], ay = positions[3 * i0 + 1], az = positions[3 * i0 + 2];
        const bx = positions[3 * i1], by = positions[3 * i1 + 1], bz = positions[3 * i1 + 2];
        const cx = positions[3 * i2], cy = positions[3 * i2 + 1], cz = positions[3 * i2 + 2];

        // edges
        const ux = bx - ax, uy = by - ay, uz = bz - az;
        const vx = cx - ax, vy = cy - ay, vz = cz - az;

        // face normal = cross(u, v)
        let nx = uy * vz - uz * vy;
        let ny = uz * vx - ux * vz;
        let nz = ux * vy - uy * vx;

        // ----- FACET AVERAGE NORMAL -----
        // normalize face normal first (direction only), then accumulate to vertices
        let len = Math.hypot(nx, ny, nz);
        if (len > 1e-6) {
            nx /= len;
            ny /= len;
            nz /= len;
        }

        // accumulate
        normals[3 * i0]     += nx;
        normals[3 * i0 + 1] += ny;
        normals[3 * i0 + 2] += nz;

        normals[3 * i1]     += nx;
        normals[3 * i1 + 1] += ny;
        normals[3 * i1 + 2] += nz;

        normals[3 * i2]     += nx;
        normals[3 * i2 + 1] += ny;
        normals[3 * i2 + 2] += nz;
    }

    // Build triangles (two per quad) and compute normals
    for (let j = 0; j < vSeg; j++) {
        for (let i = 0; i < uSeg; i++) {
            const i0 = j * vertsPerRow + i;
            const i1 = i0 + 1;
            const i2 = i0 + vertsPerRow;
            const i3 = i2 + 1;

            // triangle 1: (i0, i2, i1)
            indices.push(i0, i2, i1);
            addFace(i0, i2, i1);

            // triangle 2: (i1, i2, i3)
            indices.push(i1, i2, i3);
            addFace(i1, i2, i3);
        }
    }

    // Normalize accumulated vertex normals
    for (let k = 0; k < normals.length; k += 3) {
        let nx = normals[k];
        let ny = normals[k + 1];
        let nz = normals[k + 2];
        let len = Math.hypot(nx, ny, nz);
        if (len > 1e-6) {
            normals[k]     = nx / len;
            normals[k + 1] = ny / len;
            normals[k + 2] = nz / len;
        } else {
            // fallback
            normals[k]     = 0.0;
            normals[k + 1] = 1.0;
            normals[k + 2] = 0.0;
        }
    }

    return { positions, normals, indices };
}

/*======================  DRAW  ======================*/

function draw() {
    gl.clearColor(0, 0, 0, 1);
    gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);

    // Projection (single point perspective)
    let projection = m4.perspective(Math.PI / 8, 1, 2, 20);

    // View
    let modelView = spaceball.getViewMatrix();

    let rotateToPointZero = m4.axisRotation([0.707, 0.707, 0], 0.7);
    let translateToPointZero = m4.translation(0, 0, -10);

    let matAccum0 = m4.multiply(rotateToPointZero, modelView);
    let matAccum1 = m4.multiply(translateToPointZero, matAccum0); // ModelViewMatrix

    // Send matrices
    gl.uniformMatrix4fv(shProgram.iModelViewMatrix, false, matAccum1);
    gl.uniformMatrix4fv(shProgram.iProjectionMatrix, false, projection);

    // Rotating light position in model space (circle around Y axis)
    const lightRadius = 5.0;
    const lightHeight = 2.0;
    const lightSpeed = 0.5; // radians per second

    let angle = currentTime * lightSpeed;
    let lightPosModel = [
        lightRadius * Math.cos(angle),
        lightHeight,
        lightRadius * Math.sin(angle)
    ];

    // Transform light position to eye space using ModelView matrix
    let lightPosEye = transformPoint(matAccum1, lightPosModel);
    gl.uniform3fv(shProgram.iLightPos, new Float32Array(lightPosEye));

    // Draw the surface
    surface.Draw();
}

/*======================  ANIMATION LOOP  ======================*/

function animate(time) {
    currentTime = time * 0.001; // ms → seconds
    draw();
    requestAnimationFrame(animate);
}

/*======================  INIT GL  ======================*/

function initGL() {
    let prog = createProgram(gl, vertexShaderSource, fragmentShaderSource);

    shProgram = new ShaderProgram('Phong', prog);
    shProgram.Use();

    // Attributes
    shProgram.iAttribVertex = gl.getAttribLocation(prog, "vertex");
    shProgram.iAttribNormal = gl.getAttribLocation(prog, "normal");

    // Uniforms
    shProgram.iModelViewMatrix  = gl.getUniformLocation(prog, "ModelViewMatrix");
    shProgram.iProjectionMatrix = gl.getUniformLocation(prog, "ProjectionMatrix");

    shProgram.iLightPos      = gl.getUniformLocation(prog, "uLightPos");
    shProgram.iAmbientColor  = gl.getUniformLocation(prog, "uAmbientColor");
    shProgram.iDiffuseColor  = gl.getUniformLocation(prog, "uDiffuseColor");
    shProgram.iSpecularColor = gl.getUniformLocation(prog, "uSpecularColor");
    shProgram.iShininess     = gl.getUniformLocation(prog, "uShininess");

    // Lighting constants (you can tweak)
    gl.uniform3fv(shProgram.iAmbientColor,  new Float32Array([0.1, 0.1, 0.15]));
    gl.uniform3fv(shProgram.iDiffuseColor,  new Float32Array([0.3, 0.6, 0.9]));
    gl.uniform3fv(shProgram.iSpecularColor, new Float32Array([0.9, 0.9, 0.9]));
    gl.uniform1f(shProgram.iShininess, 32.0);

    surface = new Model('ParabolicHummingTop');

    gl.enable(gl.DEPTH_TEST);
    gl.enable(gl.CULL_FACE);
    gl.cullFace(gl.BACK);
}

/*======================  SHADER CREATION – UNCHANGED LOGIC  ======================*/

function createProgram(gl, vShader, fShader) {
    let vsh = gl.createShader(gl.VERTEX_SHADER);
    gl.shaderSource(vsh, vShader);
    gl.compileShader(vsh);
    if (!gl.getShaderParameter(vsh, gl.COMPILE_STATUS)) {
        throw new Error("Error in vertex shader:  " + gl.getShaderInfoLog(vsh));
    }

    let fsh = gl.createShader(gl.FRAGMENT_SHADER);
    gl.shaderSource(fsh, fShader);
    gl.compileShader(fsh);
    if (!gl.getShaderParameter(fsh, gl.COMPILE_STATUS)) {
        throw new Error("Error in fragment shader:  " + gl.getShaderInfoLog(fsh));
    }

    let prog = gl.createProgram();
    gl.attachShader(prog, vsh);
    gl.attachShader(prog, fsh);
    gl.linkProgram(prog);
    if (!gl.getProgramParameter(prog, gl.LINK_STATUS)) {
        throw new Error("Link error in program:  " + gl.getProgramInfoLog(prog));
    }
    return prog;
}

/*======================  INIT  ======================*/

function init() {
    let canvas;
    try {
        canvas = document.getElementById("webglcanvas");
        gl = canvas.getContext("webgl");
        if (!gl) {
            throw "Browser does not support WebGL";
        }
    }
    catch (e) {
        document.getElementById("canvas-holder").innerHTML =
            "<p>Sorry, could not get a WebGL graphics context.</p>";
        return;
    }
    try {
        initGL();
    }
    catch (e) {
        document.getElementById("canvas-holder").innerHTML =
            "<p>Sorry, could not initialize the WebGL graphics context: " + e + "</p>";
        return;
    }

    spaceball = new TrackballRotator(canvas, draw, 0);

    // Hook up sliders for U/V granularity
    const uSlider = document.getElementById("uResolution");
    const vSlider = document.getElementById("vResolution");
    const uVal = document.getElementById("uVal");
    const vVal = document.getElementById("vVal");

    function updateSurfaceFromSliders() {
        const uSeg = parseInt(uSlider.value);
        const vSeg = parseInt(vSlider.value);

        uVal.textContent = uSeg.toString();
        vVal.textContent = vSeg.toString();

        const data = CreateSurfaceData(uSeg, vSeg);
        surface.BufferData(data.positions, data.normals, data.indices);

        draw();
    }

    uSlider.oninput = updateSurfaceFromSliders;
    vSlider.oninput = updateSurfaceFromSliders;

    updateSurfaceFromSliders();

    requestAnimationFrame(animate);
}
