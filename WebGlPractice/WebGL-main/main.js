'use strict';

let gl;         // The webgl context.
let surface;    // A surface model
let shProgram;  // A shader program
let spaceball;  // A TrackballRotator object

function deg2rad(angle) {
    return angle * Math.PI / 180;
}

/*======================  MODEL  ======================*/
// Wireframe surface: stores two sets of polylines – U and V.
function Model(name) {
    this.name = name;
    this.iVertexBuffer = gl.createBuffer();

    // Info for drawing each polyline: { offset, count } in vertices
    this.uLineInfo = [];
    this.vLineInfo = [];

    // Uploads all U + V lines into one buffer and remembers offsets.
    this.BufferData = function (uLines, vLines) {
        this.uLineInfo = [];
        this.vLineInfo = [];

        // Flatten all lines into one big array
        let vertices = [];
        let currentOffset = 0; // in vertices, not bytes

        // U polylines (constant y – parallels)
        for (let i = 0; i < uLines.length; i++) {
            const line = uLines[i];
            const vertCount = line.length / 3;
            this.uLineInfo.push({
                offset: currentOffset,
                count: vertCount
            });
            vertices.push(...line);
            currentOffset += vertCount;
        }

        // V polylines (constant angle – meridians)
        for (let i = 0; i < vLines.length; i++) {
            const line = vLines[i];
            const vertCount = line.length / 3;
            this.vLineInfo.push({
                offset: currentOffset,
                count: vertCount
            });
            vertices.push(...line);
            currentOffset += vertCount;
        }

        gl.bindBuffer(gl.ARRAY_BUFFER, this.iVertexBuffer);
        gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(vertices), gl.STATIC_DRAW);
    };

    this.Draw = function () {
        gl.bindBuffer(gl.ARRAY_BUFFER, this.iVertexBuffer);
        gl.vertexAttribPointer(shProgram.iAttribVertex, 3, gl.FLOAT, false, 0, 0);
        gl.enableVertexAttribArray(shProgram.iAttribVertex);

        // Draw U polylines (e.g. yellow)
        gl.uniform4fv(shProgram.iColor, [1, 1, 0, 1]);
        for (let i = 0; i < this.uLineInfo.length; i++) {
            const info = this.uLineInfo[i];
            gl.drawArrays(gl.LINE_STRIP, info.offset, info.count);
        }

        // Draw V polylines (e.g. cyan)
        gl.uniform4fv(shProgram.iColor, [0, 1, 1, 1]);
        for (let i = 0; i < this.vLineInfo.length; i++) {
            const info = this.vLineInfo[i];
            gl.drawArrays(gl.LINE_STRIP, info.offset, info.count);
        }
    };
}

/*======================  SHADER PROGRAM  ======================*/
function ShaderProgram(name, program) {

    this.name = name;
    this.prog = program;

    this.iAttribVertex = -1;
    this.iColor = -1;
    this.iModelViewProjectionMatrix = -1;

    this.Use = function () {
        gl.useProgram(this.prog);
    };
}

/*======================  DRAW  ======================*/
function draw() {
    gl.clearColor(0, 0, 0, 1);
    gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);

    // Projection
    let projection = m4.perspective(Math.PI / 8, 1, 8, 12);

    // View (trackball)
    let modelView = spaceball.getViewMatrix();

    let rotateToPointZero = m4.axisRotation([0.707, 0.707, 0], 0.7);
    let translateToPointZero = m4.translation(0, 0, -10);

    let matAccum0 = m4.multiply(rotateToPointZero, modelView);
    let matAccum1 = m4.multiply(translateToPointZero, matAccum0);

    let modelViewProjection = m4.multiply(projection, matAccum1);

    gl.uniformMatrix4fv(
        shProgram.iModelViewProjectionMatrix,
        false,
        modelViewProjection
    );

    surface.Draw();
}

/*======================  GEOMETRY  ======================*/

// Parabolic Humming-Top parametric function, mapped to (x, y, z)
// y ∈ [-h, h], beta ∈ [0, 2π]
function parabolicHummingTopVertex(y, beta, h, p) {
    // radius in the XZ-plane
    let rBase = Math.abs(y) - h;   // |y| - h
    let r = (rBase * rBase) / (2 * p); // (|y| - h)^2 / (2p)

    let x = r * Math.cos(beta);
    let z = r * Math.sin(beta);

    return [x, y, z];
}

function CreateSurfaceData() {
    // geometric parameters
    const h = 1.0;   // height of one sheet
    const p = 0.5;   // parabola parameter

    // grid resolution
    const vSegments = 40;  // along y (vertical)
    const uSegments = 64;  // angle segments

    let uLines = []; // U-curves: constant y, varying beta
    let vLines = []; // V-curves: constant beta, varying y

    // ----- U lines (parallels: circles) -----
    for (let j = 0; j <= vSegments; j++) {
        let y = -h + (2 * h * j) / vSegments; // from -h to h
        let line = [];

        for (let i = 0; i <= uSegments; i++) {
            let beta = 2 * Math.PI * i / uSegments;
            let [x, yy, z] = parabolicHummingTopVertex(y, beta, h, p);
            line.push(x, yy, z);
        }

        uLines.push(line);
    }

    // ----- V lines (meridians) -----
    for (let i = 0; i <= uSegments; i++) {
        let beta = 2 * Math.PI * i / uSegments;
        let line = [];

        for (let j = 0; j <= vSegments; j++) {
            let y = -h + (2 * h * j) / vSegments; // from -h to h
            let [x, yy, z] = parabolicHummingTopVertex(y, beta, h, p);
            line.push(x, yy, z);
        }

        vLines.push(line);
    }

    return { uLines, vLines };
}

/*======================  INIT GL  ======================*/
function initGL() {
    let prog = createProgram(gl, vertexShaderSource, fragmentShaderSource);

    shProgram = new ShaderProgram('Basic', prog);
    shProgram.Use();

    shProgram.iAttribVertex = gl.getAttribLocation(prog, "vertex");
    shProgram.iModelViewProjectionMatrix = gl.getUniformLocation(prog, "ModelViewProjectionMatrix");
    shProgram.iColor = gl.getUniformLocation(prog, "color");

    // Create and fill surface model
    const surfaceData = CreateSurfaceData();
    surface = new Model('ParabolicHummingTop');
    surface.BufferData(surfaceData.uLines, surfaceData.vLines);

    gl.enable(gl.DEPTH_TEST);
}

/*======================  SHADER CREATION (unchanged)  ======================*/
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
        initGL();  // initialize the WebGL graphics context
    }
    catch (e) {
        document.getElementById("canvas-holder").innerHTML =
            "<p>Sorry, could not initialize the WebGL graphics context: " + e + "</p>";
        return;
    }

    spaceball = new TrackballRotator(canvas, draw, 0);
    draw();
}
