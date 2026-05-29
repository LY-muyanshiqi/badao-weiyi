"""
数字孪生3D可视化 — 拱坝数字孪生 + 多灾害传感器实时映射
Flask-SocketIO + Three.js 自定义拱坝几何 + 四种灾害可视化 + 热力云图
"""
import json
import time
import numpy as np
from flask import Flask, render_template_string, jsonify
from flask_socketio import SocketIO, emit

app = Flask(__name__)
app.config['SECRET_KEY'] = 'badao-weiyi-2.0'
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')

sensor_state = {
    'strain': [],
    'seepage': [],
    'displacement': [],
    'settlement': [],
    'seismic': [],
    'heatmap_grid': [],
    'alerts': [],
    'risk_level': '安全',
    'timestamp': time.time(),
}

TEMPLATE = r'''<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>坝道微医 2.0 — 数字孪生</title>
<style>
*{margin:0;padding:0;box-sizing:border-box}
body{font-family:"Microsoft YaHei",sans-serif;background:#060d1a;color:#c8d6e5;overflow:hidden}
#container{display:flex;height:100vh}
#viewer{flex:1;position:relative;cursor:grab}
#viewer:active{cursor:grabbing}
#panel{width:340px;background:#0a1428;overflow-y:auto;border-left:1px solid #1a2a44;display:flex;flex-direction:column}
.panel-section{padding:16px;border-bottom:1px solid #112240}
.panel-section h3{color:#4fc3f7;font-size:14px;margin-bottom:12px;text-transform:uppercase;letter-spacing:1px}
.risk-badge{display:inline-block;padding:6px 16px;border-radius:4px;font-size:20px;font-weight:bold;margin:8px 0}
.risk-安全{background:#0a3d0a;color:#4caf50;border:1px solid #2e7d32}
.risk-注意{background:#3d2e0a;color:#ff9800;border:1px solid #e65100}
.risk-警告{background:#3d0a0a;color:#f44336;border:1px solid #c62828}
.risk-危险{background:#3d0000;color:#ff1744;border:1px solid #b71c1c;animation:pulse-danger .8s infinite}
@keyframes pulse-danger{0%,100%{opacity:1}50%{opacity:.5}}
.metric-row{display:flex;justify-content:space-between;align-items:center;padding:8px 0;border-bottom:1px solid #0d1f3c}
.metric-row .label{font-size:12px;color:#5a7a9a}
.metric-row .value{font-size:16px;font-weight:bold}
.metric-row .unit{font-size:10px;color:#5a7a9a;margin-left:4px}
.hazard-indicator{display:flex;align-items:center;padding:6px 10px;margin:4px 0;border-radius:4px;background:#0d1f3c}
.hazard-dot{width:10px;height:10px;border-radius:50%;margin-right:10px;flex-shrink:0}
.hazard-info{flex:1}
.hazard-name{font-size:12px;color:#78909c}
.hazard-status{font-size:13px;font-weight:bold}
.alert-item{background:#1a0a0a;border-left:3px solid #f44336;padding:6px 10px;margin:4px 0;border-radius:2px;font-size:11px;animation:slide-in .3s ease}
@keyframes slide-in{from{transform:translateX(20px);opacity:0}to{transform:translateX(0);opacity:1}}
#alerts-container{max-height:200px;overflow-y:auto}
#status-bar{position:absolute;top:12px;left:50%;transform:translateX(-50%);background:rgba(6,13,26,.85);padding:8px 20px;border-radius:20px;font-size:12px;color:#4fc3f7;border:1px solid #1a3a5c;z-index:10;display:flex;gap:20px}
#status-bar span{display:flex;align-items:center;gap:6px}
.status-dot{width:6px;height:6px;border-radius:50%}
#tooltip{position:absolute;pointer-events:none;background:rgba(0,0,0,.85);color:#e0e0e0;padding:6px 10px;border-radius:4px;font-size:11px;display:none;z-index:20;white-space:nowrap}
</style>
</head>
<body>
<div id="container">
  <div id="viewer">
    <div id="status-bar">
      <span><div class="status-dot" style="background:#4caf50"></div> 系统在线</span>
      <span id="fps-display">FPS: --</span>
      <span id="sensor-count">传感器: 0</span>
    </div>
    <div id="tooltip"></div>
  </div>
  <div id="panel">
    <div class="panel-section">
      <h3>整体风险</h3>
      <div class="risk-badge risk-安全" id="risk-badge">安全</div>
      <div style="font-size:11px;color:#5a7a9a;margin-top:4px" id="risk-subtitle">各监测指标均在正常范围</div>
    </div>
    <div class="panel-section">
      <h3>灾害监测</h3>
      <div id="hazard-panel">
        <div class="hazard-indicator"><div class="hazard-dot" style="background:#4caf50"></div><div class="hazard-info"><div class="hazard-name">渗流灾害</div><div class="hazard-status">正常</div></div></div>
        <div class="hazard-indicator"><div class="hazard-dot" style="background:#4caf50"></div><div class="hazard-info"><div class="hazard-name">变形灾害</div><div class="hazard-status">正常</div></div></div>
        <div class="hazard-indicator"><div class="hazard-dot" style="background:#4caf50"></div><div class="hazard-info"><div class="hazard-name">沉降灾害</div><div class="hazard-status">正常</div></div></div>
        <div class="hazard-indicator"><div class="hazard-dot" style="background:#4caf50"></div><div class="hazard-info"><div class="hazard-name">地震灾害</div><div class="hazard-status">正常</div></div></div>
      </div>
    </div>
    <div class="panel-section">
      <h3>关键指标</h3>
      <div class="metric-row"><span class="label">最大应变</span><span><span class="value" id="val-strain">--</span><span class="unit">με</span></span></div>
      <div class="metric-row"><span class="label">渗流压力</span><span><span class="value" id="val-seepage">--</span><span class="unit">kPa</span></span></div>
      <div class="metric-row"><span class="label">沉降速率</span><span><span class="value" id="val-settlement">--</span><span class="unit">mm/月</span></span></div>
      <div class="metric-row"><span class="label">振动加速度</span><span><span class="value" id="val-seismic">--</span><span class="unit">gal</span></span></div>
    </div>
    <div class="panel-section" style="flex:1">
      <h3>告警记录</h3>
      <div id="alerts-container"><div style="color:#5a7a9a;font-size:11px">系统运行正常，无告警</div></div>
    </div>
  </div>
</div>

<script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/socket.io-client@4/dist/socket.io.min.js"></script>
<script>
// ===== SCENE SETUP =====
const container = document.getElementById('viewer');
const scene = new THREE.Scene();
scene.background = new THREE.Color(0x060d1a);
scene.fog = new THREE.Fog(0x060d1a, 15, 40);

const camera = new THREE.PerspectiveCamera(50, container.clientWidth/container.clientHeight, 0.1, 100);
camera.position.set(10, 6, 12);
const target = new THREE.Vector3(0, 1.5, 0);
camera.lookAt(target);

const renderer = new THREE.WebGLRenderer({antialias:true, alpha:true});
renderer.setSize(container.clientWidth, container.clientHeight);
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
renderer.shadowMap.enabled = true;
renderer.shadowMap.type = THREE.PCFSoftShadowMap;
renderer.toneMapping = THREE.ACESFilmicToneMapping;
renderer.toneMappingExposure = 1.2;
container.appendChild(renderer.domElement);

// ===== LIGHTING =====
const ambient = new THREE.AmbientLight(0x1a2a44, 0.6);
scene.add(ambient);
const sun = new THREE.DirectionalLight(0xffeedd, 1.2);
sun.position.set(15, 20, 5);
sun.castShadow = true;
sun.shadow.mapSize.set(2048, 2048);
sun.shadow.camera.near = 0.5;
sun.shadow.camera.far = 60;
sun.shadow.camera.left = -15;
sun.shadow.camera.right = 15;
sun.shadow.camera.top = 15;
sun.shadow.camera.bottom = -5;
sun.shadow.bias = -0.0001;
scene.add(sun);
const fill = new THREE.DirectionalLight(0x4488cc, 0.3);
fill.position.set(-5, 3, -3);
scene.add(fill);
const rim = new THREE.DirectionalLight(0xffffff, 0.2);
rim.position.set(0, 2, -8);
scene.add(rim);

// ===== TERRAIN =====
// Valley floor
const floorGeo = new THREE.PlaneGeometry(30, 30, 1, 1);
const floorMat = new THREE.MeshStandardMaterial({color:0x2d5016, roughness:0.9});
const floor = new THREE.Mesh(floorGeo, floorMat);
floor.rotation.x = -Math.PI/2;
floor.position.y = -3.5;
floor.receiveShadow = true;
scene.add(floor);

// Left valley wall (simplified as tilted box)
function createValleyWall(x, z, rotY, w, h, d) {
    const geo = new THREE.BoxGeometry(w, h, d);
    const mat = new THREE.MeshStandardMaterial({color:0x4a6741, roughness:0.8});
    const mesh = new THREE.Mesh(geo, mat);
    mesh.position.set(x, h/2 - 3.5, z);
    mesh.rotation.y = rotY;
    mesh.castShadow = true;
    mesh.receiveShadow = true;
    return mesh;
}
scene.add(createValleyWall(-5, 0, 0.3, 6, 5, 5));
scene.add(createValleyWall(5, 0, -0.3, 6, 5, 5));
// Back wall
const backWallGeo = new THREE.BoxGeometry(12, 4, 3);
const backWall = new THREE.Mesh(backWallGeo, new THREE.MeshStandardMaterial({color:0x3d5a34, roughness:0.8}));
backWall.position.set(0, -1.5, -4);
backWall.receiveShadow = true;
scene.add(backWall);

// ===== ARCH DAM GEOMETRY =====
function createArchDam(innerR, height, thetaStart, thetaSpan, baseThick, crestThick, segH, segV) {
    const vertices = [];
    const indices = [];
    const normals = [];

    function addVertex(x, y, z, nx, ny, nz) {
        vertices.push(x, y, z);
        normals.push(nx, ny, nz);
        return (vertices.length / 3) - 1;
    }

    // Create vertices on both faces
    for (let v = 0; v <= segV; v++) {
        const y = (v / segV) * height;
        const thick = baseThick - (v / segV) * (baseThick - crestThick);
        const outerR = innerR + thick;

        for (let h = 0; h <= segH; h++) {
            const theta = thetaStart + (h / segH) * thetaSpan;
            const cosT = Math.cos(theta);
            const sinT = Math.sin(theta);

            // Outer face vertex
            addVertex(outerR * sinT, y, -outerR * cosT, sinT, 0, -cosT);
            // Inner face vertex
            addVertex(innerR * sinT, y, -innerR * cosT, -sinT, 0, cosT);
        }
    }

    const cols = (segH + 1) * 2; // 2 faces per horizontal step

    // Build faces
    for (let v = 0; v < segV; v++) {
        for (let h = 0; h < segH; h++) {
            const base = v * cols + h * 2;
            // Outer face quad
            const o0 = base, o1 = base + 2, o2 = base + cols + 2, o3 = base + cols;
            indices.push(o0, o1, o2);
            indices.push(o0, o2, o3);
            // Inner face quad (normal reversed)
            const i0 = base + 1, i1 = base + cols + 1, i2 = base + cols + 3, i3 = base + 3;
            indices.push(i0, i1, i2);
            indices.push(i0, i2, i3);
        }
    }

    // Top cap
    const topV = segV * cols;
    for (let h = 0; h < segH; h++) {
        const a = topV + h * 2, b = topV + h * 2 + 2;
        const c = topV + h * 2 + 1, d = topV + h * 2 + 3;
        indices.push(a, b, c);
        indices.push(b, d, c);
    }

    // Bottom cap
    for (let h = 0; h < segH; h++) {
        const a = h * 2, b = h * 2 + 1;
        const c = h * 2 + 2, d = h * 2 + 3;
        indices.push(a, c, b);
        indices.push(c, d, b);
    }

    const geo = new THREE.BufferGeometry();
    geo.setAttribute('position', new THREE.Float32BufferAttribute(vertices, 3));
    geo.setAttribute('normal', new THREE.Float32BufferAttribute(normals, 3));
    geo.setIndex(indices);
    geo.computeVertexNormals();
    return geo;
}

const damGeo = createArchDam(
    5.5,    // innerRadius
    4.5,    // height
    -0.55,  // thetaStart
    1.1,    // thetaSpan (~63 degrees)
    1.5,    // baseThickness
    0.35,   // crestThickness
    24,     // horizontal segments
    10      // vertical segments
);
const damMat = new THREE.MeshStandardMaterial({
    color: 0x8899aa,
    roughness: 0.4,
    metalness: 0.1,
    flatShading: false,
});
const dam = new THREE.Mesh(damGeo, damMat);
dam.position.y = -1.2;
dam.castShadow = true;
dam.receiveShadow = true;
scene.add(dam);

// Dam crest (flat top surface)
const crestGeo = new THREE.RingGeometry(5.5, 5.85, 32, 1, -0.55, 1.1);
const crestMat = new THREE.MeshStandardMaterial({color:0x99aabb, roughness:0.3});
const crest = new THREE.Mesh(crestGeo, crestMat);
crest.rotation.x = -Math.PI / 2;
crest.position.y = 3.3;
crest.receiveShadow = true;
scene.add(crest);

// Spillway structure on crest
const spillwayGeo = new THREE.BoxGeometry(0.6, 0.3, 2.5);
const spillwayMat = new THREE.MeshStandardMaterial({color:0x778899, roughness:0.3});
const spillway = new THREE.Mesh(spillwayGeo, spillwayMat);
spillway.position.set(0, 3.55, -5.2);
spillway.castShadow = true;
scene.add(spillway);

// ===== WATER (UPSTREAM) =====
const waterGeo = new THREE.CircleGeometry(8, 48, -0.55, 1.1);
const waterMat = new THREE.MeshPhongMaterial({
    color: 0x1a5276,
    specular: 0x4488cc,
    shininess: 80,
    transparent: true,
    opacity: 0.75,
});
const water = new THREE.Mesh(waterGeo, waterMat);
water.rotation.x = -Math.PI / 2;
water.position.y = 1.6;
water.renderOrder = 1;
scene.add(water);

// River downstream
const riverGeo = new THREE.PlaneGeometry(4, 8);
const riverMat = new THREE.MeshPhongMaterial({color:0x1565c0, transparent:true, opacity:0.6});
const river = new THREE.Mesh(riverGeo, riverMat);
river.rotation.x = -Math.PI / 2;
river.position.set(0, -3.3, 5);
scene.add(river);

// ===== SENSOR VISUALIZATION GROUPS =====
const strainGroup = new THREE.Group();    // Strain: colored spheres on downstream face
const seepageGroup = new THREE.Group();   // Seepage: blue particles at foundation
const settlementGroup = new THREE.Group();// Settlement: vertical markers at base
const seismicGroup = new THREE.Group();   // Seismic: concentric rings
const heatmapGroup = new THREE.Group();   // Heat map grid on dam face
scene.add(strainGroup);
scene.add(seepageGroup);
scene.add(settlementGroup);
scene.add(seismicGroup);
scene.add(heatmapGroup);

// Heat map grid (6x4 grid on downstream face)
const heatmapDots = [];
const GRID_COLS = 8, GRID_ROWS = 5;
const innerR = 5.5, damH = 4.5, damY0 = -1.2;
const baseThick = 1.5, crestThick = 0.35, thetaStart = -0.55, thetaSpan = 1.1;

for (let r = 0; r < GRID_ROWS; r++) {
    for (let c = 0; c < GRID_COLS; c++) {
        const fracV = (r + 0.5) / GRID_ROWS;
        const fracH = (c + 0.5) / GRID_COLS;
        const y = damY0 + fracV * damH;
        const thick = baseThick - fracV * (baseThick - crestThick);
        const outerR = innerR + thick;
        const theta = thetaStart + fracH * thetaSpan;

        const dotGeo = new THREE.SphereGeometry(0.08, 8, 8);
        const dotMat = new THREE.MeshStandardMaterial({color:0x4caf50, roughness:0.3, emissive:0x000000});
        const dot = new THREE.Mesh(dotGeo, dotMat);
        dot.position.set(
            outerR * Math.sin(theta),
            y,
            -outerR * Math.cos(theta)
        );
        dot.userData = {gridR: r, gridC: c, baseColor: 0x4caf50};
        heatmapGroup.add(dot);
        heatmapDots.push(dot);
    }
}

// Strain sensor markers (9 sensors on dam face)
const strainMarkers = [];
for (let i = 0; i < 9; i++) {
    const fracH = (i + 0.5) / 9;
    const theta = thetaStart + fracH * thetaSpan;
    const geo = new THREE.SphereGeometry(0.13, 16, 16);
    const mat = new THREE.MeshStandardMaterial({color:0x4caf50, roughness:0.2, emissive:0x1a3a1a});
    const marker = new THREE.Mesh(geo, mat);
    const thick = baseThick - 0.5 * (baseThick - crestThick);
    marker.position.set(
        (innerR + thick) * Math.sin(theta),
        damY0 + 0.5 * damH,
        -(innerR + thick) * Math.cos(theta)
    );
    marker.userData = {index: i, type: 'strain'};
    strainGroup.add(marker);
    strainMarkers.push(marker);
}

// Seepage particles (blue particles at dam-foundation interface)
const seepageParticles = [];
for (let i = 0; i < 30; i++) {
    const geo = new THREE.SphereGeometry(0.05, 4, 4);
    const mat = new THREE.MeshBasicMaterial({color:0x2196f3, transparent:true, opacity:0.7});
    const particle = new THREE.Mesh(geo, mat);
    const theta = thetaStart + Math.random() * thetaSpan;
    particle.position.set(
        (innerR - 0.1) * Math.sin(theta),
        damY0 + Math.random() * 0.8,
        -(innerR - 0.1) * Math.cos(theta)
    );
    particle.userData = {
        baseY: particle.position.y,
        speed: 0.3 + Math.random() * 0.7,
        phase: Math.random() * Math.PI * 2,
        amplitude: 0.2 + Math.random() * 0.6,
    };
    seepageGroup.add(particle);
    seepageParticles.push(particle);
}

// Settlement markers (discs at foundation)
const settlementMarkers = [];
for (let i = 0; i < 6; i++) {
    const fracH = (i + 0.5) / 6;
    const theta = thetaStart + fracH * thetaSpan;
    const geo = new THREE.CylinderGeometry(0.15, 0.15, 0.15, 12);
    const mat = new THREE.MeshStandardMaterial({color:0xff9800, roughness:0.3});
    const marker = new THREE.Mesh(geo, mat);
    marker.position.set(
        (innerR + baseThick + 0.2) * Math.sin(theta),
        -3.2,
        -(innerR + baseThick + 0.2) * Math.cos(theta)
    );
    marker.userData = {index: i, baseY: marker.position.y};
    settlementGroup.add(marker);
    settlementMarkers.push(marker);
}

// Seismic rings (3 concentric rings at dam base)
const seismicRings = [];
for (let i = 0; i < 3; i++) {
    const radius = 1.0 + i * 1.2;
    const geo = new THREE.TorusGeometry(radius, 0.02, 8, 48);
    const mat = new THREE.MeshBasicMaterial({color:0xff5722, transparent:true, opacity:0});
    const ring = new THREE.Mesh(geo, mat);
    ring.rotation.x = Math.PI / 2;
    ring.position.y = -3.15;
    ring.userData = {index: i, baseOpacity: 0, targetOpacity: 0, phase: i * 1.2};
    seismicGroup.add(ring);
    seismicRings.push(ring);
}

// ===== ENVIRONMENT =====
// Grid on valley floor
const gridHelper = new THREE.PolarGridHelper(8, 32, 24, 64, 0x1a3a5c, 0x1a3a5c);
gridHelper.position.y = -3.48;
scene.add(gridHelper);

// Sky dome (simple gradient via large sphere)
const skyGeo = new THREE.SphereGeometry(30, 32, 16);
const skyMat = new THREE.ShaderMaterial({
    uniforms: {
        topColor: {value: new THREE.Color(0x0a1628)},
        bottomColor: {value: new THREE.Color(0x1a2a44)},
        offset: {value: 20},
        exponent: {value: 0.6},
    },
    vertexShader: `
        varying vec3 vWorldPosition;
        void main() {
            vec4 worldPosition = modelMatrix * vec4(position, 1.0);
            vWorldPosition = worldPosition.xyz;
            gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
        }`,
    fragmentShader: `
        uniform vec3 topColor;
        uniform vec3 bottomColor;
        uniform float offset;
        uniform float exponent;
        varying vec3 vWorldPosition;
        void main() {
            float h = normalize(vWorldPosition + offset).y;
            gl_FragColor = vec4(mix(bottomColor, topColor, max(pow(max(h, 0.0), exponent), 0.0)), 1.0);
        }`,
    side: THREE.BackSide,
});
const sky = new THREE.Mesh(skyGeo, skyMat);
scene.add(sky);

// ===== CAMERA CONTROLS =====
let isDragging = false, prevMouse = {x:0, y:0};
let spherical = {theta: Math.PI/4, phi: Math.PI/3, radius: 14};
const camCenter = new THREE.Vector3(0, 0.5, 0);

function updateCamera() {
    const sp = spherical;
    camera.position.set(
        camCenter.x + sp.radius * Math.sin(sp.phi) * Math.cos(sp.theta),
        camCenter.y + sp.radius * Math.cos(sp.phi),
        camCenter.z + sp.radius * Math.sin(sp.phi) * Math.sin(sp.theta)
    );
    camera.lookAt(camCenter);
}

renderer.domElement.addEventListener('mousedown', e => {
    isDragging = true;
    prevMouse = {x: e.clientX, y: e.clientY};
});
renderer.domElement.addEventListener('mouseup', () => { isDragging = false; });
renderer.domElement.addEventListener('mouseleave', () => { isDragging = false; });
renderer.domElement.addEventListener('mousemove', e => {
    if (!isDragging) return;
    const dx = e.clientX - prevMouse.x;
    const dy = e.clientY - prevMouse.y;
    spherical.theta -= dx * 0.005;
    spherical.phi -= dy * 0.005;
    spherical.phi = Math.max(0.2, Math.min(Math.PI/2 - 0.05, spherical.phi));
    updateCamera();
    prevMouse = {x: e.clientX, y: e.clientY};
});
renderer.domElement.addEventListener('wheel', e => {
    spherical.radius += e.deltaY * 0.02;
    spherical.radius = Math.max(4, Math.min(25, spherical.radius));
    updateCamera();
});

// Touch support
renderer.domElement.addEventListener('touchstart', e => {
    if (e.touches.length === 1) {
        isDragging = true;
        prevMouse = {x: e.touches[0].clientX, y: e.touches[0].clientY};
    }
});
renderer.domElement.addEventListener('touchmove', e => {
    if (!isDragging || e.touches.length !== 1) return;
    const dx = e.touches[0].clientX - prevMouse.x;
    const dy = e.touches[0].clientY - prevMouse.y;
    spherical.theta -= dx * 0.005;
    spherical.phi -= dy * 0.005;
    spherical.phi = Math.max(0.2, Math.min(Math.PI/2 - 0.05, spherical.phi));
    updateCamera();
    prevMouse = {x: e.touches[0].clientX, y: e.touches[0].clientY};
});
renderer.domElement.addEventListener('touchend', () => { isDragging = false; });

// ===== RAYCASTER FOR TOOLTIP =====
const raycaster = new THREE.Raycaster();
const mouse = new THREE.Vector2();
const tooltip = document.getElementById('tooltip');
renderer.domElement.addEventListener('mousemove', e => {
    mouse.x = (e.offsetX / container.clientWidth) * 2 - 1;
    mouse.y = -(e.offsetY / container.clientHeight) * 2 + 1;

    raycaster.setFromCamera(mouse, camera);
    const intersects = raycaster.intersectObjects([...strainMarkers, ...heatmapDots]);

    if (intersects.length > 0) {
        const obj = intersects[0].object;
        const ud = obj.userData;
        let text = '';
        if (ud.type === 'strain') {
            text = `应变传感器 #${ud.index + 1}`;
        } else if (ud.gridR !== undefined) {
            text = `监测网格 [${ud.gridR},${ud.gridC}]`;
        }
        tooltip.style.display = 'block';
        tooltip.style.left = (e.offsetX + 15) + 'px';
        tooltip.style.top = (e.offsetY - 10) + 'px';
        tooltip.textContent = text;
    } else {
        tooltip.style.display = 'none';
    }
});

updateCamera();

// ===== FPS COUNTER =====
let frameCount = 0, lastFpsTime = performance.now();
const fpsEl = document.getElementById('fps-display');

// ===== ANIMATION LOOP =====
const clock = new THREE.Clock();

function animate() {
    requestAnimationFrame(animate);
    const dt = Math.min(clock.getDelta(), 0.1);
    const t = performance.now() * 0.001;

    // Water animation
    water.position.y = 1.6 + Math.sin(t * 0.3) * 0.15;

    // Seepage particles animation
    seepageParticles.forEach(p => {
        const ud = p.userData;
        p.position.y = ud.baseY + Math.sin(t * ud.speed + ud.phase) * ud.amplitude;
        p.material.opacity = 0.3 + Math.abs(Math.sin(t * ud.speed + ud.phase)) * 0.6;
    });

    // Seismic ring pulse
    seismicRings.forEach(ring => {
        const ud = ring.userData;
        if (ud.targetOpacity > 0.01) {
            const s = 1 + Math.sin(t * 3 + ud.phase) * 0.3;
            ring.scale.setScalar(s);
            ring.material.opacity += (ud.targetOpacity - ring.material.opacity) * 3 * dt;
        } else {
            ring.material.opacity += (0 - ring.material.opacity) * 2 * dt;
            ring.scale.setScalar(1);
        }
    });

    // FPS
    frameCount++;
    if (t - lastFpsTime >= 1) {
        fpsEl.textContent = 'FPS: ' + Math.round(frameCount / (t - lastFpsTime));
        frameCount = 0;
        lastFpsTime = t;
    }

    renderer.render(scene, camera);
}
animate();

// ===== SOCKET.IO DATA =====
const socket = io();

function updateHazardPanel(data) {
    const hazards = [
        {key: 'seepage', name: '渗流灾害', idx: 0},
        {key: 'deformation', name: '变形灾害', idx: 1},
        {key: 'settlement', name: '沉降灾害', idx: 2},
        {key: 'seismic', name: '地震灾害', idx: 3},
    ];

    const container = document.getElementById('hazard-panel');
    const levels = data.per_hazard || {};
    const colors = {'正常':'#4caf50','轻微渗漏':'#ff9800','集中渗漏':'#f44336','管涌':'#b71c1c',
                    '微变形':'#4caf50','显著变形':'#ff9800','结构失稳':'#b71c1c',
                    '轻微沉降':'#4caf50','不均匀沉降':'#ff9800','过量沉降':'#b71c1c',
                    '微震':'#4caf50','中震':'#ff9800','强震':'#b71c1c'};

    container.innerHTML = hazards.map((h, i) => {
        const info = levels[h.key] || {risk_level: '正常'};
        const color = colors[info.risk_level] || '#4caf50';
        return `<div class="hazard-indicator">
            <div class="hazard-dot" style="background:${color}"></div>
            <div class="hazard-info"><div class="hazard-name">${h.name}</div>
            <div class="hazard-status" style="color:${color}">${info.risk_level}</div></div></div>`;
    }).join('');
}

socket.on('sensor_update', function(data) {
    // Risk badge
    const badge = document.getElementById('risk-badge');
    badge.textContent = data.risk_level || '安全';
    badge.className = 'risk-badge risk-' + (data.risk_level || '安全');
    document.getElementById('risk-subtitle').textContent =
        data.risk_level === '安全' ? '各监测指标均在正常范围' :
        data.risk_level === '注意' ? '部分指标出现轻微异常，建议加强监测' :
        data.risk_level === '警告' ? '存在中等风险，需尽快安排现场检查' :
        '工程处于高风险状态，应立即启动应急预案';

    // Hazard panel
    updateHazardPanel(data);

    // Metrics
    if (data.strain && data.strain.length > 0) {
        document.getElementById('val-strain').textContent =
            Math.max(...data.strain.map(s => s.value || 0)).toFixed(1);
    }
    if (data.seepage && data.seepage.length > 0) {
        document.getElementById('val-seepage').textContent =
            data.seepage[0].value?.toFixed(1) || '--';
    }
    if (data.settlement && data.settlement.length > 0) {
        document.getElementById('val-settlement').textContent =
            data.settlement[0].value?.toFixed(2) || '--';
    }
    if (data.seismic && data.seismic.length > 0) {
        document.getElementById('val-seismic').textContent =
            data.seismic[0].value?.toFixed(2) || '--';
    }

    // Sensor count
    const totalSensors = (data.strain?.length || 0) + (data.seepage?.length || 0) +
        (data.displacement?.length || 0) + (data.settlement?.length || 0) +
        (data.seismic?.length || 0);
    document.getElementById('sensor-count').textContent = '传感器: ' + totalSensors;

    // Strain markers color
    if (data.strain) {
        strainMarkers.forEach((m, i) => {
            if (i < data.strain.length) {
                const val = data.strain[i].value;
                let color, emissive;
                if (val > 130) { color = 0xf44336; emissive = 0x3d0000; }
                else if (val > 120) { color = 0xff9800; emissive = 0x3d1a00; }
                else if (val > 110) { color = 0xffeb3b; emissive = 0x1a1a00; }
                else { color = 0x4caf50; emissive = 0x0a1a0a; }
                m.material.color.setHex(color);
                m.material.emissive.setHex(emissive);
            }
        });
    }

    // Heat map grid
    if (data.heatmap_grid && data.heatmap_grid.length > 0) {
        heatmapDots.forEach((dot, i) => {
            if (i < data.heatmap_grid.length) {
                const val = data.heatmap_grid[i];
                let color;
                if (val > 130) color = 0xf44336;
                else if (val > 120) color = 0xff9800;
                else if (val > 110) color = 0xffeb3b;
                else color = 0x4caf50;
                dot.material.color.setHex(color);
                dot.material.emissive.setHex(color);
                dot.material.emissiveIntensity = val > 110 ? 0.4 : 0.05;
            }
        });
    }

    // Dam body color
    const riskColors = {'安全':0x8899aa,'注意':0xaa9944,'警告':0xcc6644,'危险':0xcc3333};
    dam.material.color.setHex(riskColors[data.risk_level] || 0x8899aa);

    // Seepage intensity (particle opacity + activity)
    const seepageActive = data.seepage && data.seepage.length > 0 && data.seepage[0].value > 45;
    seepageParticles.forEach(p => {
        p.material.opacity = seepageActive ? 0.5 + Math.random() * 0.4 : 0.15 + Math.random() * 0.15;
        p.visible = seepageActive || Math.random() > 0.7;
    });

    // Settlement markers
    if (data.settlement && data.settlement.length > 0) {
        const settVal = data.settlement[0].value || 0;
        settlementMarkers.forEach((m, i) => {
            m.position.y = m.userData.baseY - settVal * 0.3 * (i + 1) / settlementMarkers.length;
            m.material.color.setHex(settVal > 1.5 ? 0xf44336 : settVal > 0.8 ? 0xff9800 : 0xff9800);
        });
    }

    // Seismic rings
    const seismicActive = data.seismic && data.seismic.length > 0 && data.seismic[0].value > 5;
    seismicRings.forEach(ring => {
        ring.userData.targetOpacity = seismicActive ? 0.6 : 0;
    });

    // Alerts
    if (data.alerts && data.alerts.length > 0) {
        document.getElementById('alerts-container').innerHTML = data.alerts.slice(-10).map(a =>
            `<div class="alert-item">[${a.time}] ${a.message}</div>`
        ).join('');
    }
});

window.addEventListener('resize', () => {
    camera.aspect = container.clientWidth / container.clientHeight;
    camera.updateProjectionMatrix();
    renderer.setSize(container.clientWidth, container.clientHeight);
});
</script>
</body>
</html>'''


@app.route('/')
def index():
    return render_template_string(TEMPLATE)


@app.route('/api/status')
def api_status():
    return jsonify(sensor_state)


@socketio.on('connect')
def handle_connect():
    emit('sensor_update', sensor_state)


def generate_simulated_data():
    t = time.time()
    rng = np.random.RandomState(int(t / 3))

    # Strain data with spatial pattern
    strain = []
    for i in range(9):
        base = 100 + 12 * np.sin(i * 0.65 + t * 0.08)
        damage = 0
        if 3 <= i <= 5:
            damage = 15 * np.sin(t * 0.03) ** 2
        noise = rng.randn() * 2.5
        strain.append({'sensor_id': f'STR-{i+1:02d}', 'value': base + damage + noise, 'unit': 'με'})

    # Seepage
    seepage_base = 45 + 8 * np.sin(t * 0.04)
    seepage_val = seepage_base + rng.randn() * 3
    seepage = [
        {'sensor_id': 'SEP-01', 'value': seepage_val, 'unit': 'kPa'},
        {'sensor_id': 'SEP-02', 'value': seepage_val * 0.85 + rng.randn() * 2, 'unit': 'kPa'},
    ]

    # Displacement
    disp = [
        {'sensor_id': 'DSP-01', 'value': 0.3 + 0.15 * np.sin(t * 0.06) + rng.randn() * 0.05, 'unit': 'mm'},
    ]

    # Settlement
    settlement = [
        {'sensor_id': 'STL-01', 'value': 0.4 + 0.2 * np.sin(t * 0.02) + rng.randn() * 0.08, 'unit': 'mm/月'},
    ]

    # Seismic
    seismic_active = (np.sin(t * 0.01) > 0.85)
    seismic = [
        {'sensor_id': 'ACC-01', 'value': (rng.randn() * 12 + 25) if seismic_active else (rng.randn() * 1.5 + 2), 'unit': 'gal'},
    ]

    # Heat map grid data
    heatmap_grid = []
    for r in range(5):
        for c in range(8):
            h_val = 100 + 5 * np.sin(c * 0.5 + t * 0.05) + 8 * np.sin(r * 0.7) + rng.randn() * 1.5
            heatmap_grid.append(float(h_val))

    # Risk level
    max_strain = max(s['value'] for s in strain)
    max_seep = max(s['value'] for s in seepage)
    max_sett = max(s['value'] for s in settlement)

    if max_strain > 130 or max_seep > 65:
        risk = '危险'
    elif max_strain > 120 or max_seep > 55:
        risk = '警告'
    elif max_strain > 110:
        risk = '注意'
    else:
        risk = '安全'

    # Per-hazard risk levels
    per_hazard = {
        'seepage': {'risk_level': '正常' if max_seep < 48 else ('轻微渗漏' if max_seep < 55 else ('集中渗漏' if max_seep < 65 else '管涌')), 'risk_score': max(0, (max_seep - 40) / 25)},
        'deformation': {'risk_level': '正常' if max_strain < 105 else ('微变形' if max_strain < 115 else ('显著变形' if max_strain < 130 else '结构失稳')), 'risk_score': max(0, (max_strain - 95) / 45)},
        'settlement': {'risk_level': '正常' if max_sett < 0.5 else ('轻微沉降' if max_sett < 1.0 else ('不均匀沉降' if max_sett < 1.5 else '过量沉降')), 'risk_score': max(0, max_sett / 3)},
        'seismic': {'risk_level': '微震' if seismic_active else '正常', 'risk_score': 0.1 if seismic_active else 0},
    }

    alerts = sensor_state.get('alerts', [])
    if risk in ('警告', '危险'):
        last = alerts[-1] if alerts else None
        if not last or (t - last.get('_t', 0) > 8):
            alerts.append({
                'time': time.strftime('%H:%M:%S'),
                'risk': risk,
                'message': f'应变超阈值: max={max_strain:.1f}με' if max_strain > 120 else f'渗流超阈值: max={max_seep:.1f}kPa',
                '_t': t,
            })

    return {
        'strain': strain, 'seepage': seepage, 'displacement': disp,
        'settlement': settlement, 'seismic': seismic,
        'heatmap_grid': heatmap_grid,
        'risk_level': risk, 'per_hazard': per_hazard,
        'alerts': alerts[-20:],
        'timestamp': t,
    }


def data_generator():
    while True:
        socketio.sleep(2)
        data = generate_simulated_data()
        global sensor_state
        sensor_state.update(data)
        socketio.emit('sensor_update', data)


@socketio.on('start_stream')
def handle_start_stream():
    emit('sensor_update', sensor_state)


def run_server(host='0.0.0.0', port=5000, debug=False):
    socketio.start_background_task(data_generator)
    print(f"数字孪生服务启动: http://{host}:{port}")
    socketio.run(app, host=host, port=port, debug=debug, allow_unsafe_werkzeug=True)


if __name__ == '__main__':
    run_server(debug=True)
