"""
数字孪生3D可视化 — WebSocket实时数据推送 + Three.js 3D大坝模型
后端 Flask-SocketIO 服务
"""
import json
import time
import numpy as np
from flask import Flask, render_template_string, jsonify, request
from flask_socketio import SocketIO, emit
import threading

app = Flask(__name__)
app.config['SECRET_KEY'] = 'badao-weiyi-2.0'
socketio = SocketIO(app, cors_allowed_origins="*")

# Global state
sensor_state = {
    'strain': [],
    'seepage': [],
    'displacement': [],
    'settlement': [],
    'seismic': [],
    'alerts': [],
    'risk_level': 'normal',
    'timestamp': time.time(),
}

# HTML template with Three.js 3D dam visualization
TEMPLATE = '''<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>坝道微医 2.0 — 数字孪生</title>
<style>
* { margin:0; padding:0; box-sizing:border-box; }
body { font-family: "Microsoft YaHei", sans-serif; background:#0a1628; color:#e0e0e0; overflow:hidden; }
#container { display:flex; height:100vh; }
#viewer { flex:1; position:relative; }
#panel { width:360px; background:#0d1f3c; padding:20px; overflow-y:auto; border-left:2px solid #1a3a5c; }
h2 { color:#4fc3f7; margin-bottom:16px; font-size:18px; border-bottom:1px solid #1a3a5c; padding-bottom:8px; }
.metric { background:#122647; border-radius:8px; padding:12px; margin-bottom:12px; }
.metric .label { font-size:12px; color:#78909c; text-transform:uppercase; }
.metric .value { font-size:24px; font-weight:bold; color:#4fc3f7; }
.metric .unit { font-size:12px; color:#78909c; }
.risk-safe { color:#4caf50 !important; }
.risk-warning { color:#ff9800 !important; }
.risk-danger { color:#f44336 !important; }
.alert-item { background:#1a0a0a; border-left:3px solid #f44336; padding:8px 12px; margin-bottom:8px; border-radius:4px; font-size:13px; }
#legend { position:absolute; bottom:20px; left:20px; background:rgba(0,0,0,0.7); padding:12px; border-radius:8px; font-size:12px; }
.legend-item { display:flex; align-items:center; margin:4px 0; }
.legend-color { width:16px; height:16px; border-radius:3px; margin-right:8px; }
</style>
</head>
<body>
<div id="container">
  <div id="viewer">
    <div id="legend">
      <div class="legend-item"><div class="legend-color" style="background:#4caf50"></div> 正常 (安全)</div>
      <div class="legend-item"><div class="legend-color" style="background:#ffeb3b"></div> 注意 (轻微异常)</div>
      <div class="legend-item"><div class="legend-color" style="background:#ff9800"></div> 警告 (中度损伤)</div>
      <div class="legend-item"><div class="legend-color" style="background:#f44336"></div> 危险 (严重损伤)</div>
      <div class="legend-item"><div class="legend-color" style="background:#9c27b0"></div> 传感器测点</div>
    </div>
  </div>
  <div id="panel">
    <h2>实时监测指标</h2>
    <div class="metric"><div class="label">整体风险等级</div><div class="value" id="risk-level">--</div></div>
    <div class="metric"><div class="label">最大应变 (με)</div><div class="value" id="max-strain">--</div></div>
    <div class="metric"><div class="label">渗流压力 (kPa)</div><div class="value" id="seepage-pressure">--</div></div>
    <div class="metric"><div class="label">沉降速率 (mm/月)</div><div class="value" id="settlement-rate">--</div></div>
    <div class="metric"><div class="label">振动加速度 (gal)</div><div class="value" id="seismic-accel">--</div></div>
    <h2>告警信息</h2>
    <div id="alerts"><div style="color:#78909c;font-size:13px;">系统运行正常，无告警</div></div>
  </div>
</div>

<script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/socket.io-client@4/dist/socket.io.min.js"></script>
<script>
// Three.js scene setup
const container = document.getElementById('viewer');
const scene = new THREE.Scene();
scene.background = new THREE.Color(0x0a1628);
const camera = new THREE.PerspectiveCamera(60, container.clientWidth / container.clientHeight, 0.1, 1000);
camera.position.set(8, 6, 10);
camera.lookAt(0, 0, 0);
const renderer = new THREE.WebGLRenderer({antialias:true});
renderer.setSize(container.clientWidth, container.clientHeight);
renderer.shadowMap.enabled = true;
container.appendChild(renderer.domElement);

// Lighting
const ambientLight = new THREE.AmbientLight(0x404060);
scene.add(ambientLight);
const dirLight = new THREE.DirectionalLight(0xffffff, 0.8);
dirLight.position.set(10, 15, 5);
dirLight.castShadow = true;
scene.add(dirLight);

// Dam body (simplified trapezoidal shape)
const damGeometry = new THREE.BoxGeometry(6, 3, 2);
const damMaterial = new THREE.MeshPhongMaterial({color:0x607d8b, shininess:30});
const dam = new THREE.Mesh(damGeometry, damMaterial);
dam.position.y = 0;
dam.castShadow = true;
dam.receiveShadow = true;
scene.add(dam);

// Water surface
const waterGeometry = new THREE.PlaneGeometry(6, 4);
const waterMaterial = new THREE.MeshPhongMaterial({color:0x1e88e5, transparent:true, opacity:0.6});
const water = new THREE.Mesh(waterGeometry, waterMaterial);
water.rotation.x = -Math.PI / 2;
water.position.set(0, -1.5, 2.5);
scene.add(water);

// Ground plane
const groundGeometry = new THREE.PlaneGeometry(20, 20);
const groundMaterial = new THREE.MeshPhongMaterial({color:0x2e7d32});
const ground = new THREE.Mesh(groundGeometry, groundMaterial);
ground.rotation.x = -Math.PI / 2;
ground.position.y = -3;
scene.add(ground);

// Sensor points (sphere markers)
const sensorPoints = [];
const SENSOR_COUNT = 9;
for (let i = 0; i < SENSOR_COUNT; i++) {
    const geometry = new THREE.SphereGeometry(0.12, 16, 16);
    const material = new THREE.MeshPhongMaterial({color:0x9c27b0, emissive:0x4a0072});
    const sphere = new THREE.Mesh(geometry, material);
    const angle = (i / SENSOR_COUNT) * Math.PI * 2;
    sphere.position.set(Math.cos(angle)*1.5, Math.sin(angle*2)*0.8, Math.sin(angle)*1.2);
    sphere.userData = {index:i, baseColor:0x4caf50};
    scene.add(sphere);
    sensorPoints.push(sphere);
}

// Grid helper
const gridHelper = new THREE.GridHelper(12, 12, 0x1a3a5c, 0x0d1f3c);
gridHelper.position.y = -2.99;
scene.add(gridHelper);

// Controls
let isDragging = false;
let prevMouse = {x:0, y:0};
renderer.domElement.addEventListener('mousedown', e => { isDragging=true; prevMouse={x:e.clientX, y:e.clientY}; });
renderer.domElement.addEventListener('mouseup', () => { isDragging=false; });
renderer.domElement.addEventListener('mousemove', e => {
    if (!isDragging) return;
    const dx = e.clientX - prevMouse.x;
    const dy = e.clientY - prevMouse.y;
    camera.position.x -= dx * 0.02;
    camera.position.y += dy * 0.02;
    camera.lookAt(0, 0, 0);
    prevMouse = {x:e.clientX, y:e.clientY};
});
renderer.domElement.addEventListener('wheel', e => {
    camera.position.z += e.deltaY * 0.01;
    camera.position.z = Math.max(3, Math.min(20, camera.position.z));
});

// Animation loop
function animate() {
    requestAnimationFrame(animate);
    water.position.z += 0.002;
    if (water.position.z > 3) water.position.z = 2;
    renderer.render(scene, camera);
}
animate();

// Socket.IO data streaming
const socket = io();
socket.on('sensor_update', function(data) {
    // Update panel values
    document.getElementById('risk-level').textContent = data.risk_level || '--';
    const riskEl = document.getElementById('risk-level');
    riskEl.className = 'value';
    if (data.risk_level === '安全') riskEl.classList.add('risk-safe');
    else if (data.risk_level === '危险') riskEl.classList.add('risk-danger');
    else riskEl.classList.add('risk-warning');

    if (data.strain && data.strain.length > 0) {
        const maxStrain = Math.max(...data.strain.map(s => s.value || 0));
        document.getElementById('max-strain').textContent = maxStrain.toFixed(1);
    }
    if (data.seepage && data.seepage.length > 0) {
        document.getElementById('seepage-pressure').textContent = data.seepage[0].value?.toFixed(1) || '--';
    }
    if (data.settlement && data.settlement.length > 0) {
        document.getElementById('settlement-rate').textContent = data.settlement[0].value?.toFixed(2) || '--';
    }

    // Update sensor colors
    if (data.alerts && data.alerts.length > 0) {
        document.getElementById('alerts').innerHTML = data.alerts.map(a =>
            '<div class="alert-item">[' + a.time + '] ' + a.message + '</div>'
        ).join('');
    }

    // Update dam color based on risk
    const riskColors = {'安全':0x607d8b, '注意':0xffeb3b, '警告':0xff9800, '危险':0xf44336};
    dam.material.color.setHex(riskColors[data.risk_level] || 0x607d8b);

    // Update sensor markers
    if (data.sensor_colors) {
        sensorPoints.forEach((sp, i) => {
            if (i < data.sensor_colors.length) {
                sp.material.color.setHex(data.sensor_colors[i]);
            }
        });
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
    """Simulate real-time sensor data for demo purposes."""
    t = time.time()
    rng = np.random.RandomState(int(t / 5))

    # Simulate strain data (9 sensors)
    strain = []
    for i in range(9):
        base = 100 + 10 * np.sin(i * 0.7 + t * 0.1)
        noise = rng.randn() * 3
        strain.append({'sensor_id': f'S{i+1}', 'value': base + noise, 'unit': 'με'})

    # Simulate seepage
    seepage_base = 50 + 5 * np.sin(t * 0.05)
    seepage = [{'sensor_id': 'P1', 'value': seepage_base + rng.randn() * 2, 'unit': 'kPa'}]

    # Simulate settlement
    settlement = [{'sensor_id': 'D1', 'value': 0.5 + 0.1 * np.sin(t * 0.02) + rng.randn() * 0.05, 'unit': 'mm/月'}]

    # Determine risk level
    max_strain = max(s['value'] for s in strain)
    if max_strain > 130:
        risk = '危险'
    elif max_strain > 120:
        risk = '警告'
    elif max_strain > 110:
        risk = '注意'
    else:
        risk = '安全'

    # Sensor colors for 3D
    risk_to_color = {'安全': 0x4caf50, '注意': 0xffeb3b, '警告': 0xff9800, '危险': 0xf44336}
    sensor_colors = []
    for s in strain:
        val = s['value']
        if val > 130:
            sensor_colors.append(0xf44336)
        elif val > 120:
            sensor_colors.append(0xff9800)
        elif val > 110:
            sensor_colors.append(0xffeb3b)
        else:
            sensor_colors.append(0x4caf50)

    alerts = sensor_state.get('alerts', [])
    if risk in ('警告', '危险') and (not alerts or alerts[-1].get('risk') != risk):
        alerts.append({
            'time': time.strftime('%H:%M:%S'),
            'risk': risk,
            'message': f'应变超阈值预警: max={max_strain:.1f}με',
        })
    # Keep last 20 alerts
    alerts = alerts[-20:]

    return {
        'strain': strain,
        'seepage': seepage,
        'displacement': [],
        'settlement': settlement,
        'seismic': [],
        'risk_level': risk,
        'sensor_colors': sensor_colors,
        'alerts': alerts,
        'timestamp': t,
    }


def data_generator():
    """Background thread that pushes sensor updates every 2 seconds."""
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
    """Start the digital twin server."""
    socketio.start_background_task(data_generator)
    print(f"数字孪生服务启动: http://{host}:{port}")
    socketio.run(app, host=host, port=port, debug=debug, allow_unsafe_werkzeug=True)


if __name__ == '__main__':
    run_server(debug=True)
