"""Self-contained scientific 3D comparison of checked display meshes."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def run(directory, surface, output, adaptive=None):
    import plotly
    from plotly.offline import get_plotlyjs
    report = json.loads((directory/'verification.json').read_text())
    if report.get('passed') is not True:
        raise ValueError('Require completed display checks.')
    hashes = {}
    for section in ['input_sha256', 'source_sha256', 'output_sha256']:
        for name, expected in report[section].items():
            if hashlib.sha256(Path(name).read_bytes()).hexdigest() != expected:
                raise ValueError(f'Checked display input changed: {name}')
    hashes[str(directory/'verification.json')] = hashlib.sha256((directory/'verification.json').read_bytes()).hexdigest()
    hashes[str(surface)] = hashlib.sha256(surface.read_bytes()).hexdigest()
    with np.load(surface) as source:
        positions = source['positions'].tolist(); sources = source['source_positions'].tolist()
        time = float(source['time'])
    names = dict(ball_saved='Guarded ball pivoting', imls_saved='Local implicit: saved normals',
                 imls_fitted='Local implicit: display-fit normals', poisson_supported='Poisson + support trimming')
    eligibility = json.loads((directory/'eligibility.json').read_text())
    eligible = {r['method']: r['conditional_display_eligible'] for r in eligibility}
    meshes = {}; labels = None
    for method, name in names.items():
        path = directory/f'bubble_{method}.npz'
        with np.load(path) as data:
            np.testing.assert_array_equal(data['original_positions'], np.array(positions))
            if labels is None: labels = data['original_labels'].tolist()
            np.testing.assert_array_equal(data['original_labels'], labels)
            meshes[method] = dict(name=name, vertices=data['vertices'].tolist(), faces=data['faces'].tolist(),
                                  groups=data['vertex_groups'].tolist(), eligible=eligible[method])
    if adaptive is not None:
        adaptive_report = json.loads((adaptive/'verification.json').read_text())
        if adaptive_report.get('passed') is not True: raise ValueError('Require checked adaptive results.')
        for section in ['input_sha256', 'source_sha256', 'output_sha256']:
            for path, expected in adaptive_report[section].items():
                if hashlib.sha256(Path(path).read_bytes()).hexdigest() != expected: raise ValueError('Changed adaptive input.')
        adaptive_path = adaptive/'bubble_adaptive_imls.npz'
        hashes[str(adaptive_path)] = hashlib.sha256(adaptive_path.read_bytes()).hexdigest()
        hashes[str(adaptive/'verification.json')] = hashlib.sha256((adaptive/'verification.json').read_bytes()).hexdigest()
        with np.load(adaptive_path) as data:
            np.testing.assert_array_equal(data['original_positions'], positions)
            np.testing.assert_array_equal(data['original_labels'], labels)
            eligible['adaptive_imls'] = adaptive_report['conditional_display_eligible']
            meshes['adaptive_imls'] = dict(name='Adaptive local implicit: saved normals', vertices=data['vertices'].tolist(),
                                          faces=data['faces'].tolist(), groups=data['vertex_groups'].tolist(),
                                          eligible=eligible['adaptive_imls'])
    payload = json.dumps(dict(positions=positions, labels=labels, sources=sources, meshes=meshes, time=time), separators=(',', ':'))
    default = next((method for method in ['adaptive_imls', 'imls_saved', 'imls_fitted', 'poisson_supported', 'ball_saved'] if eligible.get(method)), 'points')
    html = '''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Shock-surface display comparison</title><style>
body{font:16px system-ui,sans-serif;color:#1c3342;background:#f4f7fa;margin:0}main{max-width:1300px;margin:auto;padding:24px}
h1{font-size:26px;margin:0 0 8px}p{max-width:1050px;line-height:1.5}.controls{display:flex;gap:24px;flex-wrap:wrap;align-items:center;background:white;padding:16px;border-radius:8px}
select{padding:8px;font:inherit}#view{height:690px;background:white;margin-top:12px}#status{font-weight:600}label{display:flex;align-items:center;gap:8px}</style>
<main><h1>Three-source shock surfaces · t = TIME</h1><p>Rotate and zoom to compare the same detected scene. Meshes are display approximations; colors identify supplied geometric groups.</p>
<div class="controls"><label>Representation <select id="method"><option value="points">Original samples</option></select></label>
<label><input id="cut" type="checkbox"> Cutaway: z ≤ 0.5</label><label><input id="overlay" type="checkbox"> Overlay original samples</label></div>
<div id="view"></div><p id="status"></p><p>Black crosses mark wind sources. New implicit/Poisson vertices are separate from the physical samples. The cutaway changes the display only.
The known two-sheet ambiguity remains unresolved when distinct sheets share a group. None of these pictures establishes physical ancestry or changes Mach numbers, heating, areas, or tracking.</p></main>
<script>PLOTLY</script><script>
const data=PAYLOAD, select=document.querySelector('#method'), view=document.querySelector('#view');
const colors=['#297d9b','#c75b39','#8a5bb0','#4a9676','#dbad3f','#626e88'];
for(const [key,value] of Object.entries(data.meshes)){const o=document.createElement('option');o.value=key;o.textContent=value.name;select.appendChild(o)}
select.value='DEFAULT';let camera={eye:{x:1.4,y:-1.6,z:1.05}},ready=false;
const xyz=points=>({x:points.map(p=>p[0]),y:points.map(p=>p[1]),z:points.map(p=>p[2])});
function draw(){const method=select.value,cut=document.querySelector('#cut').checked,overlay=document.querySelector('#overlay').checked;
let traces=[],pointRows=data.positions.map((point,i)=>({point,group:data.labels[i]})).filter(row=>!cut||row.point[2]<=.5),points=pointRows.map(row=>row.point);
if(method!=='points'){const mesh=data.meshes[method],faces=mesh.faces.filter(f=>!cut||f.every(i=>mesh.vertices[i][2]<=.5));
traces.push({...xyz(mesh.vertices),type:'mesh3d',i:faces.map(f=>f[0]),j:faces.map(f=>f[1]),k:faces.map(f=>f[2]),
facecolor:faces.map(f=>colors[mesh.groups[f[0]]%colors.length]),flatshading:false,lighting:{ambient:.5,diffuse:.7,specular:.15,roughness:.8},
lightposition:{x:100,y:-150,z:200},hoverinfo:'skip',name:mesh.name,showlegend:false});
document.querySelector('#status').textContent=mesh.name+' · '+faces.length.toLocaleString()+' displayed triangles. '+(mesh.eligible?'Passed the conditional display eligibility checks.':'Did not pass every conditional display eligibility check.');
}else document.querySelector('#status').textContent=points.length.toLocaleString()+' original samples. No reconstructed connectivity.';
if(method==='points'||overlay)traces.push({...xyz(points),type:'scatter3d',mode:'markers',marker:{size:method==='points'?2:1.4,color:method==='points'?pointRows.map(row=>colors[row.group%colors.length]):'#263c49',opacity:method==='points'?1:.5},name:'Original samples',hoverinfo:'skip'});
traces.push({...xyz(data.sources),type:'scatter3d',mode:'markers+text',text:data.sources.map((_,i)=>'S'+(i+1)),textposition:'top center',marker:{size:5,color:'black',symbol:'cross'},name:'Wind sources'});
Plotly.react(view,traces,{margin:{l:0,r:0,b:0,t:10},paper_bgcolor:'white',showlegend:false,uirevision:'same-camera',
scene:{aspectmode:'data',camera,xaxis:{title:'x',range:[.08,.92]},yaxis:{title:'y',range:[.08,.92]},zaxis:{title:'z',range:[.08,.92]}}},{responsive:true,displaylogo:false}).then(()=>{if(!ready){view.on('plotly_relayout',e=>{if(e['scene.camera'])camera=e['scene.camera']});ready=true}});
}
select.addEventListener('change',draw);document.querySelector('#cut').addEventListener('change',draw);document.querySelector('#overlay').addEventListener('change',draw);draw();
</script></html>'''
    html = html.replace('TIME', f'{time:.2f}').replace('PAYLOAD', payload).replace('DEFAULT', default).replace('PLOTLY', get_plotlyjs())
    if output.exists():
        raise ValueError('Use a fresh viewer path.')
    output.write_text(html)
    for path, expected in hashes.items():
        if hashlib.sha256(Path(path).read_bytes()).hexdigest() != expected:
            raise ValueError('Input changed while exporting.')
    verification = dict(passed=True, default_method=default, plotly_version=plotly.__version__,
                        javascript_embedded=True, external_libraries_requested=False,
                        input_sha256=hashes, output_sha256=hashlib.sha256(output.read_bytes()).hexdigest(),
                        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    output.with_suffix('.verification.json').write_text(json.dumps(verification, indent=2)+'\n')
    print(json.dumps(dict(default_method=default, output=str(output), bytes=output.stat().st_size), indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory', type=Path, required=True)
    parser.add_argument('--surface', type=Path, default=Path('outputs/three_bubbles_iwr_compgpu7_n064/online/surfaces_0016.npz'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--adaptive', type=Path)
    args = parser.parse_args()
    run(args.directory, args.surface, args.output, args.adaptive)
