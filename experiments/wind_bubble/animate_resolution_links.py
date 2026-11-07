"""Animate saved event-local temporal candidates, keeping ambiguity visible."""
from __future__ import annotations
import argparse
import gzip
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import numpy as np
from PIL import Image,ImageDraw,ImageFont
from experiments.wind_bubble.plot_resolution_links import choose_record,render


def status_counts(result):
    return dict(accepted_links=sum(x['accepted'] for x in result['links']),
                redundant_links=sum(x['accepted'] and x['redundant_support'] for x in result['links']),
                ambiguous_parents=sum(x['status']=='ambiguous_target' for x in result['parents']),
                insufficient_parents=sum(x['status']=='insufficient_parent_support' for x in result['parents']),
                unsupported_children=sum(not x['accepted'] for x in result['children']))


def run(directory,output,fps=2):
    if not isinstance(fps,int) or not 1<=fps<=10:raise ValueError('Require integer FPS from 1 to 10.')
    meta=json.loads((directory/'metadata.json').read_text())
    if not json.loads((directory/'readback.json').read_text())['passed']:raise ValueError('Require checked temporal results.')
    hashes=dict(meta['input_sha256'])
    for name in ['metadata.json','readback.json','support_records.json.gz']:
        path=directory/name;hashes[str(path)]=hashlib.sha256(path.read_bytes()).hexdigest()
    for p,h in hashes.items():
        if hashlib.sha256(Path(p).read_bytes()).hexdigest()!=h:raise ValueError(f'Changed input: {p}')
    with gzip.open(directory/'support_records.json.gz','rt') as f:records=json.load(f)
    root=Path(meta['target_directory']);run_meta=json.loads((root/'metadata.json').read_text())
    if run_meta['num_cells']!=128:raise ValueError('This controlled animation requires the checked 128³ target.')
    catalogs=[];groups=[]
    for event,time in enumerate(run_meta['actual_times']):
        with np.load(root/'online'/f'surfaces_{event:04d}.npz') as s:catalogs.append({k:s[k].copy() for k in s.files})
        with np.load(Path(meta['comparison_directory'])/f'groups_n128_event{event:04d}.npz') as g:groups.append({k:g[k].copy() for k in g.files})
        assert abs(float(catalogs[-1]['time'])-time)<1e-7
    output.mkdir(parents=True,exist_ok=False);audits=[];verification=[]
    duration=1000//fps
    # This is a postprocessing slideshow of static inspection views. No interpolation
    # between events, and no propagation of group colors or persistent identities.
    for mode in ['native_cells','native_cells_scaled_count']:
        frames=[];mode_audit=[]
        with TemporaryDirectory(prefix='shock-link-frames-') as temporary:
            for event,(a,b) in enumerate(zip(catalogs,catalogs[1:])):
                record=choose_record(records,128,mode,event,2.5)
                path=Path(temporary)/f'frame_{event:04d}.png'
                key=f'{mode}_a60_labels'
                audit=render(path,record,a,b,groups[event][key],groups[event+1][key])
                counts=status_counts(record['result']);audit['status_counts']=counts;audit['frame_index']=event
                with Image.open(path) as image:
                    small=image.convert('RGB').resize((1700,900),Image.Resampling.LANCZOS)
                canvas=Image.new('RGB',(1700,995),'white');canvas.paste(small,(0,0));draw=ImageDraw.Draw(canvas)
                font=ImageFont.load_default(size=19)
                draw.line((35,910,1665,910),fill='#cccccc',width=1)
                text=(f'Interval {event+1:02d}/{len(catalogs)-1} | Accepted links: {counts["accepted_links"]} '
                      f'(redundant: {counts["redundant_links"]}) | Ambiguous parents: {counts["ambiguous_parents"]}')
                draw.text((35,922),text,fill='#222222',font=font)
                draw.text((35,954),f'Parents with insufficient support: {counts["insufficient_parents"]} | Children without accepted support: {counts["unsupported_children"]} | No gap bridging or physical IDs',fill='#222222',font=font)
                frames.append(canvas);mode_audit.append(audit)
                if event in [9,12,13,14,15]:canvas.save(output/f'{mode}_interval_{event:04d}.png')
                print(f'{mode}: interval {event:02d} -> {event+1:02d}',flush=True)
        gif=output/f'{mode}_associations.gif'
        frames[0].save(gif,save_all=True,append_images=frames[1:],duration=duration,loop=0,disposal=2,optimize=False)
        with Image.open(gif) as decoded:
            assert decoded.n_frames==len(frames)
            durations=[]
            for i in range(decoded.n_frames):
                decoded.seek(i);decoded.load();assert decoded.size==(1700,995);durations.append(decoded.info['duration'])
                # Allow palette quantization; require the changing interval header/footer
                # to remain closer to its own rendered frame than any other frame.
                actual=np.asarray(decoded.convert('RGB').crop((0,910,1700,995)),dtype=float)
                errors=[float(np.mean(np.abs(actual-np.asarray(f.crop((0,910,1700,995)),dtype=float)))) for f in frames]
                assert int(np.argmin(errors))==i
        assert durations==[duration//10*10]*len(frames)
        verification.append(dict(mode=mode,frames=len(frames),decoded_frames=len(frames),fps=fps,
                                 frame_durations_ms=durations,frame_size=[1700,995],footer_frame_order_matches=True))
        audits.extend(mode_audit)
    with gzip.open(output/'display_audit.json.gz','wt') as f:json.dump(audits,f,separators=(',',':'))
    assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==h for p,h in hashes.items())
    report=dict(passed=True,animations=verification,events=len(catalogs),intervals=len(catalogs)-1,
                input_hashes_unchanged=True,physical_identities_assigned=False,production_tracker_changed=False,
                grouping_angle=60,association_gate_code_units=2.5/64,
                source_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in
                    [Path(__file__),Path('experiments/wind_bubble/plot_resolution_links.py'),Path('experiments/astrolink/surface_associations.py')]},
                input_sha256=hashes)
    (output/'verification.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(verification,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('directory',type=Path)
    p.add_argument('--output-dir',type=Path,required=True);p.add_argument('--fps',type=int,default=2)
    a=p.parse_args();run(a.directory,a.output_dir,a.fps)
