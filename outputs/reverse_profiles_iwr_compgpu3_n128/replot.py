"""Correct the resolution label using saved CSV data; no solver rerun."""
from pathlib import Path
import csv
import json
from experiments.wind_bubble.check_reverse_profiles import plot
root=Path(__file__).resolve().parent
metadata=json.loads((root/'metadata.json').read_text())
events=[]
for index in (12,15,20):
    with (root/f'profile_{index:04d}.csv').open() as f:
        profiles=[{k:float(v) for k,v in r.items()} for r in csv.DictReader(f)]
    with (root/f'samples_{index:04d}.csv').open() as f:
        samples=[{k:(v=='True' if v in ('True','False') else v if k=='band' else float(v))
                  for k,v in r.items()} for r in csv.DictReader(f)]
    events.append((metadata['actual_times'][index],profiles,samples))
plot(events,root/'reverse_profiles.png',num_cells=metadata['num_cells'])
