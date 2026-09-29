#!/usr/bin/env python3
"""Check sensor coverage/finite values before the proprioception pilot; no fitting."""
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from droid_qp.paths import DATA, OUT  # noqa: E402


def main():
    manifest = DATA / 'dataset_manifest.json'
    clips = json.loads(manifest.read_text())['clips']
    shards = json.loads((DATA / 'shard_index.json').read_text())
    groups = {}
    for i,c in enumerate(clips):
        key = c['split']+'/'+c['video_id'].split('+')[0]
        stats = groups.setdefault(key,dict(clips=0,frames=0,min=np.full(15,np.inf),max=np.full(15,-np.inf)))
        with np.load(DATA / 'gen'/shards[c['video_id']]/'actions/labels'/f"{c['video_id']}.npz") as a:
            state = np.concatenate([a[k] for k in ['joint_positions','joint_velocities','gripper_positions']],axis=1)
            if state.shape != (len(a['valid']),15) or not np.isfinite(state).all():
                raise ValueError(f"Bad proprioception coverage/values: {c['video_id']}")
        stats['clips'] += 1; stats['frames'] += len(state)
        stats['min'] = np.minimum(stats['min'],state.min(0));stats['max'] = np.maximum(stats['max'],state.max(0))
        if (i+1)%2000==0:
            print(f'{i+1}/{len(clips)} sensor files checked',flush=True)
    for s in groups.values():
        s['min']=s['min'].tolist();s['max']=s['max'].tolist()
    result = dict(manifest_sha256=hashlib.sha256(manifest.read_bytes()).hexdigest(),
                  fields=['joint_positions[7] radians','joint_velocities[7] radians/second',
                          'gripper_positions[1] scaled closure: 0=open, 0.725=closed'],
                  clips_checked=len(clips), all_finite_and_covered=True, per_split_lab=groups,
                  note='Read-only audit, not normalization fitting; model uses fixed physical scales.')
    out=OUT / 'proprio_data_audit.json';out.write_text(json.dumps(result,indent=2)+'\n')
    print(f'PASS: {len(clips)} sensor files. {out}',flush=True)


if __name__=='__main__':
    main()
