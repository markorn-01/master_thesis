"""Conservative provisional history segments, not physical shock identities."""
from experiments.astrolink.continuation_status import continuation_status


def stable_pairs(results):
    """Intersect unique, nonredundant continuation candidates across distances."""
    if len(results)<2:raise ValueError('Require at least two distance settings.')
    tables=[continuation_status(r) for r in results]
    parent_sets=[{p['parent_group'] for p in t['parents']} for t in tables]
    child_sets=[{c['child_group'] for c in t['children']} for t in tables]
    if any(s!=parent_sets[0] for s in parent_sets) or any(s!=child_sets[0] for s in child_sets):
        raise ValueError('All distance settings must describe the same adjacent catalogs.')
    candidates=[]
    for table in tables:
        candidates.append({(p['parent_group'],p['selected_child_group']) for p in table['parents']
                           if p['advisory_status']=='continuation_candidate'})
    return sorted(set.intersection(*candidates))


def assign_segments(previous, child_groups, pairs, next_id):
    """Inherit only from the immediately previous event; otherwise start anew."""
    children=sorted(set(child_groups));pairs=list(pairs)
    if len({p for p,c in pairs})!=len(pairs) or len({c for p,c in pairs})!=len(pairs):
        raise ValueError('Continuation pairs must be one-to-one.')
    if any(p not in previous or c not in children for p,c in pairs):raise ValueError('Unknown group in continuation pair.')
    incoming={c:p for p,c in pairs};current={};records=[]
    for child in children:
        parent=incoming.get(child)
        if parent is not None:segment=previous[parent]
        else:segment=next_id;next_id+=1
        current[child]=segment
        records.append(dict(group=child,segment_id=segment,previous_group=parent,
                            inherited=parent is not None,physical_identity_established=False))
    return current,records,next_id
