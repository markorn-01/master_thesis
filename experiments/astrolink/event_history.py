"""Assemble adjacent group events without inventing persistent physical identities."""
from experiments.astrolink.continuation_segments import assign_segments


def assemble_history(groups_by_event, times, records):
    """Continue segments only through events selected at both tested distances.

    Combination/separation events remain explicit hyperedges. Distance-specific
    events are alternatives and never propagate segment labels.
    """
    if len(times) != len(groups_by_event) or any(b <= a for a, b in zip(times, times[1:])):
        raise ValueError('Require matching catalogs and strictly increasing times.')
    nodes, events = [], []
    previous, counter = {}, 0
    for index, groups in enumerate(groups_by_event):
        groups = sorted(groups)
        if len(set(groups)) != len(groups):
            raise ValueError('Duplicate group in catalog.')
        shared, hypotheses = set(), {}
        if index:
            interval = [r for r in records if r['parent_event'] == index - 1]
            if len(interval) != 2 or len({r['gate_reference_cells'] for r in interval}) != 2:
                raise ValueError('Require exactly two distance comparisons per interval.')
            selections = []
            for record in interval:
                result = record['result']
                if ({p['parent_group'] for p in result['parents']} != set(previous)
                        or {c['child_group'] for c in result['children']} != set(groups)):
                    raise ValueError('Comparison group sets do not match catalogs.')
                selected = set()
                occupied_p, occupied_c = set(), set()
                for event in result['events']:
                    kind, parents, children = event['kind'], tuple(sorted(event['parents'])), tuple(sorted(event['children']))
                    arity = (len(parents), len(children))
                    if not (kind == 'continuation' and arity == (1, 1)
                            or kind == 'combination' and arity[0] >= 2 and arity[1] == 1
                            or kind == 'separation' and arity[0] == 1 and arity[1] >= 2):
                        raise ValueError('Invalid event arity.')
                    if (not set(parents) <= set(previous) or not set(children) <= set(groups)
                            or occupied_p.intersection(parents) or occupied_c.intersection(children)):
                        raise ValueError('Invalid or conflicting selected event.')
                    occupied_p.update(parents); occupied_c.update(children)
                    key = kind, parents, children
                    selected.add(key)
                    hypotheses.setdefault(key, []).append(record['gate_reference_cells'])
                selections.append(selected)
            shared = selections[0] & selections[1]
        pairs = [(p[0], c[0]) for kind, p, c in sorted(shared) if kind == 'continuation']
        current, rows, counter = assign_segments(previous, groups, pairs, counter)
        for row in rows:
            nodes.append(dict(node_id=f"e{index:04d}:g{row['group']}", event=index,
                              time=float(times[index]), **row))
        for (kind, parents, children), gates in sorted(hypotheses.items()):
            events.append(dict(event_id=f'h{len(events):04d}', parent_event=index-1,
                               child_event=index, kind=kind,
                               parents=[f'e{index-1:04d}:g{p}' for p in parents],
                               children=[f'e{index:04d}:g{c}' for c in children],
                               status='shared' if (kind, parents, children) in shared else 'distance_specific',
                               selected_at_gates=sorted(gates)))
        previous = current
    return dict(nodes=nodes, events=events, segment_count=counter,
                physical_identity_established=False,
                policy='Only shared continuations inherit segment labels; branches stay explicit.')
