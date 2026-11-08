"""Advisory descriptions of geometric support, without assigning track identities."""
from __future__ import annotations


def continuation_status(result):
    """Return new rows; never change the original matcher result.

    Requires support_evidence output, including exclusive-support diagnostics.
    Multiple-parent links remain multiple-parent support even when one is
    redundant. A continuation candidate is local evidence, not physical truth.
    """
    parents=[];children=[]
    for parent in result['parents']:
        group=parent['parent_group']
        links=[link for link in result['links'] if link['parent_group']==group]
        if len(links)>1:raise ValueError('Expected at most one selected target per parent.')
        link=links[0] if links else None
        accepted=bool(link and link['accepted'])
        child=next((c for c in result['children'] if link and c['child_group']==link['child_group']),None)
        if link and child is None:raise ValueError('Link target missing from child records.')
        if link:
            # Do not silently interpret missing diagnostics as nonredundancy.
            exclusive=link['exclusive_child_support_samples'];redundant=link['redundant_support']
        else:exclusive=None;redundant=False
        if parent['status']=='ambiguous_target':status='ambiguous'
        elif not accepted:status='unmatched'
        elif len(child['parent_groups'])>1:status='multiple_parent_support'
        elif redundant:status='unmatched'
        else:status='continuation_candidate'
        parents.append(dict(parent_group=group,best_child_group=parent['best_child_group'],
                            selected_child_group=link['child_group'] if link else None,
                            matcher_parent_status=parent['status'],matcher_accepted=accepted,
                            advisory_status=status,parent_coverage=parent['parent_coverage'],
                            runner_up_parent_coverage=parent['runner_up_parent_coverage'],
                            eligible_parents_of_target=len(child['parent_groups']) if child else 0,
                            exclusive_child_support_samples=exclusive,redundant_support=redundant,
                            physical_identity_established=False))
    for child in result['children']:
        incoming=[p for p in parents if p['selected_child_group']==child['child_group'] and p['matcher_accepted']]
        if not child['accepted']:status='unmatched'
        elif len(incoming)>1:status='multiple_parent_support'
        elif len(incoming)==1 and incoming[0]['advisory_status']=='continuation_candidate':status='continuation_candidate'
        else:status='unmatched'
        children.append(dict(child_group=child['child_group'],advisory_status=status,
                             matcher_accepted=child['accepted'],accepted_parent_groups=[p['parent_group'] for p in incoming],
                             union_child_coverage=child['union_child_coverage'],physical_identity_established=False))
    return dict(parents=parents,children=children)


def distance_sensitivity(tables):
    """Compare same-event tables across distances; return separate warning flags.

    Inputs must describe the same parent catalog, with unique parent IDs.
    Coverage differences alone are not classified as a status change.
    """
    maps=[{p['parent_group']:p for p in t['parents']} for t in tables]
    if not maps:raise ValueError('Require at least one distance setting.')
    if any(set(m)!=set(maps[0]) for m in maps):raise ValueError('Parent catalogs must match across distances.')
    return {group:len({(m[group]['advisory_status'],m[group]['selected_child_group'],
                        m[group]['best_child_group'],m[group]['redundant_support']) for m in maps})>1
            for group in maps[0]}
