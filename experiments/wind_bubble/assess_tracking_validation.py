"""Consolidate frozen validation evidence; no simulations or model reruns."""
import argparse
import csv
import gzip
import hashlib
import json
from pathlib import Path

from experiments.astrolink.freeze_multiframe_baseline import verify_manifest


REFERENCES = {
    'synthetic_baseline': 'multiframe_tracking_baseline_v1',
    'delayed_support': 'multiframe_support_comparison_v1',
    'unequal_full_sequence': 'expanded_full_sequence_validation_n128',
    'equal_full_sequence': 'equal_history_expanded_validation_n128',
    'equal_mode_comparison': 'equal_history_mode_comparison_n128',
    'three_full_sequence': 'three_bubble_history_expanded_n064',
    'incremental_replay': 'streaming_history_replay_v1',
    'final_display': 'three_bubble_adaptive_sequence_final_n064',
    'display_counterexamples': 'three_bubble_surface_method_research_n064',
    'online_demo_tables': 'three_bubble_online_tracking_demo_n064/tables',
    'online_demo_animation': 'three_bubble_online_tracking_demo_n064/animation',
}


def run(output):
    if output.exists():
        raise ValueError('Use a fresh assessment directory.')
    hashes, reports, provenance = {}, {}, []
    for name, directory in REFERENCES.items():
        path = Path('outputs')/directory/'verification.json'
        local = {}
        reports[name] = verify_manifest(path, local)
        for filename, expected in local.items():
            if filename in hashes and hashes[filename] != expected:
                raise ValueError('References disagree about a shared input/version.')
            hashes[filename] = expected
        provenance.append(dict(evidence=name, report=str(path), checked_hash_entries=len(local)))

    def remember(path):
        path = Path(path)
        hashes[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
        return path

    def read_json(path):
        return json.loads(remember(path).read_text())

    def read_csv(path):
        with remember(path).open() as stream:
            return list(csv.DictReader(stream))

    folder = Path('outputs/multiframe_support_comparison_v1')
    synthetic = read_csv(folder/'synthetic_comparison.csv')
    archived_summary = read_csv(folder/'synthetic_summary.csv')
    summary = []
    for suite, identifiable in (('adjacent_regression', True), ('three_state_development', True),
                                 ('three_state_development', False)):
        selected = [r for r in synthetic if r['suite'] == suite and r['identifiable'] == str(identifiable)]
        configurations = {(r['case'], r['gate']) for r in selected}
        methods = {}
        for method in ('delayed_static_family', 'delayed_normal_motion_family'):
            rows = [r for r in selected if r['method'] == method]
            assert len(rows) == len(configurations)
            totals = {key: sum(int(r[key]) for r in rows) for key in
                ('baseline_false_links', 'baseline_missed_links', 'hypothetical_union_false_links',
                 'hypothetical_union_missed_links', 'additional_support_links')}
            original = next(r for r in archived_summary if
                (r['suite'], r['identifiable'], r['method']) == (suite, str(identifiable), method))
            assert int(original['configurations']) == len(rows)
            assert all(int(original[key]) == value for key, value in totals.items())
            methods[method] = totals
        assert all(methods['delayed_static_family'][k] == methods['delayed_normal_motion_family'][k]
                   for k in ('baseline_false_links', 'baseline_missed_links'))
        summary.append(dict(suite=suite, identifiable=identifiable,
                            configurations=len(configurations), methods=methods))
    assert [r['configurations'] for r in summary] == [27, 15, 6]
    assert len(synthetic) == reports['delayed_support']['synthetic_method_configurations'] == 96

    real = read_csv(folder/'real_comparison.csv')
    families = read_csv(folder/'families.csv')
    links = read_csv(folder/'support_links.csv')
    with gzip.open(remember(folder/'real_results.json.gz'), 'rt') as stream:
        full = json.load(stream)
    assert len(real) == len(full) == reports['delayed_support']['real_method_interval_gate_results'] == 144
    assert all(r['accepted_events_unchanged'] == 'True' and r['physical_truth_available'] == 'False'
               and int(r['conflicts']) == 0 for r in real)
    method_sets = {}
    for record in full:
        key = record['case'], record['parent_event'], record['gate_reference_cells']
        method_sets.setdefault(key, {})[record['method']] = {
            (r['parent_group'], r['child_group'], r['already_in_adjacent'])
            for r in record['result']['conditional_support']}
    assert len(method_sets) == 72
    assert all(r['delayed_static_family'] == r['delayed_normal_motion_family'] for r in method_sets.values())
    focus = []
    for case, event, child in (('unequal_two_n128', 10, 1), ('three_n064', 13, 4)):
        chosen = [r for r in families if (r['case'], int(r['parent_event']), int(r['child_group']), r['method']) ==
                  (case, event, child, 'delayed_normal_motion_family')]
        assert len(chosen) == 2
        parent_lists = [json.loads(r['parent_groups']) for r in chosen]
        assert parent_lists[0] == parent_lists[1]
        accepted_counts = []
        for gate in (1.5, 2.5):
            record = next(r for r in full if (r['case'], r['parent_event'], r['gate_reference_cells'], r['method']) ==
                          (case, event, gate, 'delayed_normal_motion_family'))
            combination = [e for e in record['result']['accepted_adjacent_events']
                           if e['kind'] == 'combination' and e['children'] == [child]]
            assert len(combination) == 1
            accepted_counts.append(len(combination[0]['parents']))
        assert accepted_counts[0] == accepted_counts[1]
        additions = [r for r in links if r['case'] == case and int(r['parent_event']) == event
                     and int(r['child_group']) == child and r['already_in_adjacent'] == 'False']
        additional_parents = sorted({int(r['parent_group']) for r in additions})
        overlaps = {}
        for parent in additional_parents:
            flags = {int(r['upstream_source_overlaps']) for r in additions if int(r['parent_group']) == parent}
            assert len(flags) == 1
            overlaps[str(parent)] = flags.pop()
        focus.append(dict(case=case, parent_event=event, child_event=event+1,
            accepted_combination_parents=accepted_counts[0], conditional_family_parents=len(parent_lists[0]),
            additional_parent_groups=additional_parents, added_parent_upstream_source_overlaps=overlaps,
            union_coverage_by_gate={r['gate_reference_cells']: float(r['child_union_coverage']) for r in chosen},
            physical_truth_available=False, inherited_identity_added=False))
    assert [(r['accepted_combination_parents'], r['conditional_family_parents']) for r in focus] == [(5, 7), (4, 5)]
    assert len(focus[0]['additional_parent_groups']) == 2 and len(focus[1]['additional_parent_groups']) == 1
    terminal = [r for r in real if r['window_status'] == 'pending_future']
    assert len(terminal) == 8  # Two cases, two variants and two gates; not eight physical shocks.

    histories = []
    for case, directory in (('equal_two_n128', 'equal_history_expanded_validation_n128'),
                            ('unequal_two_n128', 'expanded_full_sequence_validation_n128'),
                            ('three_n064', 'three_bubble_history_expanded_n064')):
        rows = read_csv(Path('outputs')/directory/'distance_agreement.csv')
        agreement_key = 'expanded_distances_agree' if case == 'unequal_two_n128' else 'event_sets_identical'
        agreeing = sum(r[agreement_key] == 'True' for r in rows)
        histories.append(dict(case=case, intervals=len(rows), both_gate_event_sets_agree=agreeing,
                              interpretation='Parameter agreement; no physical truth labels.'))
    assert [(r['both_gate_event_sets_agree'], r['intervals']) for r in histories] == [(16,16), (20,20), (15,16)]

    execution = []
    for name in ('online_streaming_iwr_compgpu14_n032', 'online_streaming_iwr_compgpu14_n064',
                 'three_bubbles_online_tracking_iwr_compgpu8_n064'):
        root = Path('outputs')/name
        meta, eq = read_json(root/'metadata.json'), read_json(root/'equivalence.json')
        assert meta['status'] == 'complete' and meta['backend'] == 'gpu' and meta['temporal_tracking']
        for key in ('passed', 'histogram_counts_exact', 'sparse_surfaces_match', 'background_slices_match',
                    'streaming_tracking_match', 'tracking_full_records_exact', 'tracking_batch_reference_match',
                    'final_gas_state_matches'):
            assert eq[key] is True, (name,key)
        assert eq['max_absolute_scalar_difference'] == 0. and eq['compared_events'] == len(meta['actual_times']) == 17
        for source, expected in meta['source_sha256'].items():
            source = source.split('/master_thesis/', 1)[-1] if source.startswith('/') else source
            path = Path(source)
            assert not path.is_absolute() and '..' not in path.parts
            assert hashlib.sha256(remember(path).read_bytes()).hexdigest() == expected
        tracking = read_json(root/'online/tracking/metadata.json')
        final = read_json(root/'online/tracking/finish.json')
        retained = []
        for index in range(17):
            with gzip.open(remember(root/'online/tracking'/f'event_{index:04d}.json.gz'), 'rt') as stream:
                record = json.load(stream)
            retained.append(record['footprint']['retained_geometry_catalogs'])
        assert max(retained) <= 2 and final['footprint']['max_processing_geometry_catalogs'] <= 3
        assert final['footprint']['retained_geometry_catalogs'] == 0 and tracking['status'] == 'complete'
        execution.append(dict(directory=str(root), num_sources=meta['num_sources'], num_cells=meta['num_cells'],
            events=eq['compared_events'], group_observations=eq['compared_group_observations'],
            scalar_fields_per_event=eq['fields_per_event'], max_absolute_scalar_difference=0.,
            complete_tracking_records_exact=True, batch_reference_matches=True, gpu_final_state_check_passed=True,
            max_retained_geometry_catalogs=max(retained), final_retained_geometry_catalogs=0))
    assert [r['group_observations'] for r in execution] == [25,39,45]

    accuracy = read_csv(Path('outputs/three_bubble_adaptive_sequence_final_n064/group_accuracy.csv'))
    minimum = min(float(r['fraction_within_quarter_cell']) for r in accuracy)
    assert len(accuracy) == 45 and minimum >= .95
    research = read_csv(Path('outputs/three_bubble_surface_method_research_n064/summary.csv'))
    assert any(r['case'] == 'near_sheets_same_group_counterexample' for r in research)
    demo = read_json('outputs/three_bubble_online_tracking_demo_n064/verification.json')
    assert demo['origin'] == 'hydrodynamic_host_callback' and demo['decoded_frames'] == 17
    assert demo['additional_conditional_relationships'] == 1 and demo['first_conditional_event'] == 15

    assessment = dict(step_2_complete=True, scope='Consolidation of existing controlled tracking and display evidence.',
        simulations_run=0, algorithms_changed=False, thresholds_changed=False,
        synthetic_development_summary=summary,
        synthetic_scope='Hypothetical union diagnostics, known development cases/gates, not held-out accuracy or promoted ancestry.',
        hydro_focus=focus, static_and_motion_hydro_memberships_identical_configurations=len(method_sets),
        hydro_pending_method_gate_records=len(terminal), gate_agreement=histories, gpu_execution_checks=execution,
        display=dict(group_observations=45, minimum_fraction_within_quarter_cell=minimum,
                     decoded_frames=17, first_added_support_event=15, physical_topology_established=False),
        claim='Online geometric shock-surface grouping and relationship histories with explicit uncertainty on the tested cases.',
        unique_physical_ancestry_established=False, hydro_false_link_rate_measured=False,
        general_turbulent_box_scaling_established=False, complete_workflow_cost_measured=False,
        source_reports=provenance)
    for filename, expected in hashes.items():
        assert hashlib.sha256(Path(filename).read_bytes()).hexdigest() == expected
    output.mkdir(parents=True)
    payload = output/'assessment.json'
    payload.write_text(json.dumps(assessment, indent=2)+'\n')
    verification = dict(passed=True, unique_checked_input_paths=len(hashes), source_reports=len(REFERENCES),
        synthetic_tables_reaggregated=True, hydro_relationships_reaggregated=True,
        gpu_execution_manifests_checked=True, historical_sources_and_inputs_currently_match=True,
        simulations_run=0, optimizer_reruns=0, input_sha256=hashes,
        source_sha256={str(Path(__file__)): hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},
        output_sha256={str(payload):hashlib.sha256(payload.read_bytes()).hexdigest()})
    (output/'verification.json').write_text(json.dumps(verification, indent=2)+'\n')
    print(json.dumps({k:verification[k] for k in ('passed','source_reports','unique_checked_input_paths','simulations_run','optimizer_reruns')}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    run(parser.parse_args().output_dir)
