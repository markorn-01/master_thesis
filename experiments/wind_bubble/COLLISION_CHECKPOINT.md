# Collision investigation and thesis completion checkpoint

Date: 2026-11-24. Evidence reviewed through commit `982235c`.

## Main conclusion

The project has a working online workflow for detected shock surfaces, compact
statistics, geometric group histories, and smoother visualization. It does
**not** yet provide complete detection or physically verified ancestry of every
shock in colliding bubbles. That remaining scientific goal must not be marked
complete by finishing documentation or by passing execution-equivalence tests.

The latest reconstruction route is closed with a negative bubble-transfer
result. No accepted replacement detector has been found for the disputed
central layer. Another automatic threshold change or GPU rerun is not a
specified solution to that problem.

## Answer to Tobias's question

There are two different questions:

1. **Should genuine, sufficiently resolved colliding shocks be detected?** Yes.
   Independent exact reference cases demonstrate that opposing fronts can be
   detected and measured. We corrected two concrete surface-selection/sampling
   failures exposed by those cases.
2. **Does the particular late central layer in these bubble runs contain
   physical shock fronts that our finder misses?** This remains unverified.
   Its temperature-based zone rejection is established, but that alone neither
   proves the presence of a physical shock nor establishes its absence.

The known shock/contact counterexample establishes an incompleteness mechanism
for this finder. It is not proof that the same physical configuration explains
the central bubble layer.

### Short update to send manually

> I separated the detector question from the bubble interpretation. In exact
> reference cases, genuine opposing shocks should be found, and those tests
> exposed two bugs that are now corrected. We also confirmed that a nearby
> contact can hide a known shock from the current local criteria. However, I
> have not yet established whether the specific late central bubble layer
> contains missed physical shock fronts. The alternative methods tested so
> far do not pass the required controls on the bubble data, so I cannot present
> them as a validated fix. The gap occurs before grouping or visualization.

This is a draft, not a message sent to Tobias.

## Evidence that supports this answer

| Finding | Evidence and scope |
|---|---|
| Opposing-front failures corrected | [GPU before/after comparison](../../outputs/opposing_front_gpu_review_compgpu3_n032_n064/verification.json): all four targeted 32/64-cubed cases pass; six sampled configurations improve, none regress. [128-cubed reference run](../../outputs/opposing_front_gpu_compgpu3_n128/verification.json) extends the controlled check. These are analytic inputs, not evolved bubble convergence. |
| Corrected workflow remains equivalent | [Two-bubble 64-cubed comparison](../../outputs/opposing_front_online_iwr_compgpu3_n064/equivalence.json): 17 events, zero maximum finite scalar difference, exact histograms/sparse catalogs/tracking records, batch agreement and final-state agreement. This verifies execution paths, not physical completeness. |
| Contact mixing can hide a known shock | [228-case reference](../../outputs/shock_contact_reference_n032_n064/verification.json): of 180 shock/contact pairs, 74 recover the declared measurements, 96 are Mach-biased, six have no zone candidates, and four fail the location/direction criterion. The two unfiltered misses are projections of an exact Euler fan; the four additional misses use synthetic broadening. These development-case counts are not a population accuracy estimate. |
| Resolving the same known fan can restore detection | [Fixed physical gap](../../outputs/shock_contact_fixed_gap_n032_n064_n128/verification.json): no zone at 32 cells, biased Mach 2.236 at 64, correct Mach 3 at 128. This does not establish convergence of the evolved bubble. |
| Density-normal replacement rejected | [Fixed comparison](../../outputs/density_normal_detector_comparison_n032_n064/verification.json): none of the six known zone-rejected cases passes the alternative's sampled-entropy test. It is not promoted. |
| Joint reconstruction has conditional feasibility | [Controlled fits](../../outputs/joint_shock_contact_feasibility/verification.json): 10 positive profiles reconstruct the declared state under the assumed fan/blur family. Other controls receive no usable joint-pair claim. A known shock with an unmodeled blur is rejected, exposing model dependence. |
| That reconstruction does not transfer to these bubble controls | [Saved bubble check](../../outputs/bubble_joint_reconstruction_n128/verification.json): zero accepted shock models in 40 outer control windows. Forty central candidates have eight guarded cells each, below the unchanged prototype's 16-cell requirement, and are not fitted. This is not physical shock-absence evidence. |

The [time-resolved strip investigation](../../outputs/collision_front_motion_iwr_n128/assessment/verification.json)
also remained inconclusive after failing its strict outer control. The
[failure audit](../../outputs/collision_front_motion_iwr_n128/control_failure_audit/verification.json)
records sampled-state mismatch and sensitivity of pressure-feature speeds.
Neither result independently certifies or excludes hidden central fronts.

## Position relative to the original goal

| Goal | Current standing | Remaining limit |
|---|---|---|
| Analyze during hydrodynamic evolution | Implemented; matching online/offline and batch products checked | Shared calculations can reproduce the same physical bias. |
| Save compact evolving statistics and surfaces | Implemented for the tested one/two/three-source workflows | Saved statistics describe the selected detected samples, not all physical shocks. |
| Display smoother 3D fronts | Checked adaptive triangle displays available | Triangles are display products; close sheets within a group can still be problematic. A smooth surface cannot recover undetected samples. |
| Follow colliding structures | Streaming geometric group histories and branching/support records implemented and checked | Labels describe geometric continuation segments. Ambiguous support, fragmentation and combinations do not establish unique physical ancestry or source ownership. |
| Run efficiently for many bubbles | Snapshot storage is reduced; tracking cost is measured | The recorded three-bubble tracking benchmark adds 23.020 s, about 598.5%, to the diagnostics baseline. General turbulent-box scalability is unestablished. |
| Recover every relevant collision front | Not established | The central-layer question and contact-mixing incompleteness remain open. |

Useful existing deliverables are the
[three-bubble streaming demonstration](../../outputs/three_bubble_streaming_demo_preview/README.md),
[cleaned adaptive 3D animation](../../outputs/three_bubble_adaptive_animation_final_n064/README.md),
and [tracking cost results](../../outputs/tracking_benchmark_iwr_compgpu6_gpu0_n064/README.md).
They retain their original code/data provenance; do not relabel historical
movies or timings as results regenerated with the later detector corrections.

## Finite plan from this checkpoint

### 1. Communicate the finding

Use the short update above with the controlled shock/contact example and the
failed bubble-transfer figure. Finish when the distinction between a proven
finder limitation and an unresolved bubble interpretation is explicit. This
does not require a new simulation or imply that the original scientific goal
has been fully achieved.

### 2. Align the thesis with the verified implementation

Prepare the methods/validation text for the opposing-front correction and the
shock/contact counterexample. Update the online chapter to include the completed
streaming tracking checks and measured tracking cost, preserving the distinction
between execution equivalence and physical validation. State the limits of
geometric identities and display reconstruction in the discussion.

Finish when every result used in the thesis has one identified run/version,
the relevant figure, the measured outcome, and its scope. These are writing and
consistency tasks; the detector should not be changed merely to improve the
appearance of a final figure. Existing animation inputs retain their provenance.

### 3. Keep further research as an explicit extension

If complete recovery of the disputed fronts remains a required research
deliverable, that objective is still open. A next method must address evolving,
nonuniform, multiwave flow, and pass known shocks/contacts/smooth controls plus
the real outer control before central claims are accepted. An independent
propagation/jump test must likewise pass its outer control. A larger resolution
or a more flexible fit is not a success criterion by itself.

Before implementing such an extension, specify one candidate method, the evidence
it can add, positive/negative controls, available resolution/memory, and a
stopping rule. If it fails, report that failure rather than silently starting
another detector variant. No extension or GPU launch is triggered by this note.

This plan closes the current prototype branch and organizes the write-up. It
does not convert an unresolved physical question into a solved thesis result.
