# Four-bubble fragment review

Read-only review of the completed equal-strength 64³ GPU run at
`t=0.13 → 0.14`, with `t=0.15` supplying delayed evidence.

## What the figures mean

Each figure follows one small geometric family through three observations:

- **A, t=0.13:** six fragments, each containing 20 detected surface samples.
  Five have accepted links to B (teal); the sixth has conditional support
  from the saved delayed analysis (orange).
- **B, t=0.14:** an independently detected group containing 303 samples.
- **C, t=0.15:** its accepted continuation, containing 339 samples.
  This observation supplies the additional evidence about A→B.

The four families contain 20 accepted fragment links and four additional
conditional links. Both matching distances (1.5 and 2.5 reference cells) and
both delayed variants (static and normal-motion) agree on family membership.
Union coverage is 84.16% at the narrow distance and 100% at the wide distance.
These are geometric coverage fractions, not confidence probabilities.
The motion variant does not improve on static support in this symmetric case.

| B group | Accepted A groups | Additional conditional A group | C group |
|---|---|---|---|
| 1 | 3, 5, 7, 8, 21 | 6 | 1 |
| 2 | 9, 10, 11, 12, 22 | 4 | 2 |
| 3 | 27, 30, 31, 32, 45 | 29 | 3 |
| 4 | 28, 33, 35, 36, 46 | 34 | 4 |

The accepted model allows at most five parents in a combination. Its accepted
decisions remain unchanged: delayed support does not promote the sixth link,
assign an inherited identity, or establish physical ancestry. Other acceptance
and exclusivity rules also affect the accepted graph.

The figures show original detected positions, without smoothing or regrouping.
Sample counts are not conserved mass or surface area. Each C group has six
upstream sampling endpoints inside an injection sphere; A and B have none.
Geometric support does not resolve this sampling-quality limitation or
validate the physical interpretation of the smaller surfaces.

## Verification and reproduction

`verification.json` records four independent blocked-dense geometry audits,
original catalog row joins, source-overlap checks, input/output/source hashes,
and the unchanged accepted/conditional roles. All checks passed. The four
symmetry-related families are examples from one run, not independent physical
validation cases.

```sh
python -m experiments.wind_bubble.review_streaming_fragments \
  --directory outputs/four_bubbles_iwr_compgpu2_n064 \
  --parent-event 13 \
  --output-dir outputs/four_bubble_fragment_review_n064_repeat
```

Use a fresh output directory. No simulation or new tracking is performed.
