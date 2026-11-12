# Bounded delayed fragment-support check

Primary-paper review and exact method attribution are documented in
`experiments/astrolink/TRACKLET_FRAGMENTS.md`.

**Real result:** all 14 earlier inner patches at 128³ form two conditional
support families of seven; the analogous 64³ window gives five per family.
Membership is identical at both tested gates and with/without backward normal-
motion prediction. Thus these data support the family description but show no
extra membership benefit from motion prediction. This is not proven ancestry.

`controlled_summary.csv` and `controlled_results.json` compare adjacent events,
static families and delayed families on seven cases × three gates. The delayed
version has 13/21 exact stipulated correspondences versus 9/21 static and 6/21
adjacent; its six false links on identical-observation ancestry controls remain
explicit. Do not summarize only the successful cases or treat this as general
tracking accuracy. `controlled_comparison.png` shows both false and missed links.

`real_families.csv` / `real_results.json` retain all eight real configurations,
family memberships, motion models, score margins and union coverage. Narrow
gate union coverage is 89.47% at 64³ and 91.70% at 128³; wide gate coverage is
100%. `verification.json` records 42 controlled dense checks, 42 shuffle checks,
eight real dense checks and source/input hashes. Original graphs remain intact.

Display comparison: `outputs/inner_fragment_family_review_v1/`.
Unequal-wind collision validation: prepared, awaiting the user's remote run.
