# Knee MRI structured-reading knowledge base (source for RAG chunking)

Each `## ` heading below becomes one retrievable chunk in the knowledge index
(see `rag/knowledge_base.py`). Condensed from Dr. Sandeep's `mri-interpreter`
skill — knee section — for use as the text side of the multimodal RAG.

## Meniscus grading
Grade 1: focal globular intrameniscal hyperintensity, not reaching any surface —
normal variant/early degeneration, asymptomatic. Grade 2: linear intrameniscal
hyperintensity approaching but not unequivocally breaching a surface. Grade 3
(true tear): signal unequivocally contacts/breaches a surface, or causes surface
deformity. Two-slice rule: don't call a tear on a single slice — signal must
breach the surface on at least 2 consecutive slices, or across sequences.

## Meniscus tear morphology
Horizontal/oblique (cleavage, degenerative — most common overall, especially
posterior horn medial meniscus). Radial ("parrot-beak," free-edge — causes
blunting/truncation on coronal/axial). Longitudinal (breaches both surfaces).
Vertical flap (displaced fragment). Displaced bucket-handle: absent bow-tie sign
on at least 2 consecutive peripheral sagittal slices, plus double-PCL sign
(fragment flipped anterior/inferior to PCL), plus truncated meniscal body on
coronal. Root tear: complete fluid gap ("empty root") within ~9mm of the tibial
anchor — functionally as significant as a bucket-handle tear even though
visually subtler.

## Meniscus mimics and pitfalls
Transverse (intermeniscal) ligament across the anterior horns in Hoffa's fat pad.
Ligament of Wrisberg (posterior meniscofemoral, ~84% prevalence) and ligament of
Humphrey (anterior meniscofemoral, ~37%) near the posterior horn of the lateral
meniscus — the "Wrisberg pseudotear." Popliteus tendon sheath adjacent to the
posterior horn lateral meniscus. Peripheral linear intermediate signal from a
normal perimeniscal vessel. Speckled ACL-fiber insertion into the anterior horn
lateral meniscus (up to 60% of normals). Magic-angle artifact. Postoperative
meniscus stays bright for years — MR arthrography needed to call re-tear.

## ACL assessment
Parallels Blumensaat's line, posterolateral femoral condyle to anteromedial
tibial plateau. Direct signs of tear: discontinuity, wavy/lax non-parallel
contour, empty notch sign at the femoral attachment, gap sign (fluid line
interrupting the normally jet-black femoral attachment), footprint sign
(fluid/detachment at tibial insertion). Partial tear grading: low-grade (<50%
fibers, still roughly parallel to Blumensaat's) vs high-grade (>50-75%, lax/
wavy). Indirect/secondary signs when hemarthrosis obscures the ligament itself:
anterior tibial translation >9mm, deepening of the lateral femoral condyle notch
>2mm, buckling of the PCL, and the pivot-shift bone bruise pattern (lateral
femoral condyle terminal sulcus + posterolateral tibial plateau) — near-
pathognomonic even before assessing the ligament directly. Mucoid degeneration:
markedly thickened/expanded ACL, diffuse internal hyperintensity, intact
margins, no secondary tear signs ("celery stalk" appearance) — don't mistake for
tear.

## PCL assessment
"Inverted hockey-stick"/comma shape, medial femoral condyle to posterior tibial
plateau, uniformly dark, seen in its entirety on one sagittal slice. No
interdigitating fat (unlike ACL) — use a lower threshold to call intrasubstance
signal abnormal here. Complete disruption is rare; more often intrasubstance
fluid signal/thickening/contour expansion (partial). Mechanism: posterior tibial
force (dashboard injury).

## MCL and LCL grading
MCL: flat band, extrasynovial, continuous with the capsule, inserts into the
medial meniscus. Grading — I: periligamentous edema, fibers intact. II:
thickened ligament with intrasubstance signal, fibers still continuous
(partial). III: frank discontinuity/complete disruption. Isolated MCL injury is
uncommon given how tightly it's bound to adjacent medial structures — recheck
the meniscus and posteromedial corner. LCL: round/cord-like, forms a conjoined
"V" insertion with biceps femoris at the fibular head (interdigitating fat here
is normal). Grading mirrors MCL.

## Posterolateral and posteromedial corners
Posterolateral corner (biceps femoris, LCL, popliteus tendon/muscle/
popliteofibular ligament, arcuate ligament, fabellofibular ligament): secondary
stabilizer to the cruciates, resists varus stress and external rotation.
Frequently co-injured with ACL tears; missed PLC injury is a leading cause of
failed ACL reconstruction. Watch for popliteus magic-angle artifact (curves
through all 3 planes) — confirm real injury via muscle-belly edema, tendon
irregularity, or wavy contour, not signal alone. Posteromedial corner (posterior
oblique ligament, oblique popliteal ligament, semimembranosus tendon/
expansions, posterior horn medial meniscus, posteromedial capsule): dynamic
stabilizer against valgus stress and anteromedial rotatory instability. Check
for a ramp lesion (meniscocapsular separation at the medial meniscus posterior
horn) — commonly missed alongside ACL tears.

## Cartilage grading (modified Outerbridge)
Grade 0 normal. Grade 1 signal change/softening, intact contour. Grade 2
partial-thickness fissuring/fraying <50% depth. Grade 3 fissuring >50% depth
without exposed bone. Grade 4 full-thickness loss with subchondral bone
exposure, often with subchondral edema-like signal or cysts — subchondral
marrow signal change is the hallmark clue that a defect is full-thickness;
always look at the bone under a cartilage lesion.

## Bone marrow / bone bruise patterns
Contusion/bone bruise: low T1, high T2/STIR. Always name the implied mechanism:
lateral femoral condyle + posterolateral tibial plateau = pivot-shift/ACL.
Lateral femoral condyle (anterior) + medial patellar facet ("kissing
contusion") = transient lateral patellar dislocation. Marrow reconversion (red
marrow) is a normal-variant mimic of infiltrative disease — patchy, usually
symmetric, intermediate signal, no cortical destruction/soft-tissue mass.

## Joint fluid, synovium, Baker cyst
Small effusion after activity is normal. Distinguish effusion from synovial
hypertrophy by windowing. A Baker (popliteal) cyst must arise specifically
between the semimembranosus tendon and the medial head of gastrocnemius — a
cyst elsewhere in the popliteal fossa is a ganglion, not a true Baker cyst.

## Extensor mechanism
Quadriceps tendon is trilaminar (superficial = rectus femoris, middle = vastus
medialis/lateralis, deep = vastus intermedius) with normal interposed fat —
don't mistake normal laminar fat for a tear. Patellar tendon: check width on
axial, since partial tears can be confined to one edge and missed on sagittal
alone. Hoffa's fat pad: superolateral edema = impingement pattern; shearing
injury = acute edema; chronic Hoffa's disease = contracted, puckered, fibrotic
fat pad from prior shearing.

## Osteoarthritis pattern
Full-thickness cartilage loss, subchondral sclerosis, osteophytes, subchondral
cysts/geodes. Classically asymmetric (compartment-selective) — a symmetric,
diffuse pattern should raise suspicion for secondary OA from an underlying
cause (crystal disease, RA, hemophilia, prior trauma/meniscal-labral tear)
rather than primary OA.
