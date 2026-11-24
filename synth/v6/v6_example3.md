# Example 3

[EXAMPLE INPUT]

Sequence:
1 M
2 V
3 N
4 F
5 Y
6 D
7 D
8 V
9 D
10 E
11 S
12 K
13 S
14 H
15 G
16 E
17 F
18 P
19 L
20 I
21 P
22 V
23 V
24 L
25 Q
26 N
27 S
28 S
29 E
30 L
31 S
32 V
33 R
34 T
35 I
36 P
37 T
38 G
39 N
40 E
41 I
42 I
43 E
44 S
45 V
46 H
47 L
48 T
49 K
50 W
51 L
52 R
53 K
54 Y
55 R
56 N
57 A
58 L
59 A
60 S
61 Q
62 L
63 D
64 R
65 Y
66 E
67 K
68 G
69 W
70 Q
71 S
72 K
73 I
74 A
75 N
76 F
77 R
78 L
79 Q
80 V
81 Q
82 H
83 V
84 I
85 N
86 Y
87 S
88 R
89 K
90 N
91 I
92 F
93 N
94 V
95 D
96 S
97 E
98 N
99 K
100 H
101 T
102 V
103 V
104 P
105 G
106 S
107 L
108 I
109 A
110 L
111 G
112 A
113 F
114 F
115 A
116 G
117 S
118 I
119 A
120 V
121 N
122 R
123 S
124 N
125 W
126 G
127 A
128 K
129 R
130 L
131 I
132 F
133 G
134 H
135 K
136 S
137 S
138 I
139 L
140 E
141 K
142 L
143 C
144 T
145 S
146 L
147 P
148 S
149 R
150 I
151 L
152 L
153 P
154 W
155 V
156 L
157 A
158 A
159 A
160 T
161 F
162 K
163 Y
164 W
165 A
166 P
167 Q
168 T
169 S
170 Q
171 N
172 L
173 V
174 N
175 A
176 T
177 E
178 N
179 D
180 L
181 L
182 P
183 A
184 D
185 F
186 V
187 K
188 S
189 Y
190 H
191 N
192 T
193 W
194 K
195 R
196 I
197 Y
198 E
199 E
200 G
201 Y
202 V
203 A
204 K
205 K
206 C
207 D
208 L
209 K
210 R
211 Q
212 I
213 D
214 Q
215 T
216 L
217 Q
218 K
219 N
220 I
221 R
222 Y
223 A
224 R
225 E
226 Q
227 L
228 Y
229 E
230 K
231 L
232 E
233 Q
234 A

Residue features:
- 1-234 (CHAIN): MICOS complex subunit MIC27
- 1-100 (TOPO_DOM): Mitochondrial intermembrane
- 44-227 (FAMILY): MICOS complex subunit MIC27 [IPR061953]
- 101-120 (TRANSMEM): Helical
- 121-141 (TOPO_DOM): Mitochondrial matrix
- 142-161 (TRANSMEM): Helical
- 162-234 (TOPO_DOM): Mitochondrial intermembrane

Protein: MICOS complex subunit MIC27 (Altered inheritance of mitochondria protein 37) (Mitochondrial contact site complex 27 kDa subunit)
UniProt: P50945
Organism: Saccharomyces cerevisiae (strain ATCC 204508 / S288c) (Baker's yeast)
Function summary: Component of the MICOS complex, a large protein complex of the mitochondrial inner membrane that plays crucial roles in the maintenance of crista junctions, inner membrane architecture, and formation of contact sites to the outer membrane.

Additional context:
- Family: Belongs to the apolipoprotein O/MICOS complex subunit Mic27 family.
- Subunit structure: Component of the mitochondrial contact site and cristae organizing system (MICOS) complex, composed of at least MIC10, MIC12, MIC19, MIC26, MIC27 and MIC60. This complex was also known under the names MINOS or MitOS complex.
- Subcellular location: Mitochondrion inner membrane (curator inference); Multi-pass membrane protein (curator inference). Note=Enriched at crista junctions. (experimental).
- Disruption phenotype: Increases frequency of mitochondrial genome loss. Partially altered shape of the mitochondrial network with condensed, fragmented mitochondria accumulating at the periphery of cells. 20-40% of mitochondria exhibit an increased inner membrane surface and stacks of lamellar cristae disconnected from the inner boundary membrane.

Gene Ontology terms:
- Molecular function:
  - None annotated
- Cellular component:
  - mitochondrial crista junction GO:0044284
  - MICOS complex GO:0061617
- Biological process:
  - positive regulation of protein oligomerization GO:0032461
  - cristae formation GO:0042407
  - protein-containing complex organization GO:0043933

Known interaction partners:
- MICOS complex subunit MIC12
- MICOS complex subunit MIC10
- MICOS complex subunit MIC19
- MICOS subunit MIC26
- MICOS complex subunit MIC60
- Sorting assembly machinery 50 kDa subunit
- Mitochondrial inner membrane m-AAA protease component YTA10
- Peroxisomal membrane protein PEX34
- Mitochondrial import receptor subunit TOM22
- Mitochondrial import receptor subunit TOM40

[EXAMPLE OUTPUT]

<|REASONING_START|>
Forty of these 234 residues sit in lipid, in two helices at 101 to 120 and 142 to 161, and everything else hangs off the membrane on one side or the other. The hundred before the first and the seventy-three after the second face the intermembrane space, with only 121 to 141 in the matrix between them, which makes this an inner membrane and puts nearly all the chain on its intermembrane side. Charge follows that split, strongly acidic at the amino end and strongly basic in the matrix loop, and tryptophan 125 sits five residues past the first helix at the depth where interfacial aromatics stack against headgroups, so that short matrix loop is a lipid-binding surface and not a linker. That leaves the helices themselves, and down the first the small residues line up, threonine 101, glycine 105, alanine 109, alanine 112, glycine 116 and alanine 119 within one eighty-degree arc, with every valine, leucine, isoleucine and phenylalanine outside it; lipid would tolerate leucine all the way round, so I read that flat stripe as a face for packing against a neighbouring helix rather than against acyl chains. The second carries arginine 149 near its middle, where lipid would charge for it, so something must bury that face too. Two faces to hold with, then, and nothing to transform with, its few cysteines too far apart to pair, its histidines scattered singly and none of the polar faces a conducting path would need, so it binds and does not transform. IPR061953 then names 44 to 227 for the MICOS complex subunit Mic27, which puts the chain in that complex without saying what it contributes there, and leaves 1 to 43 and 228 to 234 outside any entry. So I have a two-pass wedge in the mitochondrial inner membrane, holding with a packing face inside the bilayer and a lipid surface on the matrix leaflet, a subunit of a complex whose lipid, whose neighbouring helix and whose job in it are all still open.
<|REASONING_END|>

--- END OF WHAT YOU PRODUCE. Everything below is shown for context only. ---

The four paragraphs below, and the Final Answer after them, are written later by
someone working from the paragraph above. They are here so you can see what that
paragraph has to support and what it must leave open. Do not produce any of it.

For every route into the endoplasmic reticulum, that amino-terminal acid run is the wrong shape, being neither a hydrophobic signal sequence nor anything a signal recognition particle would grip. The first helix begins a hundred residues in, far too late to serve as a signal anchor, and neither helix sits near the carboxyl terminus, so it is not a tail anchor either. With all three entries closed I have an imported protein and not one inserted from the cytosol, and with two polar masses to move across whole it needs a protein-conducting translocase and helices that arrest transfer. There is no cleavable arginine-rich presequence, so it is not a matrix protein exported back into the membrane, and its cysteines are nowhere near the twin spacing that traps a small protein in the intermembrane space by oxidative folding. What decides which way round it lands is the potential across this membrane, on the order of a hundred and fifty millivolts and negative on the matrix face, which pulls the basic loop in and holds the acidic amino terminus out, the way round they already sit, so the lipid surface is on the matrix leaflet.

Anionic phospholipid is as far as charge alone goes, and this membrane makes one of its own: cardiolipin, a dimeric species carrying two phosphates and four acyl chains, synthesised on the matrix face and essentially confined to mitochondria. A basic patch on that leaflet therefore meets cardiolipin more often than anything else, and its two phosphates are a better match to a patch of basic side chains than any monoanionic lipid would be. The small headgroup over a wide four-chain base also gives it a cone shape. Charge complementarity still would not tell it from phosphatidylglycerol, the monomeric lipid it is built from, and nothing in this sequence draws a shaped headgroup pocket. What persuades me is that the animal subunits built on this same plan, APOO and APOOL, discriminate between exactly those two lipids, and a preference between two lipids of like charge is the behaviour of a shaped site rather than of a charge patch. That is extrapolation from the build and not anything this chain proves, and I place the site no more precisely than the interfacial band. Its activity is therefore cardiolipin binding and nothing catalytic, and what it contributes to the assembly it sits in is structural molecule activity.

This membrane is not flat. It is thrown into cristae, and each crista opens to the boundary membrane through a narrow neck, the crista junction, which is the tightest curvature the inner membrane carries. A neck of that kind is a saddle, curved one way along its axis and the other way around it, and a lipid whose headgroup is small over a wide acyl base packs into that geometry where a cylindrical one cannot. Cardiolipin therefore partitions to necks rather than to flat membrane, and a subunit that carries it while packing against the helix doing the bending concentrates the accommodating lipid exactly where the strain is, lowering the free-energy cost of the curved state. Nothing on this chain bends a membrane, though, since an anchor and a flat stripe cannot deform a bilayer, so what I do is stabilise a curved scaffold, not generate the curvature. An oligomer whose curved product costs less free energy is an oligomer that holds together, so delivering the lipid into its neighbourhood should be positive regulation of that protein's oligomerisation. Setting the assembly state of the thing one is a subunit of is protein-containing complex organisation, and since what that assembly builds is the neck itself, the process served at the scale of the organelle is cristae formation.

A stripe cut for packing has to meet a matching one, and what presents a glycine face on its own transmembrane helices in this membrane is Mic10, the core subunit of the complex I am already in. It folds as a hairpin and oligomerises through those faces, and an oligomer of hairpins is what bends the membrane into a neck, so a glycine and alanine stripe like the one from 101 to 119 is both the surface that oligomer presents and the surface that would join it. What this chain brings that Mic10 lacks is the lipid. Mic26, the paralogue of the same build, competes for that packing surface without carrying it, and Mic12 completes the same subcomplex as a connector, so the two paralogues should set the oligomer's assembly state between them with the lipid-carrying one as the positive arm. A neck also has to hold its position, which it does because the inner membrane is pinned to the outer one. The same complex does that through its other arm, Mic60 with Mic19 reaching across the intermembrane space to the Tom40 import channel, its receptor Tom22 and the Sam50 barrel-assembly machinery, so membership puts this protein at an inner-outer membrane contact site as well as at the neck. A patch where the two mitochondrial membranes are already pinned together is also where anything docking from outside finds purchase, and in this yeast Pex34 tethers the peroxisome to the mitochondrion, so I expect that contact over the same patch. The matrix-facing stretch the field placed there is also a handle, since the m-AAA protease Yta10 with its partner Yta12 presents its catalytic ring to the matrix and can pull only on a substrate that offers it one. Copies that never find the complex are degraded from that side, and the protein's steady-state level reports on its own assembly. The F1Fo ATP synthase oligomerises into dimer rows that bend this same bilayer the opposite way along crista rims, and two systems imposing opposite curvature on one continuous membrane have to partition it, so I expect the two to meet where necks give way to rims. Losing this subunit should blunt inner-membrane architecture and respiratory growth without abolishing crista junctions, since Mic10 remains.


<|FINAL_ANSWER_START|>
- Summary: A non-catalytic, architectural subunit of the MICOS complex in the budding yeast mitochondrial inner membrane, anchored by a single transmembrane helix whose flat glycine and alanine stripe is cut for packing against a neighbouring helix. A basic patch with an interfacial tryptophan lies on the matrix leaflet just after the anchor and binds cardiolipin, the cone-shaped anionic lipid this membrane makes for itself, while the acidic amino-terminal region is held on the intermembrane-space side by the potential across the same bilayer. A crista junction is a saddle that a cone-shaped lipid packs into where a cylindrical one cannot, so packing alongside the glycine-faced oligomer that bends the neck while holding cardiolipin against it lowers the cost of the curved state, stabilising the scaffold rather than generating the curvature. Opposite a paralogue that competes for the same surface without bringing the lipid, it is the positive arm setting that oligomer's assembly state, and through that it organises its complex and shapes cristae. The same complex reaches across the intermembrane space to the outer-membrane import channel and the barrel-assembly machinery, so it also sits where the two mitochondrial membranes are held in apposition; and its matrix-exposed stretch leaves unassembled copies open to proteolytic quality control from that side.
- InterPro:
  - IPR061953 MICOS complex subunit MIC27 [44-227]
- Confident GO Terms:
  - Molecular Function:
    - None
  - Cellular Component:
    - mitochondrial crista junction GO:0044284
    - MICOS complex GO:0061617
  - Biological Process:
    - positive regulation of protein oligomerization GO:0032461
    - protein-containing complex organization GO:0043933
    - cristae formation GO:0042407
- Speculated GO Terms:
  - Molecular Function:
    - cardiolipin binding GO:1901612
    - structural molecule activity GO:0005198
  - Cellular Component:
    - mitochondrial inner-outer membrane contact site GO:0044289
  - Biological Process:
    - None
- Hypothesized Interaction Partners:
  - MICOS complex subunit MIC10
  - MICOS subunit MIC26
  - MICOS complex subunit MIC12
  - MICOS complex subunit MIC60
  - MICOS complex subunit MIC19
  - Mitochondrial import receptor subunit TOM40
  - Mitochondrial import receptor subunit TOM22
  - Sorting assembly machinery 50 kDa subunit
  - Mitochondrial inner membrane m-AAA protease component YTA10
  - Mitochondrial inner membrane m-AAA protease component YTA12
  - Peroxisomal membrane protein PEX34
  - F1Fo ATP synthase dimer-row subunits
<|FINAL_ANSWER_END|>
