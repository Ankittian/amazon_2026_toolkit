# EDA Report — Business Entity Resolution
_Generated at 2026-09-25 09:57:48_
---
## 1  Ground Truth Structure (Train)
    - **total_s1_entities**: 2206821
    - **no_match_pct**: 5.584820880352326
    - **avg_matches_per_s1**: 3.4612526344456573
    - **median_matches**: 3.0
    - **max_matches**: 11
    - **avg_s2_matches**: 1.673728408420982
    - **avg_s3_matches**: 1.7875242260246753
    - **s2_match_counts_hist**: {1: 789108, 2: 652779, 3: 333957, 0: 287745, 4: 119078, 5: 24154}
    - **s3_match_counts_hist**: {1: 716417, 2: 668375, 3: 372443, 0: 266276, 4: 145116, 5: 35378, 6: 2816}
---
## 2  Per-Source Field Statistics
### Train-S1
**Rows**: 2,206,821

#### `business_name`
- Missing: **0.00%**
- Unique: **1,539,229**
- Duplicate rate: **30.25%**
- Length (min/median/max): **3 / 24 / 105**
- Avg token count: **3.55**
- **Top 5 values**:
  - `Primary Care Group` (253)
  - `Ear Nose & Throat Group` (251)
  - `Pediatric Group` (222)
  - `Womens Health Group` (220)
  - `Physical Therapy Group` (218)

#### `business_address`
- Missing: **0.00%**
- Unique: **2,130,606**
- Duplicate rate: **3.45%**
- Length (min/median/max): **11 / 41 / 256**
- Avg token count: **8.03**
- **Top 5 values**:
  - `108 Norle Street, College Twp, PA` (14)
  - `104 Roadrunner Circle, Elephant Butte, NM` (13)
  - `218 Phillips Street, Storm Lake, IA` (13)
  - `43 Franklin Street, Delaware, OH` (13)
  - `55, Lavkush Vihar, Singhpur, Kachhar Bithoor Road, Kalyanpur, Kanpur Nagar, Utta` (13)

#### `country`
- Missing: **0.00%**
- Unique: **2**
- Duplicate rate: **100.00%**
- Length (min/median/max): **2 / 2 / 5**
- Avg token count: **1.00**
- **Top 5 values**:
  - `US` (1323633)
  - `India` (883188)

#### Country Distribution
- US: 1,323,633
- India: 883,188

#### Name-Specific Metrics
- Legal suffix present: **62.40%**
- Avg char length: **24.0**
- Avg token count: **3.55**
- Avg punctuation chars per name: **0.36**

#### Address-Specific Metrics
- Missing: **0.00%**
- Avg numeric-token %: **18.76%**
- Avg token count: **8.03**
- Common abbreviation counts:
  - `ct`: 19,894
  - `apt`: 19,768
  - `rd`: 19,722
  - `st`: 16,162
  - `dr`: 8,330
  - `pl`: 4,367
  - `ave`: 3,180
  - `ste`: 2,688
### Train-S2
**Rows**: 5,034,616

#### `business_name`
- Missing: **0.00%**
- Unique: **4,402,009**
- Duplicate rate: **12.57%**
- Length (min/median/max): **2 / 25 / 104**
- Avg token count: **3.50**
- **Top 5 values**:
  - `Primary Care` (320)
  - `Physical Therapy` (307)
  - `Urgent Care` (297)
  - `Womens Health` (297)
  - `Behavioral Health` (296)

#### `business_address`
- Missing: **3.36%**
- Unique: **4,337,262**
- Duplicate rate: **13.85%**
- Length (min/median/max): **8 / 37 / 249**
- Avg token count: **7.54**
- **Top 5 values**:
  - `` (168967)
  - `842 38, CALEDONAI TOWNSHIP, WI` (18)
  - `1948 WEAVER FOREST WAY, MORRISVILLE, NC` (18)
  - `66 FRANK EDGE, ALTO, TX` (17)
  - `G-58/1, BENSON CROSS ROAD, BANGALORE., Karnataka` (16)

#### `country`
- Missing: **0.00%**
- Unique: **2**
- Duplicate rate: **100.00%**
- Length (min/median/max): **2 / 2 / 5**
- Avg token count: **1.00**
- **Top 5 values**:
  - `US` (3016817)
  - `India` (2017799)

#### Country Distribution
- US: 3,016,817
- India: 2,017,799

#### Name-Specific Metrics
- Legal suffix present: **45.40%**
- Avg char length: **25.1**
- Avg token count: **3.50**
- Avg punctuation chars per name: **0.62**

#### Address-Specific Metrics
- Missing: **3.36%**
- Avg numeric-token %: **18.40%**
- Avg token count: **7.54**
- Common abbreviation counts:
  - `st`: 333,485
  - `rd`: 330,958
  - `dr`: 283,537
  - `ave`: 224,070
  - `ln`: 115,505
  - `ct`: 110,281
  - `pl`: 39,921
  - `blvd`: 25,910
### Train-S3
**Rows**: 5,285,603

#### `business_name`
- Missing: **0.00%**
- Unique: **4,651,609**
- Duplicate rate: **11.99%**
- Length (min/median/max): **2 / 25 / 123**
- Avg token count: **3.53**
- **Top 5 values**:
  - `Primary Care` (421)
  - `Physical Therapy` (399)
  - `Pediatric Dental` (393)
  - `Womens Health` (379)
  - `Urgent Care` (377)

#### `business_address`
- Missing: **3.33%**
- Unique: **4,632,765**
- Duplicate rate: **12.35%**
- Length (min/median/max): **2 / 42 / 240**
- Avg token count: **7.42**
- **Top 5 values**:
  - `` (175916)
  - `Ground Floor, Bangalore, KA` (29)
  - `Floor, Mumbai, MH` (26)
  - `18, Kolkata, Howrah, WB` (26)
  - `303, Mumbai, MH` (25)

#### `country`
- Missing: **0.00%**
- Unique: **2**
- Duplicate rate: **100.00%**
- Length (min/median/max): **2 / 2 / 5**
- Avg token count: **1.00**
- **Top 5 values**:
  - `US` (3170056)
  - `India` (2115547)

#### Country Distribution
- US: 3,170,056
- India: 2,115,547

#### Name-Specific Metrics
- Legal suffix present: **47.87%**
- Avg char length: **25.2**
- Avg token count: **3.53**
- Avg punctuation chars per name: **0.64**

#### Address-Specific Metrics
- Missing: **3.33%**
- Avg numeric-token %: **18.78%**
- Avg token count: **7.42**
- Common abbreviation counts:
  - `st`: 331,436
  - `rd`: 326,238
  - `dr`: 281,587
  - `ave`: 224,220
  - `ln`: 116,402
  - `ct`: 72,623
  - `pl`: 39,049
  - `apt`: 35,543
### Test-S1
**Rows**: 1,732,544

#### `business_name`
- Missing: **0.00%**
- Unique: **1,238,867**
- Duplicate rate: **28.49%**
- Length (min/median/max): **3 / 24 / 92**
- Avg token count: **3.52**
- **Top 5 values**:
  - `Bordeaux Club SARL` (205)
  - `Nantes Club SARL` (157)
  - `Lille Club SARL` (147)
  - `Urgent Care Group` (123)
  - `Lille Club SAS` (122)

#### `business_address`
- Missing: **0.00%**
- Unique: **1,677,483**
- Duplicate rate: **3.18%**
- Length (min/median/max): **11 / 50 / 268**
- Avg token count: **8.59**
- **Top 5 values**:
  - `12 RUE Lyderic, Lille, Hauts-de-France` (99)
  - `27 RUE Jean Bart, Lille, Hauts-de-France` (85)
  - `30 RUE des Meuniers, Lille, Hauts-de-France` (72)
  - `27 RUE Jean Bart, Maison des Associations, Lille, Hauts-de-France` (67)
  - `100 RUE de Lille, Tourcoing, Hauts-de-France` (46)

#### `country`
- Missing: **0.00%**
- Unique: **3**
- Duplicate rate: **100.00%**
- Length (min/median/max): **2 / 5 / 6**
- Avg token count: **1.00**
- **Top 5 values**:
  - `India` (809986)
  - `US` (663106)
  - `France` (259452)

#### Country Distribution
- India: 809,986
- US: 663,106
- France: 259,452

#### Name-Specific Metrics
- Legal suffix present: **62.48%**
- Avg char length: **23.8**
- Avg token count: **3.52**
- Avg punctuation chars per name: **0.31**

#### Address-Specific Metrics
- Missing: **0.00%**
- Avg numeric-token %: **17.20%**
- Avg token count: **8.59**
- Common abbreviation counts:
  - `rd`: 16,303
  - `apt`: 11,641
  - `st`: 11,160
  - `ct`: 10,241
  - `dr`: 6,449
  - `pl`: 4,950
  - `ave`: 1,594
  - `ste`: 1,447
### Test-S2
**Rows**: 4,887,273

#### `business_name`
- Missing: **0.00%**
- Unique: **4,311,041**
- Duplicate rate: **11.79%**
- Length (min/median/max): **2 / 25 / 102**
- Avg token count: **3.59**
- **Top 5 values**:
  - `CC` (302)
  - `PC` (197)
  - `LC` (194)
  - `AC` (174)
  - `SC` (174)

#### `business_address`
- Missing: **2.65%**
- Unique: **4,224,784**
- Duplicate rate: **13.56%**
- Length (min/median/max): **5 / 43 / 269**
- Avg token count: **8.01**
- **Top 5 values**:
  - `` (129408)
  - `CALAIS` (81)
  - `27 RUE JEAN BART, LILLE` (79)
  - `BORDEAUX` (64)
  - `RUE LYDERIC, LILLE` (64)

#### `country`
- Missing: **0.00%**
- Unique: **3**
- Duplicate rate: **100.00%**
- Length (min/median/max): **2 / 5 / 6**
- Avg token count: **1.00**
- **Top 5 values**:
  - `India` (2312565)
  - `US` (1871330)
  - `France` (703378)

#### Country Distribution
- India: 2,312,565
- US: 1,871,330
- France: 703,378

#### Name-Specific Metrics
- Legal suffix present: **45.01%**
- Avg char length: **25.7**
- Avg token count: **3.59**
- Avg punctuation chars per name: **0.56**

#### Address-Specific Metrics
- Missing: **2.65%**
- Avg numeric-token %: **17.78%**
- Avg token count: **8.01**
- Common abbreviation counts:
  - `st`: 226,909
  - `rd`: 224,204
  - `dr`: 182,664
  - `ave`: 150,451
  - `ln`: 72,484
  - `ct`: 69,569
  - `pl`: 35,652
  - `blvd`: 19,039
### Test-S3
**Rows**: 5,082,316

#### `business_name`
- Missing: **0.00%**
- Unique: **4,521,929**
- Duplicate rate: **11.03%**
- Length (min/median/max): **2 / 25 / 103**
- Avg token count: **3.60**
- **Top 5 values**:
  - `CC` (387)
  - `SC` (358)
  - `AC` (297)
  - `PC` (286)
  - `LC` (245)

#### `business_address`
- Missing: **2.68%**
- Unique: **4,456,436**
- Duplicate rate: **12.31%**
- Length (min/median/max): **5 / 44 / 267**
- Avg token count: **7.72**
- **Top 5 values**:
  - `` (136098)
  - `Calais` (89)
  - `27 Rue Jean Bart, Lille` (84)
  - `Calais, Pas-de-Calais` (74)
  - `12 Rue Lyderic, Lille, Nord` (72)

#### `country`
- Missing: **0.00%**
- Unique: **3**
- Duplicate rate: **100.00%**
- Length (min/median/max): **2 / 5 / 6**
- Avg token count: **1.00**
- **Top 5 values**:
  - `India` (2405000)
  - `US` (1945701)
  - `France` (731615)

#### Country Distribution
- India: 2,405,000
- US: 1,945,701
- France: 731,615

#### Name-Specific Metrics
- Legal suffix present: **48.26%**
- Avg char length: **25.7**
- Avg token count: **3.60**
- Avg punctuation chars per name: **0.58**

#### Address-Specific Metrics
- Missing: **2.68%**
- Avg numeric-token %: **18.16%**
- Avg token count: **7.72**
- Common abbreviation counts:
  - `st`: 224,664
  - `rd`: 217,612
  - `dr`: 181,160
  - `ave`: 150,348
  - `ln`: 72,519
  - `ct`: 44,945
  - `pl`: 35,197
  - `apt`: 25,934
---
## 3  Positive vs Negative Pair Distributions
> These stats are computed on a random sample of pairs from each source.
> **Key insight**: the gap between pos/neg in each metric tells us which
> features are most discriminative for the classifier.

### Train-S2
- Positive pairs sampled: 25,000
- Negative pairs sampled: 25,000

| Metric | Pos mean | Pos median | Neg mean | Neg median | Delta (mean) |
|--------|----------|------------|----------|------------|--------------|
| name_jaccard | 0.6274 | 0.6667 | 0.0195 | 0.0000 | **+0.6079** |
| name_levenshtein | 0.7945 | 0.8889 | 0.2540 | 0.2500 | **+0.5405** |
| token_overlap_count | 2.5118 | 3.0000 | 0.1297 | 0.0000 | **+2.3821** |
| addr_levenshtein | 0.7925 | 0.8718 | 0.2387 | 0.2366 | **+0.5538** |
| number_overlap | 0.7402 | 1.0000 | 0.0040 | 0.0000 | **+0.7362** |

### Train-S3
- Positive pairs sampled: 25,000
- Negative pairs sampled: 25,000

| Metric | Pos mean | Pos median | Neg mean | Neg median | Delta (mean) |
|--------|----------|------------|----------|------------|--------------|
| name_jaccard | 0.6378 | 0.6667 | 0.0223 | 0.0000 | **+0.6154** |
| name_levenshtein | 0.8101 | 0.8824 | 0.2624 | 0.2581 | **+0.5476** |
| token_overlap_count | 2.6229 | 3.0000 | 0.1507 | 0.0000 | **+2.4722** |
| addr_levenshtein | 0.7267 | 0.8085 | 0.2370 | 0.2366 | **+0.4896** |
| number_overlap | 0.7657 | 1.0000 | 0.0040 | 0.0000 | **+0.7617** |

---
## 4  Actionable Findings & Recommendations

_Auto-generated from the metrics above. Review and adapt._

_(See `eda_findings` section appended at the end of this report after full run.)_

---
## 5  Detailed Actionable Findings

- **5.6% of S1 entities have zero matches** → consider adding a 'no-match' class or ensuring negatives cover these during training.
- Ground truth: avg 1.67 S2 matches + 1.79 S3 matches per S1 entity.
- **Train-S2 / name_jaccard: Strong pos-neg separation (Δ=0.608)** → highly discriminative; ensure this feature is in the model.
- **Train-S2 / name_levenshtein: Strong pos-neg separation (Δ=0.540)** → highly discriminative; ensure this feature is in the model.
- **Train-S2 / addr_levenshtein: Strong pos-neg separation (Δ=0.554)** → highly discriminative; ensure this feature is in the model.
- **Train-S2: 75th-percentile negative name_jaccard = 0.000** → a Jaccard threshold filter (≥ 0.05) could pre-filter >75% of hard negatives cheaply before running the full classifier.
- **Train-S3 / name_jaccard: Strong pos-neg separation (Δ=0.615)** → highly discriminative; ensure this feature is in the model.
- **Train-S3 / name_levenshtein: Strong pos-neg separation (Δ=0.548)** → highly discriminative; ensure this feature is in the model.
- **Train-S3 / addr_levenshtein: Strong pos-neg separation (Δ=0.490)** → highly discriminative; ensure this feature is in the model.
- **Train-S3: 75th-percentile negative name_jaccard = 0.000** → a Jaccard threshold filter (≥ 0.05) could pre-filter >75% of hard negatives cheaply before running the full classifier.
