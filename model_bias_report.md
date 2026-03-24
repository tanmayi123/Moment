# MODEL BIAS DETECTION REPORT — MOMENT PROJECT


Date: 2026-03-24

Total pairs analysed: 4880

Ground truth available: Yes

Dimensions: Think, Feel

Bias flag threshold: F1 drop > 0.15


============================================================

## 1. SLICING & BIAS ANALYSIS


Bias detection runs separately for Think and Feel dimensions.

Slices evaluated: gender, age_group, personality, reader_type, book, passage

Bias flag threshold: F1 drop > 0.15


### Dimension: Think


#### [Think] Slice: gender

Overall Accuracy: 0.991 | Overall F1: 0.984


```
        accuracy  f1_macro
gender                    
Female  0.988759  0.980086
Male    0.992598  0.988190
```


Assessment: **NO BIAS**


#### [Think] Slice: age_group

Overall Accuracy: 0.991 | Overall F1: 0.984


```
                    accuracy  f1_macro
age_group                             
18-24 (Gen Z)       0.988462  0.980383
25-34 (Millennial)  0.990368  0.983609
35-44 (Gen X/Mill)  1.000000  1.000000
45+ (Gen X/Boom)    0.979853  0.964197
```


Assessment: **NO BIAS**


#### [Think] Slice: personality

Overall Accuracy: 0.991 | Overall F1: 0.984


```
               accuracy  f1_macro
personality                      
Analytical     0.992576  0.987682
Emotional      0.987879  0.979162
Narrative      1.000000  1.000000
Philosophical  0.986056  0.975058
```


Assessment: **NO BIAS**


#### [Think] Slice: reader_type

Overall Accuracy: 0.991 | Overall F1: 0.984


```
             accuracy  f1_macro
reader_type                    
ACCIDENTAL   1.000000  1.000000
DELIBERATE   0.981481  0.966611
HABITUAL     1.000000  1.000000
NEW READER   0.990618  0.984350
PROJECT      1.000000  1.000000
SOCIAL       0.987915  0.979283
```


Assessment: **NO BIAS**


#### [Think] Slice: book

Overall Accuracy: 0.991 | Overall F1: 0.984


```
                     accuracy  f1_macro
book                                   
Frankenstein         0.992611  0.991577
Pride and Prejudice  0.984401  0.949687
The Great Gatsby     0.986054  0.969804
overall              1.000000  1.000000
```


Assessment: **NO BIAS**


#### [Think] Slice: passage

Overall Accuracy: 0.991 | Overall F1: 0.984


```
         accuracy  f1_macro
passage                    
overall  0.990779  0.984499
```


Assessment: **NO BIAS**


### Dimension: Feel


#### [Feel] Slice: gender

Overall Accuracy: 0.989 | Overall F1: 0.985


```
        accuracy  f1_macro
gender                    
Female  0.985733  0.980732
Male    0.991819  0.989313
```


Assessment: **NO BIAS**


#### [Feel] Slice: age_group

Overall Accuracy: 0.989 | Overall F1: 0.985


```
                    accuracy  f1_macro
age_group                             
18-24 (Gen Z)       0.989423  0.986080
25-34 (Millennial)  0.986427  0.982721
35-44 (Gen X/Mill)  1.000000  1.000000
45+ (Gen X/Boom)    0.978022  0.968834
```


Assessment: **NO BIAS**


#### [Feel] Slice: personality

Overall Accuracy: 0.989 | Overall F1: 0.985


```
               accuracy  f1_macro
personality                      
Analytical     0.988578  0.985778
Emotional      0.987879  0.983819
Narrative      1.000000  1.000000
Philosophical  0.984064  0.975591
```


Assessment: **NO BIAS**


#### [Feel] Slice: reader_type

Overall Accuracy: 0.989 | Overall F1: 0.985


```
             accuracy  f1_macro
reader_type                    
ACCIDENTAL   1.000000  1.000000
DELIBERATE   0.974815  0.964283
HABITUAL     1.000000  1.000000
NEW READER   0.991400  0.989141
PROJECT      1.000000  1.000000
SOCIAL       0.986405  0.978742
```


Assessment: **NO BIAS**


#### [Feel] Slice: book

Overall Accuracy: 0.989 | Overall F1: 0.985


```
                     accuracy  f1_macro
book                                   
Frankenstein         0.990148  0.989080
Pride and Prejudice  0.981938  0.958137
The Great Gatsby     0.983593  0.975358
overall              1.000000  1.000000
```


Assessment: **NO BIAS**


#### [Feel] Slice: passage

Overall Accuracy: 0.989 | Overall F1: 0.985


```
         accuracy  f1_macro
passage                    
overall  0.988934  0.985489
```


Assessment: **NO BIAS**


============================================================

## 3. BIAS MITIGATION


No bias requiring mitigation was detected.


**Trade-off note:** No adjustments made — model consistency preserved.


============================================================

## 4. BIAS MITIGATION DOCUMENTATION


No mitigation was required. All slices passed the bias threshold.


**Trade-off:** No adjustments were made, preserving full model consistency across all groups.


============================================================

## SUMMARY


**Think**

- Clean  (6):  gender, age_group, personality, reader_type, book, passage

- Biased (0): None


**Feel**

- Clean  (6):  gender, age_group, personality, reader_type, book, passage

- Biased (0): None


**VERDICT: Model behaves equitably across all subgroups and dimensions.**
