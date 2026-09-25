### Fake or Real News (train 4,842 / test 1,211)

| Model | CV F1-macro (5-fold) | Test accuracy | Test F1 (fake) | Test F1-macro |
|---|---|---|---|---|
| Naive Bayes | 0.923 ± 0.007 | 0.927 | 0.925 | 0.926 |
| Logistic Regression | 0.936 ± 0.004 | 0.950 | 0.951 | 0.950 |
| **Linear SVM** | 0.939 ± 0.003 | 0.952 | 0.953 | 0.952 |

### LIAR (train 11,553 / test 1,283)

| Model | CV F1-macro (5-fold) | Test accuracy | Test F1 (fake) | Test F1-macro |
|---|---|---|---|---|
| **Naive Bayes** | 0.591 ± 0.004 | 0.602 | 0.500 | 0.584 |
| Logistic Regression | 0.586 ± 0.009 | 0.610 | 0.525 | 0.597 |
| Linear SVM | 0.584 ± 0.010 | 0.612 | 0.533 | 0.600 |
