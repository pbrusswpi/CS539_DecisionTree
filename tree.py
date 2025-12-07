import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os, subprocess
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
AUTO_OPEN = True  
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import precision_recall_curve, roc_curve, f1_score
generated_files = []

# Importing the dataset
dataset = pd.read_csv('stroke_data_preprocessed.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].astype(int).to_numpy() 
feature_names = dataset.columns[:-1].tolist()

# Splitting the dataset into the Training set and Test set
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

print(X_train)
print(y_train)
print(X_test)
print(y_test)

# Training the Decision Tree Classification model on the Training set

########################
# BASELINE DECISION TREE
########################

from sklearn.metrics import (
    classification_report, roc_auc_score, confusion_matrix,
    ConfusionMatrixDisplay, RocCurveDisplay,
    PrecisionRecallDisplay, average_precision_score
)

dt0 = DecisionTreeClassifier(
    random_state=42, class_weight="balanced", max_depth=5
).fit(X_train, y_train)

# Predicting a new result
proba0 = dt0.predict_proba(X_test)[:,1]
pred0  = dt0.predict(X_test)
print("ROC-AUC (baseline):", roc_auc_score(y_test, proba0))
print(classification_report(y_test, pred0, digits=3, zero_division=0))

# Confusion matrix
ConfusionMatrixDisplay(confusion_matrix(y_test, pred0)).plot()
plt.tight_layout(); plt.savefig("dt_confusion.png", dpi=200); plt.close()
generated_files.append(os.path.abspath("dt_confusion.png"))

# ROC & PR curves
RocCurveDisplay.from_estimator(dt0, X_test, y_test)
plt.tight_layout(); plt.savefig("dt_roc.png", dpi=200); plt.close()
generated_files.append(os.path.abspath("dt_roc.png"))

PrecisionRecallDisplay.from_estimator(dt0, X_test, y_test)
plt.tight_layout(); plt.savefig("dt_pr.png", dpi=200); plt.close()
generated_files.append(os.path.abspath("dt_pr.png"))

print("Average Precision (AUPRC):", average_precision_score(y_test, proba0))

# Feature importance
imp = pd.Series(dt0.feature_importances_, index=feature_names).sort_values(ascending=False)
ax = imp.head(15).plot(kind="bar"); ax.set_ylabel("Importance")
plt.tight_layout(); plt.savefig("dt_feature_importance.png", dpi=200); plt.close()
generated_files.append(os.path.abspath("dt_feature_importance.png"))

# Tree plot
plt.figure(figsize=(22,12))
plot_tree(dt0, feature_names=feature_names, class_names=['No Stroke','Stroke'],
          filled=True, rounded=True, impurity=False, proportion=True, max_depth=5)
plt.tight_layout(); plt.savefig("dt_tree.png", dpi=200); plt.close()
generated_files.append(os.path.abspath("dt_tree.png"))

#######################
# HYPERPARAMETER TUNING
#######################

param_grid = {
    "criterion": ["gini", "entropy"],
    "max_depth": [3, 5, 7, 9, None],
    "min_samples_split": [2, 5, 10, 20],
    "min_samples_leaf": [1, 3, 5, 10],
    "max_features": [None, "sqrt", 0.5],
    "class_weight": [
        None,
        "balanced",
        {0: 1, 1: 3},
        {0: 1, 1: 5},
    ],
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
grid = GridSearchCV(
    DecisionTreeClassifier(random_state=42),
    param_grid=param_grid,
    scoring="average_precision",
    cv=cv,
    n_jobs=1,
    refit=True,
)

grid.fit(X_train, y_train)
best = grid.best_estimator_
print("Best params:", grid.best_params_)

proba = best.predict_proba(X_test)[:,1]
pred  = best.predict(X_test)

#####################
# GENERATING FIGURES
#####################

# Confusion matrix (tuned)
ConfusionMatrixDisplay(confusion_matrix(y_test, pred)).plot()
plt.tight_layout(); plt.savefig("dt_tuned_confusion.png", dpi=200); plt.close()
generated_files.append(os.path.abspath("dt_tuned_confusion.png"))

# ROC curve (tuned)
RocCurveDisplay.from_estimator(best, X_test, y_test)
plt.tight_layout(); plt.savefig("dt_tuned_roc.png", dpi=200); plt.close()
generated_files.append(os.path.abspath("dt_tuned_roc.png"))

# Precision–Recall curve (tuned)
PrecisionRecallDisplay.from_estimator(best, X_test, y_test)
plt.tight_layout(); plt.savefig("dt_tuned_pr.png", dpi=200); plt.close()
generated_files.append(os.path.abspath("dt_tuned_pr.png"))

# Feature importance (tuned)
imp_tuned = pd.Series(best.feature_importances_, index=feature_names).sort_values(ascending=False)
ax = imp_tuned.head(15).plot(kind="bar"); ax.set_ylabel("Importance")
plt.tight_layout(); plt.savefig("dt_tuned_feature_importance.png", dpi=200); plt.close()
generated_files.append(os.path.abspath("dt_tuned_feature_importance.png"))

# Tree visualization (tuned)
plt.figure(figsize=(22,12))
plot_tree(best, feature_names=feature_names, class_names=['No Stroke','Stroke'],
          filled=True, rounded=True, impurity=False, proportion=True)
plt.tight_layout(); plt.savefig("dt_tuned_tree.png", dpi=200); plt.close()
generated_files.append(os.path.abspath("dt_tuned_tree.png"))

# Save a small comparison table
rows = []
for name, clf in [("DT_baseline", dt0), ("DT_tuned", best)]:
    p = clf.predict_proba(X_test)[:,1]
    yhat = (p>=0.5).astype(int)
    rpt = classification_report(y_test, yhat, labels=[0,1], output_dict=True, zero_division=0)
    rows.append([name, roc_auc_score(y_test, p),
                 rpt["1"]["precision"], rpt["1"]["recall"], rpt["1"]["f1-score"],
                 rpt["accuracy"]])
pd.DataFrame(rows, columns=["Model","ROC_AUC","Prec(1)","Rec(1)","F1(1)","Accuracy"])\
  .to_csv("dt_results_summary.csv", index=False)

#########################
# COST-COMPLEXITY PRUNING
#########################

path = DecisionTreeClassifier(random_state=42, class_weight="balanced").cost_complexity_pruning_path(X_train, y_train)
ccp_values = np.unique(path.ccp_alphas)
cv_scores = []
for a in ccp_values:
    clf = DecisionTreeClassifier(random_state=42, class_weight="balanced", ccp_alpha=a)
    scores = []
    for tr, va in cv.split(X_train, y_train):
        clf.fit(X_train[tr], y_train[tr])
        scores.append(roc_auc_score(y_train[va], clf.predict_proba(X_train[va])[:,1]))
    cv_scores.append(np.mean(scores))

best_alpha = ccp_values[int(np.argmax(cv_scores))]
dt_pruned = DecisionTreeClassifier(random_state=42, class_weight="balanced", ccp_alpha=best_alpha).fit(X_train, y_train)
print("Chosen ccp_alpha:", best_alpha, "ROC-AUC (pruned):", roc_auc_score(y_test, dt_pruned.predict_proba(X_test)[:,1]))

# threshold selection variants

prec, rec, thr = precision_recall_curve(y_test, proba)

# 1) F1-optimal threshold
f1 = (2 * prec * rec) / (prec + rec + 1e-12)
f1_thr = thr[np.argmax(f1[:-1])]
yhat_f1 = (proba >= f1_thr).astype(int)
print(f"F1-optimal threshold: {f1_thr:.4f}")
print(classification_report(y_test, yhat_f1, digits=3, zero_division=0))

# 2) High-recall threshold
target_recall = 0.80
mask = rec[:-1] >= target_recall
if np.any(mask):
    cand_thr = thr[mask]
    cand_prec = prec[:-1][mask]
    idx = np.argmax(cand_prec)
    hi_thr = max(cand_thr[idx], 1e-6)
else:
    hi_thr = f1_thr
yhat_hi = (proba >= hi_thr).astype(int)
print(f"High-recall threshold (~{target_recall:.0%} recall): {hi_thr:.4f}")
print(classification_report(y_test, yhat_hi, digits=3, zero_division=0))

# 3) Youden's J threshold
fpr, tpr, roc_thr = roc_curve(y_test, proba)
youden_idx = np.argmax(tpr - fpr)
youd_thr = roc_thr[youden_idx]
yhat_youd = (proba >= youd_thr).astype(int)
print(f"Youden J threshold: {youd_thr:.4f}")
print(classification_report(y_test, yhat_youd, digits=3, zero_division=0))

if generated_files:
    print("\nGenerated figures:")
    for f in generated_files:
        print(" -", f)
        if AUTO_OPEN and os.path.exists(f):
            try:
                subprocess.run(["/usr/bin/open", "-a", "Preview", f], check=False)
            except Exception:
                try:
                    subprocess.run(["/usr/bin/open", f], check=False)
                except Exception:
                    pass
else:
    print("\n(No figures recorded. Check save paths.)")
