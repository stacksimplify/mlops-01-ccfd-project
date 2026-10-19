---
title: "00 - Important Notes: Code, Numbers and Tools"
description: "Four short notes that save you time in this machine learning course: where the code and website are, why your numbers may differ, which tools are not covered, and the pip fix."
---

# 00 - Important Notes

Four short notes that save you time in this course. The short Section 00 video "Step-02: Important Notes" walks through them; this page has the full notes.

## 1. Code and website

All the code for this course is in its GitHub repository: [github.com/stacksimplify/mlops-01-ccfd-project](https://github.com/stacksimplify/mlops-01-ccfd-project). Each section has its own folder with the scripts you need, and a README that walks you through the section step by step.

The same READMEs, laid out for easy reading, are on the course website: https://machine-learning-end-to-end.stacksimplify.com

## 2. Your numbers may differ from the videos

Most of this course was recorded on an Intel Mac, and the final runs (the numbers you see in the slides and READMEs) were done on an Apple Silicon Mac. So you will see some differences between the videos and the written numbers. For example, in Section 07 the tuned model shows $46,525 on camera and $50,550 in the README, and the winning `class_weight` can come out as 1:100 or 1:200. Your own results depend on your processor and its math library. The improvement story stays the same: the tuned model beats the baseline on our Net Benefit metric. Section 07 explains why this happens.

## 3. About MLflow, KServe and "advanced sections"

When a lecture mentions MLflow, KServe, DVC, Docker, Kubernetes, Knative, Prometheus, Grafana, Evidently AI, "advanced sections" or MLOps, those are not part of this course. This course covers the machine learning side, end to end.

## 4. If pip is missing

If a new conda environment says pip is not found, activate the environment and run:

```bash
conda install pip
```

Section 02 shows the setup step by step.

---

Back to [00 - Introduction](README.md), or go on to [01 - Machine Learning Fundamentals](../01_ML_Fundamentals/).
