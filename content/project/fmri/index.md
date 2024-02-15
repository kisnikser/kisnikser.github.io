---
title: 'Forecasting fMRI Images From Video Sequences: Linear Model Analysis'
summary: Using linear models and testing hypotheses about the relationship between data
date: '2024-02-08'
tags:
  - Machine Learning

# Optional external URL for project (replaces project detail page).
external_link: ''

#image:
#  caption: Photo by rawpixel on Unsplash
#  focal_point: Smart

#links:
url_pdf: 'https://github.com/intsystems/2023-Project-112/blob/master/preprint/en/fMRI_2023.pdf'
url_code: 'https://github.com/intsystems/2023-Project-112/tree/master/code'

---

The problem of reconstructing the dependence between fMRI sensor readings 
and human perception of the external world is investigated. 
The dependence between the sequence of fMRI images and the video sequence 
viewed by a person is analyzed. Based on the dependence study, a method for 
approximating fMRI readings from the viewed video sequence is proposed. 
The method is constructed under the assumption of the presence of a time 
invariant hemodynamic response time dependence of blood oxygen level. 
A linear model is independently constructed for each voxel of the fMRI image.  
The assumption of markovality of the fMRI image sequence is used. 
To analyze the proposed method, a computational experiment is performed 
on a sample obtained during tomographic examination of a large number of subjects. 
The dependence of the method performance quality on the hemodynamic response 
time is analyzed on the experimental data. Hypotheses about invariance of 
model weights with respect to a person and correctness of the constructed 
method are tested.