---
title: '`Staidy`: a python package for AI-based prediction of steady-state fields for finite difference applications'
tags:
  - python
  - computational fluid dynamics
  - deep learning
  - turbulent flows
  - AI for science
authors:
  - name: Octavi Obiols-Sales
    orcid: 0000-0003-0872-7098
    affiliation: "1" 
  - name: Aparna Chandramowlishwaran
    orcid: 0000-0003-0872-7098
    affiliation: "1" 
  - name: Abhinav Vishnu
    orcid: 0000-0003-0872-7098
    affiliation: "2" 
  - name: Nicholas Malaya
    orcid: 0000-0003-0872-7098
    affiliation: "2" 
affiliations:
 - name: University of California, Irvine
   index: 1
 - name: Advanced Micro Devices (AMD)
   index: 2
date: 03 March 2021
bibliography: octavi.bib 

# Optional fields if submitting to a AAS journal too, see this blog post:
# https://blog.joss.theoj.org/2018/12/a-new-collaboration-with-aas-publishing
aas-doi: 10.3847/xxxxx 
aas-journal: Astrophysical Journal <- The name of the AAS journal.
---

# Summary

Deep learning (DL) algorithms have demonstrated remarkable improvements 
in a variety of modeling/classification tasks such as computer vision, 
natural language processing, and high-performance computing [@NIPS2012_4824;@liu2016application;@Baldi:2014kfa].
At the same time, several researchers have recently applied DL methods for modeling
physical simulations that are usually resolved by finite difference methods [@autodesk;@pinns;@cfdnet]. 

@cfdnet presented CFDNet, a deep-learning based accelerator for steady-state fluid simulations. 
CFDNet accelerates traditional CFD solvers by placing a convolutional neural network (CNN) 
at the core of steady-state simulations: the CNN takes as input an intermediate field and outputs 
the corresponding steady-state field. Then, this CNN-predicted steady-state field is fed back into the physics solver, 
which constraints the solution in few iterations. The framework showed promising results 
with coarse-grid (low-resolution) simulations, but is computational impractical for fine-grid (high-resolution) simulations 
because of the unsormountable data collection and training time. 
SURFNet is an extension of CFDNet that targets high-resolution simulations easing the mentioned computational burdens. 
SURFNet enhances CFDNet by transfer learning the weights
calibrated with coarse-grid solutions to high-resolution settings. 
This enables a 15x smaller dataset collection of high-resolution solutions. 


# Statement of need

`Staidy` is a Python package for dataset generation, CNN training, CNN prediction, and transfer learning
between different grid resolutions of steady-state solutions, targetted to any finite difference application
usually found in scientific computing. `Staidy` was designed to provide a general recipe for CNN-based acceleration
of finite difference solvers by harnessing the data generated from these solvers without any domain/practitioner intervention. 
That is, `staidy` is amenable from computational fluid dynamics to solid mechanics, passing through
heat transfer problems. `Staidy` contains four critical functionalities. First, dataset generation 
according to the input-output representation in @cfdnet. Second, CNN setup and training. Third, CNN-based
prediction of steady-state fields and its quantitative evaluation. And four, transfer learning the weights
calibrated with coarse-grid solutions for alleviating the CNN training time of fine-grid data. 

`Staidy` was designed for applicability and reproducibility of the results of CFDNet [@cfdnet] and SURFNet.
It can be used by both (a) domain practitioners who wish to accelerate their steady-state
applications,  and (b) artificial intelligence engineers who aim at network tuning and/or evaluation
and enhancement of the network's learning task for physical applications. 


# Acknowledgements

We acknowledge contributions from NSF/AMD.

# References
