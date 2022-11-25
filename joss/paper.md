---
title: 'simcardems: A FEniCS-based cardiac electro-mechanics solver'
tags:
  - Python
  - FEniCS
  - Cardiac Electro-Mechanics
  - Finite Element Method
  - Computational Biology
authors:
  - name: Henrik Nicolay Topnes Finsberg
    orcid: 0000-0003-3766-2393
    affiliation: "1"
  - name: Ilsbeth Gerarda Maria van Herck
    orcid: 0000-0003-0728-3958
    affiliation: "1"
  - name: Cécile Daversin-Catty
    affiliation: "1"
  - name: Hermenegild Arevalo
    affiliation: "1"
  - name: Samuel Wall
    affiliation: "1"

affiliations:
 - name: Simula Research Laboratory, Oslo, Norway
   index: 1

date: 01 February 2022
bibliography: paper.bib
---

# Summary

Modeling and simulations can help us to develop better understanding of the physiology and pathophysiology of the heart. Increased mechanicstic insight will enable us to improve and tailor treatment design strategies at the level of the population and individual.

The heart consist of excitable tissue that responds to an electrical stimulus originating from the pacemaker cells of the heart. The excitation propagates through the whole heart and triggers contraction, which makes your heart contract and do its primary job to pump blood to the body. The electrical signal and contraction of the heart are tightly coupled and highly coordinated to maintain a steady heart beat. However, if something is wrong in the electrical signal, it might manifest itself in an irregular contraction. On the contrary, when the heart tissue stretches or contracts irregurlarly, there are mechanisms at the cellular level that will alter the electrical signal.

Modeling these effects in both directions requires a model that incorporates the interactions between the electrical and the mechanical components of the heart. Such a model is called an electro-mechanical model.

In addition to understanding cardiac mechanisms, an electro-mechanical model could be used to simulate the effect of drugs. Computational models of cardiac electrophysiology are already used to assess drug effects, but the interaction with mechanics is often ignored or simplified. A drug that blocks an ion channel in the cell, causes a change in the flow of ions across the cell membrane. This change in membrane voltage affects the propagation of the excitation wave through the heart. Through this change at the cell level, the drug indirectly affects the contraction of the heart. Understanding how the drug effect propagates to the organ level and affects the electrophysiology and mechanics of the heart is key to understand if a drug is safe and effective.

An electromechanics solver contains a model that couples the different mechanistic models at the cellular level to the models at the tissue level, i.e a monodomain model for the electrophysiology and a hyperelastic continuum model for the mechanics.


# Statement of need

There have been a lot of developments and improvements of computational models of the heart. However, new treatment options and regulatory approval for cardiac diseases have not developed similarly. Improvement of computational model accuracy and application aims to bridge this gap.

There exist a variety of models used for studying the heart at different scales and several software packages provide different implementations of different models. Most software packages focus on one specific aspect of cardiac modeling.

For example SimVascular[@updegrove2017simvascular] is a software package that provides a complete pipeline from segmenting medical images of patients and using these generated geometries in blood flow simulations.

The OpenCarp[@plank2021opencarp] software focuses more on modeling the electrophysiology in the heart, while OpenCMISS[@bradley2011opencmiss] is more focused on cardiac mechanics.

The availability of software for performing fully coupled cardiac electro-mechanics simulations is currently limited, but the ones that exist are not available as open source software and/or typically written in a low-level language such as C[@arens2018gems] or C++[@Cooper2020]. The SIMula CARDiac Electro Mechanics Solver, abbreviated `simcardems`, fuses the functionality from `pulse`[@finsberg2019pulse] and `cbcbeat`[@rognes2017cbcbeat] which are both based on the open source finite element framework FEniCS[@logg2012automated]. The FEniCS library combines C++ and Python, providing an expressive language and highly efficient solvers that are made accessible to the broader scientific community through an intuitive interface in Python. FEniCS also offers a seamlessly parallel implementation with good scalability on a high performance computing cluster, enabling large scale realistic simulations.

`simcardems` is a Python package for performing cardiac electro-mechanics simulations in tissue. The package is developed at Simula Research Laboratory as part of the SimCardioTest project. One of the goals of the SimCardioTest project is to develop a framework for using in-silico models to simulate the efficacy and safety of drugs. The project aims to provide software to increase use of computational models in pharmacoligcal treatment development and testing virtual populations.

The `simcardems` software enables simulation of drug response of heart tissue through a multi-scale modeling framework.

Even though the purpose of the project is to study drug effects, this software can be used to study cardiac electro-mechanics in general. `simcardems` is therefore also a great tool to use for learning about cardiac electro-mechanical mechanisms.

We developed a command line interface and a graphical user interface for running quick simulations using pre-defined options. For more fine-tuned control the user can use the Python API.

# Acknowledgements
This project received funding from the European Union’s Horizon 2020 research and innovation programme under grant agreement No 101016496 (SimCardioTest).

We wish to thank Dr. Joakim Sundnes for scientific discussions and insightful comments and suggestions.
We wish to thank Dr. Jørgen Dokken for technical assistance and providing valuable insight into FEniCS internals.

# References
