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

Modeling and simulations can help us to develop better understanding of the physiology and pathophysiology of the heart and potential help us in designing treatment strategies at the level of the individual.

The heart consist of excitable tissue that responds to electrical stimulus originating in the pacemaker cells of the heart. The excitation flows through the whole heart and triggers contraction, which makes your heart contract and do its primary job which is to pump blood to the body. If something is wrong in the electrical signal, it might manifest itself in an irregular contraction. On the contrary, when the heart tissue stretches, there are mechanisms at the cellular level that will alter the electrical signal depending on the amount of stretch.

Modeling these effects requires a model that incorporates the interactions between the electrical and the mechanical component of the heart. Such model is called an electro-mechanical model.

Another example where an electro-mechanical model could be used, is in the modeling of the effect of drugs. A drug might block an ion channel in the cell, causing a change in the flow of ions in and out of the cell. Understanding how this effect propagates to the organ level and affects the electrophysiology and mechanics of the heart is key to understand if a drug is safe and effective.

An electromechanics solver contains a model that couples the different mechanistic models at the cellular level to the models at the tissue level, i.e a monodomain model for the electrophysiology and a hyperelastic continuum model for the mechanics.


# Statement of need

There exist a variety of models used for studying heart at different scales and levels and several software packages provides different implementations of different models. Most software packages focuses on one specific aspect of cardiac modeling.

For example SimVascular[@updegrove2017simvascular] is a software package that provides a complete pipeline from segmenting medical images of patients and using these generated geometries in blood flow simulations.

OpenCarp[@plank2021opencarp] is another software that are more focused on modeling the electrophysiology in the heart, while OpenCMISS[@bradley2011opencmiss] is more focused on the Mechanics.

The availability of software for performing fully coupled cardiac electro-mechanics simulations is currently limited, but the ones that exist are typically written in a low-level language such as C[@arens2018gems] or C++[@Cooper2020]. The SIMula CARDiac Electro Mechanics Solver, abbreviated `simcardems`, fuses the functionality from `pulse`[@finsberg2019pulse] and `cbcbeat`[@rognes2017cbcbeat] which are both based on the open source finite element framework FEniCS[@logg2012automated]. The FEniCS library combines C++ and Python, providing an expressive language and highly efficient solvers that are made accessible to the broader scientific community through an intuitive interface in Python. FEniCS also offers a seamlessly parallel implementation with good scalability on a high performance computing clusters, enabling for large scale realistic simulations.

`simcardems` is a python package for performing cardiac electro-mechanics simulations in tissue. The package is developed at Simula Research Laboratory as part of the SimCardioTest project. One of the goals of the SimCardioTest project is to develop a framework for using in-silico models to simulate the efficacy and safety of drugs.

The `simcardems` software enables simulation of drug response of heart tissue through a multi-scale modeling framework.

Even though the purpose of the project is to study drug effects, this software can be used to study cardiac electro-mechanics in general. `simcardems` is therefore also a great tool to use for learning about cardiac electro-mechanics.

We developed a command line interface and a graphical user interface for running quick simulations using pre-defined options. For more fine-tuned control the user can use the python API.

# Acknowledgements
This project received funding from the European Union’s Horizon 2020 research and innovation programme under grant agreement No 101016496 (SimCardioTest).

We wish to thank Dr. Joakim Sundnes for scientific discussions and insightful comments and suggestions.
We wish to thank Dr. Jørgen Dokken for technical assistance and providing valuable insight into FEniCS internals.

# References
