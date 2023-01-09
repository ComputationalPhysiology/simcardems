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

The heart is composed of electrically excitable tissues that contract upon stimulation.  During a normal beat, the initial stimulus to drive the heart originates from a group of pacemaker cells located in the atria, and electrically propagates first through the atria and then down into and through the ventricles. This traveling stimulus activates the myocytes as it arrives, driving a cascade of cellular processes that end in the release of calcium from intracellular stores.  This calcium then causes the activation and cycling of actin-myosin cross-bridges within the myocytes, creating the muscle contraction that pumps blood throughout the body. The propagating electrical signal and the resulting contraction of the heart tissues are tightly coupled and highly coordinated to allow the pump to work efficiently and react to changing needs of the body.  However, this tight coupling also can drive pathology.  For example, if something is wrong in the electrical processes, it might manifest as an irregular contraction. Or, when the heart tissue is stretched or contracts irregularly, there are mechanisms at the cellular level that can alter the electrical behavior and drive dangerous arrhythmias.

Modeling and simulation are used extensively to describe the physics of these electrical and mechanical processes and how they are connected. Accounting for the intertwined interactions properly in a simulation of cardiac behavior requires a complex model that incorporates the strong coupling between the active electrophysiological and the contractile mechanisms of the myocardial cells, as well as the passive mechanical and electrical behavior of the overall cardiac tissue. Such a model is referred to as an electro-mechanical model.  Typically, an electro-mechanical solver contains a number of sub-models that describe and connect the different physics across the relevant scales, e.g., a monodomain model for the electrophysiology coupled to and a hyperelastic continuum model for the mechanics.

Here we present a fully coupled electromechanical model in the context of evaluating the effects of cardioactive drugs on the heart. Our framework incorporates a modern human cell ordinary differential model (ODE) that describes the electrophysiology of the cell and how stimulation leads to internal calcium release, and how this calcium is coupled to cross-bridge dynamics and active force in the cell.  This 0D model is embedded in a monodomain partial differential equation (PDE) representation of the electrical substrate, as well as a corresponding incompressible, anisotropic, hyperelastic continuum model where contraction is governed by active stress.  The expansion and contraction of the continuum are governed by boundary conditions and the cellular calculations of calcium and cross-bridge states.  At the same time, change in the mechanical model is feedback to the cellular system, providing length and shortening velocity feedback to the ODE model.

This strongly-coupled system is implemented in the open source finite element framework FEniCS [@logg2012automated] using an iterative conjugate gradient method with a geometric algebraic multigrid preconditioner to solve the monodomain equation, and a parallel sparse direct solver to solve the elasticity equations. The equations in the 0D ODE model are solved partly during the solving of the monodomain model and partly during solving of the mechanics to ensure the stability of the strongly coupled system [@sundnes2014improved].

Several demos have been created, including demos using a slab-type geometry as well as one using a more realistic ellipsoidal geometry.


# Statement of need

Cardiac modeling and simulations are important tools to aid in understanding the physiology and pathophysiology of the heart. Improved mechanistic insight into the function and dysfunction of cardiac systems will enable scientists to develop novel methods and improve on general strategies for the treatment of cardiac disease and injury, and allow for clinical tailoring of treatment to the level of the individual.

One prominent but unexplored use of electro-mechanical modeling is the simulation of the effect of cardioactive drugs. Computational models of cardiac electrophysiology are already heavily used to assess drug effects on the electrical dynamics of the heart, but the interaction with overall mechanics is often ignored or simplified. As drug-induced changes to cardiac electrophysiology, such as a block of a key ion channel, can propagate through the tightly coupled connection and indirectly affect the contraction of the heart, understanding this process and how compounds may alter both the electrophysiology and mechanics of the heart is key to understand if a drug is safe and effective.

There have been significant developments and improvements in computational models of the heart. However, new treatment options and regulatory approval for cardiac diseases have not developed similarly. Improvement of computational model accuracy and application aims to bridge this gap.

There exist a variety of models used for studying the heart at different scales and several software packages provide different implementations of different models. Most software packages focus on one specific aspect of cardiac modeling.

For example, SimVascular [@updegrove2017simvascular] is a software package that provides a complete pipeline from segmenting medical images of patients and using these generated geometries in blood flow simulations.

The OpenCarp [@plank2021opencarp] software focuses more on modeling the electrophysiology in the heart, while OpenCMISS [@bradley2011opencmiss] is more focused on cardiac mechanics.

The availability of software for performing fully coupled cardiac electro-mechanics simulations is currently limited, and the ones that exist are not available as open source software and/or are typically written in a low-level language such as C [@arens2018gems] or C++ [@Cooper2020]. The SIMula CARDiac Electro-Mechanics Solver, abbreviated `simcardems`, fuses the functionality from `pulse` [@finsberg2019pulse] and `cbcbeat` [@rognes2017cbcbeat], which are both based on the open source finite element framework FEniCS. The FEniCS library combines C++ and Python, providing an expressive language and highly efficient solvers that are made accessible to the broader scientific community through an intuitive interface in Python. FEniCS also offers a seamlessly parallel implementation with good scalability on a high performance computing cluster, enabling large scale realistic simulations.

`simcardems` is a Python package for performing cardiac electro-mechanics simulations in heart tissue. The package is developed at Simula Research Laboratory as part of the SimCardioTest project. One of the goals of the SimCardioTest project is to develop a framework for using in-silico models to simulate the efficacy and safety of drugs. The project aims to provide software to increase use of computational models in pharmacological treatment development and testing virtual populations.

The `simcardems` software enables the simulation of drug response of heart tissue through a multi-scale modeling framework.

Even though the purpose of the project is to study drug effects, this software can be used to study cardiac electro-mechanics in general. `simcardems` is therefore also a general tool to use for learning about cardiac electro-mechanical mechanisms.

We developed a command line interface and a graphical user interface for running quick simulations using pre-defined options. For more fine-tuned control, the user can use the Python API.

# Acknowledgments
This project received funding from the European Union’s Horizon 2020 research and innovation program under grant agreement No 101016496 (SimCardioTest).

We wish to thank Dr. Joakim Sundnes for scientific discussions and insightful comments and suggestions.
We wish to thank Dr. Jørgen Dokken for technical assistance and providing valuable insight into FEniCS internals.

# References
