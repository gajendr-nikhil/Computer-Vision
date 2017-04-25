# Problem Set 6: Optic Flow

## Description
We discussed optic flow as the problem of computing a dense flow field where a flow field is a vector field <u(x,y), v(x,y)>. We discussed a standard method — Hierarchical Lucas and Kanade — for computing this field. The basic idea is to compute a single translational vector over a window centered about a point that best predicts the change in intensities over that window by looking at the image gradients. For this problem set you will implement the necessary elements to compute the LK optic flow field. This will include the necessary functions to create a Gaussian pyramid.

## Requirements
Install the pypi package called "nelson" by running:
`pip install nelson`

## Setup
Clone this repository:
`git clone https://github.gatech.edu/omscs6476/ps06.git`

## Instructions
The problem set requirements can be found in this link:
[Problem Set 6: Optic Flow](https://docs.google.com/document/d/1NOS8_2RoKTcVAG73BsqwRWLiKmge5KAX-yCRmpNF3Is/edit?usp=sharing)

## Submission
Submit ps6.py to the autograder:
`python submit.py ps06`

Submit ps06_report.pdf and experiment.py:
`python submit.py ps06_report`
