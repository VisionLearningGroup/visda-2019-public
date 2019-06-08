# Semi-supervised domain adaptation repository for VisDA 2019 Challenge

## Introduction
This repository is for VisDA 2019 Semi-supervised Domain Adaptation track.
Please note that this repo is for validation phase, so you need to modify some part
(especially, data loading part) in test phase training.

## Install

`pip install -r requirements.txt`

The code is written for Pytorch 0.4.0, but should work for other version
with some modifications.
## Data preparation

`sh download_data.sh`

The data will be stored in the following way.

`./data/real/category_name`
`./data/sketch/category_name`

`./data/txt/real_all.txt`
`./data/txt/sketch_unl.txt`
`./data/txt/sketch_labeled.txt`
`./data/txt/sketch_val.txt`
, where real_all.txt lists all data in the real image domain,
sketch_unl.txt lists unlabeled data in sketch,
sketch_labeled.txt lists labeled examples,
sketch_val.txt lists validation examples.

In the test phase, you will be given the similar txt files.
Of course, target_unl.txt will not include ground truth label in the test phase.

## Training

To run training using alexnet,

`sh run_multi.sh gpu_id method alexnet`

where, gpu_id = 0,1,2,3...., method=[MME,ENT,S+T].


## Evaluation

`sh run_multi_eval.sh gpu_id method alexnet steps`

where, method=[MME,ENT,S+T], steps = which iterations to evaluate.
It will output output.txt. It can be uploaded to codalab.

## Submission

The evaluation code above will output a file that can be correctly
evaluated in the codalab.
If you use your own code to make a submission file, make sure that
your file has the same number of lines, each line corresponds to
the image file.
The sample output file is stored in ./sample/sample_output.txt.






