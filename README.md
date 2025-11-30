# DST_Assessment_2
Group project for Data Science Toolbox

## Project Group 

- Kateryna Chorna  
- Harry Clarke  
- Minnie Jacobs
- James Railton
 
This project is equally owned by the four project partners.

## Reading Order
All report content is in the directory:
- Report/
  
The Report folder takes the following structure:

- Models
- Report.ipy
- References.bib
- 
Due to the fact we are using models that required long training loops, we have included a folder of all of our trained models, the code and the results, and when discussing this have loaded them into the report notebook via github.
The Report is rather dense since it includes a full exploration of our datasets, models and conclusions.
It takes the following structure:
- Intro
- About our datasets
   - CIFAR10
   - Food101
- EDA
- Methods
  - Optimisation on CIFAR10
  - ResNet18
  - ViT
  - Robustness
 - Scalability
 - Results
 - References

## Description
Throughout our project we focus on two data sets; the CIFAR 10 and Food 101 datasets. The smaller CIFAR-10 dataset is used as a toy data set to explore optimisation, specifically focusing on the number of convolution layers. 
We then focus on the ResNet18 architecture since it is sufficiently deep while also being manageable with limited GPU access. 
We use ViT transformers and transfer learning to explore other models and see how these perform (using accuracy, loss and time as metrics), before focusing on testing their robustness. 
We then compare and contrast our results in the results section of our report. 
