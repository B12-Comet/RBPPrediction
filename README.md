# Prediction of RNA-Protein binding sites based on CNN-BLSTM-Attention
## Environment
python: 3.6

tensorflow==1.11.0

Keras==2.2.4

imbalanced-learn==0.8.1

matplotlib==3.3.4

numpy==1.16.4

pandas==1.1.5

scikit-learn==0.24.2

shap==0.41.0

## Instruction
The function of the directory and the files in the directory are described as follows:

```code```:

This directory holds the core code for this study.

```datasets```:
This directory holds the datasets used in this study.

```model```:

In this directory are the models obtained from the training of each of the 27 datasets.

## About Predictor
To facilitate online prediction, we have developed an online predictor based on Python.
### Download and Setup
The predictor can be downloaded at https://pan.baidu.com/s/1NNtVw7Xzqy9dfhjAC9KKFA?pwd=z9u3 .

Download and get a zip package, unzip it and get two folders 'build' and 'dist', find the exe file under the 'build' folder and click it to run the predictor.
### How to Use Predictor
The initial interface of the predictor is shown in the figure, which is divided into two parts: selecting a prediction file and displaying the prediction results. First, click the "Open File" button at the top of the interface, and select a local test file, the file format is "fasta" or "fa".
![image](https://github.com/B12-Comet/RBPPrediction/assets/81473454/8943c83f-401c-49c2-9bee-afc9b8d2bdfe)
After selecting the test file, wait for the prediction result. The prediction results are displayed in the interface in the form of a list, as shown in Figure, the list contains four columns, the first column is the sequence number, the second and fourth columns are the real label and the prediction label, which are represented by "negative" and "positive" respectively. Are denoted by "negative" and "positive". The third column is the probability predicted by the model.
![image](https://github.com/B12-Comet/RBPPrediction/assets/81473454/19149540-a780-4e78-9c49-ebe6d112b8e7)
As shown in the figure, there is a button labeled "Save Results" at the bottom right of the interface. By clicking this button, the user can easily store the model's prediction results in CSV file format to the local computer for subsequent data processing and analysis.
![image](https://github.com/B12-Comet/RBPPrediction/assets/81473454/d372f3a6-94c8-4984-ab7f-2c4feae4478a)



