# Problem understanding: Cat vs. Classifier #
In this toy project, we aim to build a deep learning clssifier that predicts whether an image is cat or not. </br>

<!-- <img src="images/cat.0.jpg" align="center">
<img src="images/dog.0.jpg" align="right"> -->


  Cat            |  Dog
:--------------:|:--------:
![](figures/cat.1.jpg)  |  ![](figures/dog.4.jpg)



<!-- <br> -->
Refer to this [link](https://www.kaggle.com/competitions/dogs-vs-cats/data) to get the data.

# Run the project #
In this project, we have two steps: training and predicting. In the predict step, you can upload any image from your laptop and predict it. Let's show you how to run the project.

If you do not have venv package, please refer to this [link](https://linuxize.com/post/how-to-create-python-virtual-environments-on-ubuntu-18-04/)
</br>

## Create virtual environment ##

```
$ python3 -m venv ENV_NAME
```
## Activate your environment ##

```
$ source ENV_NAME/bin/activate
```

## Requirement installations ##
To run this, make sure to install all the requirements by:

```
$ pip install -r requirements.txt 
```
# Training the model #

```
$ python3 main.py --epochs --numFold --withKfold  --valL1  --valL2  
```
## Example of running models ##
```
$ python3 main.py --epochs 100 --numFold 6 --withKfold 1 --valL1 0 --valL2 0 
```
```
$ python3 main.py --epochs 100 --numFold 6 --withKfold 0 --valL1 0.01 --valL2 0 
```
```
$ python3 main.py --epochs 100 --numFold 6 --withKfold 0 --valL1 0 --valL2 0.01 
```
```
$ python3 main.py --epochs 100 --numFold 6 --withKfold 0 --valL1 0.01 --valL2 0.01 
```
```
$ python3 main.py --epochs 100 --numFold 6 --withKfold 0 --valL1 0 --valL2 0
```


# Results Presentation

``` Resnet result```  </br>
![caption](figures/train_res.png) 

``` CNN result```  </br>
![caption](figures/train_res.png) 

# Make prediction #

```
$python3 predict.py --image_path "./data/Images/cat.0.jpg"
```

``` Example: Result```  </br>
![caption](figures/cat.0.jpg) </br>
{'class': 'cat', 'confidence': '0.5970718264579773'}
---
___

---
___

# Related Papers #

* <a href= 'https://arxiv.org/pdf/1512.03385.pdf'> Resnet </a>
* <a href= 'https://cs.nju.edu.cn/wujx/paper/CNN.pdf'> Convolutional neural networks</a>


# Contributors #
<div style="display:flex;align-items:center">

<div style="display:flex;align-items:center">
    <div>
        <h5> <a href='..'> Mr. A </a> </h5> <img src="figures/cat.1.jpg" height= 7% width= 7%>
<div>
    <h5> <a href='.'> Mr. B </a> </h5> <img src="figures/cat.1.jpg" height= 7% width= 7%>
    
<div>
    <h5> <a href='.'> Mm. K </a> </h5> <img src="figures/cat.1.jpg" height= 7% width= 7%>
    
</div>