import numpy as np
from data import dataloader
from config import args
import train 

#### This the first model #######
class LinearRegressionModel:
  def __init__(self, dataset, l1 = 0.0, l2 = 0.0, lr = 0.001, epochs=10000):
    self.dataset = dataset
    self.l1 = l1
    self.l2 = l2
    self.learning_rate = lr
    self.epochs = epochs
    self.theta = np.random.randn(dataset.train_inputs.shape[1]+1)

  def add_one(self, x):
    X_new=np.hstack([np.ones((x.shape[0], 1)), x])
    return X_new

  def predict(self, x):
    y_pred=x@self.theta
    return y_pred

  def compute_mse_loss(self, y_true, y_pred):
    loss=np.square(np.subtract(y_true,y_pred)).mean()
    return loss


 ####### Cross Validation classss ########
class kFoldsCV:
  '''
   Provides train/val indices to split data in train/val sets


  '''
  def __init__(self, n_folds: int=10):
    self.n_folds = n_folds

  def __call__(self,dataset):
    self.train_inputs = dataset.train_inputs
    self.train_targets = dataset.train_targets

    fold_size=len(self.train_inputs)//self.n_folds
    indices=np.arange(len(self.train_inputs))

    fold_indices=[]
    #avg_loss_final=[]

    for i in range(self.n_folds-1):
      fold_indices.append((i*fold_size,(i+1)*fold_size))

    fold_indices.append(((self.n_folds)*fold_size,len(self.train_inputs)))
    fold_losses=[]
    for fold in fold_indices:
      X_val=self.train_inputs[fold[0]:fold[1]]
      Y_val=self.train_targets[fold[0]:fold[1]]
      tr_indices=list(set(indices).difference(set(range(fold[0], fold[1]))))

      X_tr=self.train_inputs[tr_indices]
      Y_tr=self.train_targets[tr_indices]

      fold_dataset=dataloader.Dataset(path=None)

      fold_dataset.train_inputs=X_tr
      fold_dataset.test_inputs=X_val
      fold_dataset.train_targets=Y_tr
      fold_dataset.test_targets=Y_val

      model=LinearRegressionModel(fold_dataset , lr = args.lr, epochs=args.epoch)
      train.train(model,batch_size=128, plot= True)

      X_val=model.add_one(X_val)
      ypred=model.predict(X_val)

      fold_loss=model.compute_mse_loss(ypred, Y_val)

      fold_losses.append(fold_loss)
      avg_loss=sum(fold_losses)/len(fold_losses)
   
