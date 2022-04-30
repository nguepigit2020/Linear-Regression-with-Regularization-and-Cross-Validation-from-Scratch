import numpy as np
from utils import plot_loss_and_params


def train(model, batch_size, plot=True):
    train_losses = []
    test_losses = []
    for epoch in range(model.epochs):
      indices=np.arange(len(model.dataset.train_inputs))
      np.random.shuffle(indices)

      model.dataset.train_inputs=model.dataset.train_inputs[indices]
      model.dataset.train_targets=model.dataset.train_targets[indices]
      number_of_batches=len(model.dataset.train_inputs)//batch_size

      epoch_losses=[]
      for i in range(number_of_batches):
        x=model.dataset.train_inputs[i*batch_size: (i+1)*batch_size]
        y=model.dataset.train_targets[i*batch_size: (i+1)*batch_size]
        x=model.add_one(x)
        
        gradient=(x.T@(model.predict(x)-y))
        #L1 regularization
        if model.l1>0:
          gradient+=np.sign(model.theta)*model.l1
        #l2 Reg
        elif model.l2>0:
          gradient+=2*model.theta*model.l2
        else:
        #Elastic
          gradient+=np.sign(model.theta)*model.l1+2*model.l2*(model.theta)

        model.theta -=gradient*model.learning_rate
        y_pred=model.predict(x)
        loss=model.compute_mse_loss(y_pred, y)

        epoch_losses.append(loss)

      train_losses.append(np.mean(epoch_losses))
      test_inputs=model.dataset.test_inputs
      test_inputs=model.add_one(test_inputs)
      y_pred_test=model.predict(test_inputs)
      test_loss=model.compute_mse_loss(model.dataset.test_targets, y_pred_test)
      test_losses.append(test_loss)
      if epoch%1000==0:
        print("Epoch: {} Train - Loss: {:.4f} Test - Loss: {:.4f} ".format(epoch, train_losses[-1], test_losses[-1]))
    if plot:
       plot_loss_and_params(model,train_losses, test_losses)