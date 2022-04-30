from data import dataloader
from config import args
import models
import train



dataset = dataloader.Dataset(args.metadata)

# model = models.LinearRegressionModel(dataset,l2=0.0, l1=0.0, lr = args.lr, epochs=args.epoch)
# train.train(model,args.bs)



import argparse## Al
parser = argparse.ArgumentParser()
parser.add_argument('-eps','--epochs',help='This is the number of epoch',required=True)
parser.add_argument('-k','--numFold',help='This is the number of fold',type=int,required=True)
parser.add_argument('-kf','--withKfold',help='This is to allow you to use cross-validation',type=int,required=True)
parser.add_argument('-l1','--valL1',help='This is to allow you to use cross-validation',type=float,required=True)
parser.add_argument('-l2','--valL2',help='This is to allow you to use cross-validation',type=float,required=True)


mains_args = vars(parser.parse_args())

if mains_args['withKfold']== 1:
    kv = models.kFoldsCV(mains_args['numFold'])
    kv.__call__(dataset)
elif mains_args['withKfold']== 0:
    if int(mains_args['valL1']) == 0 and int(mains_args['valL2']) != 0:
        model = models.LinearRegressionModel(dataset,l2=mains_args['valL2'], l1=0.0, lr = args.lr, epochs=args.epoch)
    elif int(mains_args['valL1']) != 0 and int(mains_args['valL2']) == 0:
        model = models.LinearRegressionModel(dataset,l2=0.0, l1=mains_args['valL1'], lr = args.lr, epochs=args.epoch)
    else:
        model = models.LinearRegressionModel(dataset,l2=mains_args['valL2'], l1=mains_args['valL1'], lr = args.lr, epochs=args.epoch)
    train.train(model,args.bs)
else:
    print('Wrong arguments,please check and run again!!!')

