import os,sys
# the python script to show the results under model/
# usage: put this file under model/. Then python parse.py
files = os.listdir('.')
excepts=[]
for file in files:
    if file != 'parse.py':
        try:
            a=open(os.path.join(file,'model_w_best_test_log.txt')).readlines()[0].split(',')
            t2ir1, t2ir5, t2ir10, i2tr1, i2tr5, i2tr10 = [float(aa.split(':')[-1]) for aa in a]
            print("{:50s}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}".format(file, t2ir1, t2ir5, t2ir10, i2tr1, i2tr5, i2tr10))
        except:
            excepts.append(file)
            continue
print(excepts)

