from opt import *
from model import *
from util import *
from data_loader import SDataLoader
from youcook_data_loader import YoucookDataLoader
from evaluator import Evaluator
from trainer import Trainer
import torch
import torch.backends.cudnn as cudnn
torch.manual_seed(43211)
torch.cuda.manual_seed(43211)

def main():
    cudnn.benchmark = True
    params = parse_arguments()
    # dl = SDataLoader(params)
    dl = YoucookDataLoader(params)
    params.vocab_size = len(dl.vocab)+2
    evaluator = Evaluator(params, dl)

    if params.is_train:
        print("Training...")
        if params.glove:
            model = QAN(params, dl.w2v)
        else:
            model = QAN(params)
        model_epoch = 0
        best_prev = 0.
        if params.model_ckpt:
            model_file_path = os.path.join(params.model_path, params.model_ckpt)
            if os.path.exists(model_file_path):
                if params.is_transfer:
                    print('Transferring {}'.format(model_file_path))
                    checkpoint = torch.load(model_file_path)
                    model.load_state_dict(checkpoint['state_dict'])
                else:
                    print('loading {}'.format(model_file_path))
                    checkpoint = torch.load(model_file_path)
                    model.load_state_dict(checkpoint['state_dict'])
                    model_epoch = checkpoint['model_epoch']
                    best_prev = checkpoint['best_prev']
            else:
                print(("Error: no checkpoint found at '{}'".format(model_file_path)))
                exit(0)

        model = model.cuda()
        trainer = Trainer(params, model, dl, evaluator)
        trainer.train(model, model_epoch, best_prev)


    else: # testing
        if params.glove:
            model = QAN(params, dl.w2v)
        else:
            model = QAN(params)

        if params.load_best:
            model_file_path = os.path.join(params.model_dir, params.model_name, 'model_w_best.pt')
        else:
            model_file_path = os.path.join(params.model_dir, params.model_name, params.model_ckpt)

        if os.path.exists(model_file_path): 
            print("Loading model:{}".format(model_file_path))
            checkpoint = torch.load(model_file_path)
            model.load_state_dict(checkpoint['state_dict'])
            model_epoch = checkpoint['model_epoch']
            best_prev = checkpoint['best_prev']
        else:
            print(("Error: no checkpoint found at '{}'".format(model_file_path)))
            exit(0)
            
        model = model.cuda()

        if not params.is_inf:
            print("Evaluating t2i model on test set...")
            t2i_r1, t2i_r5, t2i_r10 = evaluator.t2i(model, is_test=True)
            print("t2i R@1 : {}".format(t2i_r1))
            print("t2i R@5 : {}".format(t2i_r5))
            print("t2i R@10 : {}".format(t2i_r10))

            print("Evaluating i2t model on test set...")
            i2t_r1, i2t_r5, i2t_r10 = evaluator.i2t(model, is_test=True)
            print("i2t R@1 : {}".format(i2t_r1))
            print("i2t R@5 : {}".format(i2t_r5))
            print("i2t R@10 : {}".format(i2t_r10))


            logger = LogCollector(os.path.join(model_file_path.replace('.pt', '_test_log.txt')))
            logger.set('t2i_r@1', t2i_r1)
            logger.set('t2i_r@5', t2i_r5)
            logger.set('t2i_r@10', t2i_r10)
            logger.set('i2t_r@1', i2t_r1)
            logger.set('i2t_r@5', i2t_r5)
            logger.set('i2t_r@10', i2t_r10)
            logger.log()


        else:
            print("Inference t2i model on test set...")
            evaluator.t2i_inf(model)


if __name__ == '__main__':
    main()
