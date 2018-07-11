import os
import argparse
import torch
from datasets.init_loader import init_dataloader_train, init_dataloader_valid, init_dataloader_test
from modellibs.init_model import init_model
from trainer.init_trainer import init_trainer
from evaluations.init_eval import init_eval
from configs.init_config import init_config
from utils.save_files import save_files


if __name__ == "__main__":

    # torch.multiprocessing.set_start_method("spawn", force=True)

    parser = argparse.ArgumentParser()

    parser.add_argument('--command', type=str, default='train')
    parser.add_argument('--dataset', type=str, default='v_caption')
    parser.add_argument('--task', type=str, default='classification')
    parser.add_argument('--model', type=str, default='chanet')
    parser.add_argument('--resume', type=bool, default=False)
    parser.add_argument('--resume_path', type=str, default='/home/son/PycharmProjects/Object_Detection/detectron/experiments/v_caption_classification_resnet/v_caption_classification_resnet_best_loss_0.180272/model_best.pth')

    parser.add_argument('--data_root_dir', type=str, default='/media/son/Repository2')

    parser.add_argument('--batch_size_train', type=int, default=128)
    parser.add_argument('--batch_size_valid', type=int, default=256)
    parser.add_argument('--batch_size_test', type=int, default=32)

    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--num_gpus', type=str, default=[0, 1])

    parser.add_argument('--print_freq', type=int, default=10)
    parser.add_argument('--print_freq_eval', type=int, default=100)

    parser.add_argument('--start_epochs', type=int, default=1)
    parser.add_argument('--max_epochs', type=int, default=90)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    if parser.parse_args().command == 'train':

        opt = init_config(parser)
        opt.device = device

        train_dataloader = init_dataloader_train(opt)
        valid_dataloader = init_dataloader_valid(opt)

        model = init_model(opt)
        trainer = init_trainer(opt, train_dataloader, valid_dataloader, model)

        save_files(opt)

        # training
        trainer.train_model(max_epoch=opt.max_epochs,
                            learning_rate=opt.lr)

    elif parser.parse_args().command == 'valid':

        opt = init_config(parser)
        opt.device = device
        opt.proj_dir = '/'.join(os.path.join(__file__).split('/')[:-1])

        opt.batch_size_test = 1
        test_dataloader = init_dataloader_test(opt)

        model = init_model(opt)
        evaluator = init_eval(opt, test_dataloader, model)

        evaluator.validate()

    elif parser.parse_args().command == 'test':

        opt = init_config(parser)