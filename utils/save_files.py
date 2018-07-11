import os
import shutil
from datetime import datetime


def make_dir(dir_name):
    if not os.path.exists(dir_name):
        os.mkdir(dir_name)


def save_files(opt):
    
    expr_root_dir = 'experiments'


    opt.expr_dir = os.path.join(expr_root_dir, '%s_%s_%s', datetime.today().strftime('%Y%m%d_%H%M%S')) % (opt.dataset, opt.task, opt.model)
    opt.proj_dir = '/'.join(os.path.join(__file__).split('/')[:-1])
    opt.log_path_train = os.path.join(opt.expr_dir, 'train_log.txt')
    opt.log_path_valid = os.path.join(opt.expr_dir, 'valid_log.txt')
    opt.expr_code_dir = os.path.join(opt.expr_dir, 'code')

    make_dir(expr_root_dir)
    make_dir(os.path.join(expr_root_dir, '%s_%s_%s') % (opt.dataset, opt.task, opt.model))
    make_dir(opt.expr_dir)
    make_dir(opt.expr_code_dir)

    # save config
    make_dir(os.path.join(opt.expr_code_dir, 'configs'))
    shutil.copyfile('configs/init_config.py', os.path.join(os.path.join(opt.expr_code_dir, 'configs/%s.py') % 'init_config.py'))
    if opt.model == 'ssd300' or opt.model == 'ssd512':
        shutil.copyfile('configs/%s.py' % 'ssd', os.path.join(os.path.join(opt.expr_code_dir, 'configs/%s.py') % 'ssd'))
    else:
        shutil.copyfile('configs/%s.py' % opt.model, os.path.join(os.path.join(opt.expr_code_dir, 'configs/%s.py') % opt.model))


    # save trainer
    make_dir(os.path.join(opt.expr_code_dir, 'trainer'))
    shutil.copyfile('trainer/init_trainer.py', os.path.join(os.path.join(opt.expr_code_dir, 'trainer/%s.py') % 'init_trainer.py'))
    if opt.model == 'ssd300' or opt.model == 'ssd512':
        shutil.copyfile('trainer/train_%s.py' % 'ssd', os.path.join(os.path.join(opt.expr_code_dir, 'trainer/train_%s.py') % 'ssd'))
    else:
        shutil.copyfile('trainer/train_%s.py' % opt.model, os.path.join(os.path.join(opt.expr_code_dir, 'trainer/train_%s.py') % opt.model))


    # save dataset
    make_dir(os.path.join(opt.expr_code_dir, 'datasets'))
    shutil.copyfile('datasets/init_loader.py', os.path.join(os.path.join(opt.expr_code_dir, 'datasets/%s.py') % 'init_loader.py'))
    if opt.dataset not in ['voc07', 'voc12', 'voc0712']:
        shutil.copyfile('datasets/%s.py' % opt.dataset, os.path.join(os.path.join(opt.expr_code_dir, 'datasets/%s.py') % opt.dataset))
    else:
        shutil.copyfile('datasets/voc.py', os.path.join(os.path.join(opt.expr_code_dir, 'datasets/voc.py')))

    # save transforms
    shutil.copytree('transforms', os.path.join(os.path.join(opt.expr_code_dir, 'transforms')))

    # save loss
    shutil.copytree('loss', os.path.join(os.path.join(opt.expr_code_dir, 'loss')))

    # save metric
    shutil.copytree('metrics', os.path.join(os.path.join(opt.expr_code_dir, 'metrics')))

    # save utils
    shutil.copytree('utils', os.path.join(os.path.join(opt.expr_code_dir, 'utils')))

    # save preprocess
    if opt.dataset not in ['voc07', 'voc12', 'voc0712']:
        shutil.copytree('preprocess/%s' % opt.dataset, os.path.join(os.path.join(opt.expr_code_dir, 'preprocess/%s')% opt.dataset))
    else:
        pass # TODO implement voc preprocessing


    # save model source code
    if opt.model == 'ssd300' or opt.model == 'ssd512':
        shutil.copytree('modellibs/ssd', os.path.join(opt.expr_code_dir, 'modellibs/ssd'))
    else:
        shutil.copytree(os.path.join('modellibs', opt.model), os.path.join(opt.expr_code_dir, 'modellibs/%s') % opt.model)
    shutil.copyfile('modellibs/init_model.py', os.path.join(opt.expr_code_dir, 'modellibs/init_model.py'))


    # save main.py
    shutil.copyfile('main.py', os.path.join(opt.expr_code_dir, 'main.py'))