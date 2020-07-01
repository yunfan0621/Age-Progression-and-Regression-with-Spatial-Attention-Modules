import os
from options.test_options import TestOptions
from util import util
from data import CreateDataLoader
from models import create_model
from PIL import Image

from pdb import set_trace as ST

if __name__ == '__main__':

    opt = TestOptions().parse()
    opt.nThreads = 1    # test code only supports nThreads = 1
    opt.batchSize = 1   # test code only supports batchSize = 1
    opt.serial_batches = True  # no shuffle
    opt.no_flip = True  # no flip

    epoch_strlist = opt.epochs.split(',')
    opt.epochs = []
    for epoch_str in epoch_strlist:
        epoch_int = int(epoch_str)
        if epoch_int >= 0:
            opt.epochs.append(epoch_int)

    # create
    data_loader = CreateDataLoader(opt)
    dataset = data_loader.load_data()

    # go over specified epochs
    for epoch in opt.epochs:
        opt.which_epoch = epoch
        model = create_model(opt)
        model.set_eval()
    
        # test
        for i, data in enumerate(dataset):
            if i >= opt.how_many:
                break

            # obtain the forward results
            model.set_input(data)

            # convert to PIL images
            gen_ret = model.get_face_aging_results()

            gen_ret_im = {}
            for age_label in gen_ret.keys():
                gen_ret_im[age_label] = Image.fromarray(gen_ret[age_label])

            img_name = data['A_name'][0]
            save_dir = os.path.join(opt.results_dir, opt.save_suffix, str(epoch))

            # save the individual generation result
            gen_save_dir = os.path.join(save_dir, 'generation')
            util.mkdirs(gen_save_dir)
            ST()
            for age_label in gen_ret_im.keys():
                img_base_name = img_name.split('.')[0]
                gen_save_path = os.path.join(gen_save_dir, img_base_name + '_cluster%d.jpg' % (age_label + 1))
                gen_ret_im[age_label].save(gen_save_path)            

            print('%05d/%05d: process image %s, saved to %s' % (i, len(dataset), img_name, save_dir))
    