# -*- coding: UTF-8 -*-
import os
import time
import argparse
import multiprocessing

import numpy as np
import paddle.fluid as fluid
from paddle.fluid.contrib.slim import Compressor

import paddle
import reader
import utils
import creator
from model_check import check_cuda



# the function to train model
def do_compress(args):
    train_program = fluid.default_main_program()
    startup_program = fluid.default_startup_program()

    dataset = reader.Dataset(args)
    with fluid.program_guard(train_program, startup_program):
        train_program.random_seed = args.random_seed
        startup_program.random_seed = args.random_seed

        with fluid.unique_name.guard():
            train_ret = creator.create_model(
                args, dataset.vocab_size, dataset.num_labels, mode='train')

    test_program = train_program.clone()

    optimizer = fluid.optimizer.Adam(learning_rate=args.base_learning_rate)

    # init executor
    if args.use_cuda:
        place = fluid.CUDAPlace(int(os.getenv('FLAGS_selected_gpus', '0')))
        dev_count = fluid.core.get_cuda_device_count()
    else:
        dev_count = min(multiprocessing.cpu_count(), args.cpu_num)
        if (dev_count < args.cpu_num):
            print("WARNING: The total CPU NUM in this machine is %d, which is less than cpu_num parameter you set. "
                  "Change the cpu_num from %d to %d" % (dev_count, args.cpu_num, dev_count))
        os.environ['CPU_NUM'] = str(dev_count)
        place = fluid.CPUPlace()

    train_reader = paddle.batch(
                    dataset.file_reader(args.train_data),
                    batch_size=args.batch_size
                )
    test_reader = paddle.batch(
        dataset.file_reader(args.test_data),
        batch_size=args.batch_size
    )

    exe = fluid.Executor(place)
    exe.run(startup_program)

    if args.init_checkpoint:
        utils.init_checkpoint(exe, args.init_checkpoint+'.pdckpt' , train_program)

    train_feed_list = [('words',train_ret['words'].name),
                       ("targets", train_ret["targets"].name)]
    train_fetch_list = [('loss', train_ret['avg_cost'].name)]

    test_feed_list = [('words',train_ret['words'].name),
                       ("targets", train_ret["targets"].name)]
    test_fetch_list = [('f1_score', train_ret['f1_score'].name)]
    print(train_ret['crf_decode'].name)

    com_pass = Compressor(
        place,
        fluid.global_scope(),
        train_program=train_program,
        train_reader=train_reader,
        train_feed_list=train_feed_list,
        train_fetch_list=train_fetch_list,
        eval_program=test_program,
        eval_reader=test_reader,
        eval_feed_list=test_feed_list,
        eval_fetch_list=test_fetch_list,
        teacher_programs=[],
        train_optimizer=optimizer,
        distiller_optimizer=None)
    com_pass.config(args.compress_config)
    com_pass.run()



if __name__ == "__main__":
    # 参数控制可以根据需求使用argparse，yaml或者json
    # 对NLP任务推荐使用PALM下定义的configure，可以统一argparse，yaml或者json格式的配置文件。

    parser = argparse.ArgumentParser(__doc__)
    utils.load_yaml(parser, 'conf/args.yaml')

    user_parser = utils.ArgumentGroup(parser, "model", "model configuration")
    user_parser.add_arg("compress_config", str, 'conf/quantization.yaml', "The compress configure file")

    args = parser.parse_args()
    check_cuda(args.use_cuda)

    print(args)

    do_compress(args)

