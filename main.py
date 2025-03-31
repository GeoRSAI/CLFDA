
# coding=utf-8
if __name__ == '__main__': #避免num_works>0,引起的来回重新调用程序

    import os
    import random
    import numpy as np
    import time

    import torch
    import torch.nn.parallel
    import torch.backends.cudnn as cudnn
    import torch.optim

    import torch.utils.data
    import torch.utils.data.distributed

    from data import Data
    from logger import Logger
    from utils import Timer
    from utils import AverageMeter, ProgressMeter, get_evaluat, get_multi_evaluat
    from utils import cross_entropy_loss, loss_calc, FCCDN_loss_without_seg, logger_updata
    from model import model_dict
    from option import args

    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    pjoin = os.path.join
    global logger
    logger = Logger(args)


    def main():
        logprint = logger.log_printer.logprint
        accprint = logger.log_printer.accprint
        netprint = logger.netprint
        timer = Timer(args.epochs)

        if args.seed is not None:
            random.seed(args.seed)
            torch.manual_seed(args.seed)
            np.random.seed(args.seed)
            os.environ['PYTHONHASHSEED'] = str(args.seed)
            logprint("=>use seed= {}".format(args.seed))
        if args.gpu is not None or torch.cuda.is_available():
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            if device != 'cpu':
                args.gpu = device
                torch.cuda.manual_seed(args.seed)
                torch.cuda.manual_seed_all(args.seed)
                cudnn.deterministic = True
                cudnn.benchmark = False
                # torch.use_deterministic_algorithms(True)
                logprint("=>GPU：{},GPU seed= {}".format(args.gpu, args.seed))
            else:
                logprint("=>no GPU")
                args.gpu = None

        global best_index, best_epoch

        loader = Data(args)
        train_loader_task1 = loader.train_loader_task1
        val_loader_task1 = loader.test_loader_task1


        # num_classes = num_classes_dict[args.dataset]
        WEIGHTS = torch.ones(args.num_classes)
        if args.dataset == 'gid15' or args.dataset == "FUSU":
            WEIGHTS[0] = 0

        logprint("Use Model: '{}' for training".format(args.arch))
        model = model_dict[args.arch](num_classes=args.num_classes, num_channels=3, use_bn=False)
        model.to(args.gpu)

        optimizer = torch.optim.AdamW(model.parameters(), lr=args.LR, weight_decay=args.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=16)
        criterion = loss_calc

        # @mst: save the model after initialization if necessary
        if args.save_init_model:
            state = {
                'arch': args.arch,
                'model': model,
                'state_dict': model.state_dict(),
                'ExpID': logger.ExpID,
            }
            save_model(state, mark='init')


        netprint(model, comment='base model arch')

        best_index = 0.
        best_epoch = 0
        # 是否从检查点恢复
        if args.resume:
            if os.path.isfile(args.resume):
                logprint("=> loading checkpoint ''".format(args.resume))
                if args.gpu is not None:
                    checkpoint = torch.load(args.resume)
                else:
                    #
                    loc = 'cuda:{}'.format(args.gpu)
                    checkpoint = torch.load(args.resume, map_location=loc)
                args.start_epoch = checkpoint['epoch']
                best_index = checkpoint['mIou']
                # if args.gpu is not None:
                #     best_index = best_index.to(args.gpu)
                try:
                    best_epoch = checkpoint['best_epoch']
                except:
                    pass
                model.load_state_dict(checkpoint['state_dict'])
                optimizer.load_state_dict(checkpoint['optimizer'])
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=checkpoint['epoch'])
                Resume_ExpID = checkpoint['ExpID']
                logger_updata(logger, Resume_ExpID)
                logprint = logger.log_printer.logprint



                logprint("=> loaded checkpoint '{}' (epoch {})"
                         .format(args.resume, checkpoint['epoch']))
            else:
                logprint("=> no checkpoint found at '{}'".format(args.resume))


        if hasattr(model, "module"):
            model = model.module
        else:
            model = model



        for epoch in range(args.start_epoch+1,args.epochs+1):

            train_loss, train_mF1S, train_mIou\
                = train(train_loader_task1, model, criterion, optimizer, scheduler, epoch, args, print_log=True,weights=WEIGHTS)

            test_loss, test_mF1S, test_mIou, test_F1S, test_Iou, test_conf\
                = validate(val_loader_task1, model, criterion, args, logger, epoch,weights=WEIGHTS)

            is_best = test_mIou > best_index
            if is_best:
                best_epoch = epoch
            best_index = max(test_mIou, best_index)

            accprint(
                "train_loss %.4f train_mF1S %.4f train_mIou %.4f  | Epoch %d (best_index %.4f @ Best_Epoch %d)" %
                (train_loss, train_mF1S, train_mIou, epoch, best_index, best_epoch))

            accprint(
                "test_loss %.4f test_F1S %.4f test_mIou %.4f | Epoch %d (best_index %.4f @ Best_Epoch %d) F1 %s Iou %s conf %s" %
                (test_loss, test_mF1S, test_mIou, epoch, best_index, best_epoch, test_F1S, test_Iou, test_conf))
            logprint('predicted finish time: %s' % timer())

            if args.arch:
                # @mst: use our own save func
                state = {'epoch': epoch,
                         'arch': args.arch,
                         'model': model,
                         'state_dict': model.state_dict(),
                         'mF1S': test_mF1S,
                         'mIou': test_mIou,
                         'optimizer': optimizer.state_dict(),
                         'ExpID': logger.ExpID,
                         'best_epoch': best_epoch,
                         }

                save_model(state, is_best, mark='Last')


    def train(train_loader_task1,
              model,
              criterion,
              optimizer,
              scheduler,
              epoch,
              args,
              print_log,
              weights,
              ):
        batch_time = AverageMeter('Time', ':6.3f')
        data_time = AverageMeter('Data', ':6.3f')
        loss_ = AverageMeter('Loss', ':.4e')
        ## 精度
        mF1S = AverageMeter('mF1S', ':6.2f')
        mIou = AverageMeter('mIou', ':6.2f')

        progress = ProgressMeter(
            len(train_loader_task1),
            [batch_time, data_time, loss_,  mF1S, mIou],
            prefix="Epoch: [{}]".format(epoch))
        model.train()
        weights = weights.cuda()
        if args.dataset == 'gid15' or args.dataset == "FUSU":
            conf_matrix = get_multi_evaluat(num_classes=args.num_classes, ignore=0)
        else:
            conf_matrix = get_multi_evaluat(num_classes=args.num_classes)
        optimizer.zero_grad()
        end = time.time()
        batch_time.update(time.time() - end)
        for i, (images, target) in enumerate(train_loader_task1):
            if args.gpu is not  None:
                images = images.cuda(args.gpu,non_blocking=True)
                target = target.cuda(args.gpu,non_blocking=True)
            if torch.sum(target) == 0:
                continue

            output = model(images)
            loss= criterion(output, target,weights) / args.itersize
            loss.backward()

            output = torch.argmax(output,dim=1)
            conf_matrix.calculator(output, target.to(torch.int64))
            mIoU, IoU = conf_matrix.cal_miou()
            mean_F1, F1 = conf_matrix.cal_F1score()

            if (i+1)% args.itersize == 0:
                optimizer.step()
                optimizer.zero_grad()

                loss_.update(loss.item()*args.itersize, images.size(0))
                # 精度
                mF1S.update(mean_F1, 1)
                mIou.update(mIoU, 1)
                batch_time.update(time.time() - end)
                end = time.time()
            if print_log and i % args.print_freq == 0:
                progress.display(i)

        if scheduler is not None:
            scheduler.step()
        return loss_.val, mF1S.val, mIou.val
        ###

    def validate(val_loader, model, criterion, args,logger, epoch, weights):
        batch_time = AverageMeter('Time', ':6.3f')
        loss_ = AverageMeter('Loss', ':.4e')
        mF1S = AverageMeter('mF1S', ':6.2f')
        mIou = AverageMeter('mIou', ':6.2f')

        progress = ProgressMeter(
            len(val_loader),
            [batch_time, loss_, mF1S, mIou],
            prefix='Test: ')


        model.eval()
        weights = weights.cuda()
        if args.dataset == 'gid15' or args.dataset == "FUSU":
            conf_matrix = get_multi_evaluat(num_classes=args.num_classes, ignore=0)
        else:
            conf_matrix = get_multi_evaluat(num_classes=args.num_classes)
        val_path_list = val_loader.dataset.img_path
        filename = [os.path.splitext(os.path.basename(path))[0] for path in val_path_list]
        with torch.no_grad():
            end = time.time()
            for i, (images, target) in enumerate(val_loader):
                if args.gpu is not  None:
                    images = images.cuda(args.gpu,non_blocking=True)
                    target = target.cuda(args.gpu,non_blocking=True)
                output = model(images.to(args.gpu))
                if torch.sum(target) == 0: #标签全部为0的时候，loss会报错，跳过去，但是不反向传播，对结果没有任何影响。
                    loss = torch.tensor(loss_.avg)
                else:
                    loss = criterion(output, target, weights)
                    # loss_change, diceloss, foclaloss = criterion(output, target) / args.itersize
                    # loss = loss_change.mean()

                output = torch.argmax(output, dim=1)
                conf_matrix.calculator(output, target.to(torch.int64))
                mIoU, IoU = conf_matrix.cal_miou()
                mean_F1, F1 = conf_matrix.cal_F1score()

                loss_.update(loss.item(), images.size(0))
                mF1S.update(mean_F1, 1)
                mIou.update(mIoU, 1)

                # measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()

                if i != 0 and i % args.print_freq == 0:
                    progress.display(i)

        return loss_.avg, mF1S.val, mIou.val, ' '.join(map(str, F1.tolist())), ' '.join(map(str, IoU.tolist())), ' '.join(map(str, conf_matrix.return_conf().flatten().tolist()))


    def save_model(state, is_best=False, mark=''):
        # out = pjoin(logger.weights_path, "checkpoint.pth")
        # torch.save(state, out)
        if is_best:
            out_best = pjoin(logger.weights_path, "checkpoint_best.pth")
            torch.save(state, out_best)
        if mark:
            out_mark = pjoin(logger.weights_path, "checkpoint_{}.pth".format(mark))
            torch.save(state, out_mark)

    def apply_weights( weights, model):
        """flush weights to model"""
        model.load_state_dict(weights)


    main()

