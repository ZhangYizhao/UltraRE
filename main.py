import argparse

from config import InsParam, Instance

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='ml1m', help='dataset name')
parser.add_argument('--epoch', type=int, default=50, help='number of epochs')
parser.add_argument('--worker', type=int, default=24, help='number of CPU workers')
parser.add_argument('--verbose', type=int, default=1, help='verbose type')
parser.add_argument('--group', type=int, default=2, help='number of groups')
parser.add_argument('--layer', nargs='+', default=[64, 32], help='setting of layers')
parser.add_argument('--learn', type=str, default='sisa', help='type of learning and unlearning')
parser.add_argument('--delper', type=int, default=2, help='deleted user proportion')
parser.add_argument('--deltype', type=str, default='rand', help='deletion type')
parser.add_argument('--dis', type=str, default='nor', help='type of distinguishability loss')

# this is an example of main
def main():
    # read parser
    args = parser.parse_args()

    assert args.dataset in ['toy', 'db', 'ah', 'ad', 'am', 'ml1', 'ml20', 'ml25', 'netf', 'adn', 'ml1m']
    dataset = args.dataset

    assert args.epoch > 0
    epochs = args.epoch

    assert args.worker > 0
    n_worker = args.worker
    
    '''
    print verbose
    if verbose == 0, print nothing
    if verbose == 1, print every epoch
    if verbose == 2, print every batch
    '''
    assert args.verbose in [0, 1, 2]
    verbose = args.verbose

    assert args.group >= 0
    n_group = args.group

    for i in args.layer:
        assert  type(i) == int
    layers = args.layer

    assert args.learn in ['seq', 'add', 'sisa']
    learn_type = args.learn

    assert args.delper in [2, 5]
    del_per = args.delper

    assert args.deltype in ['rand', 'top', 'test']
    del_type = args.deltype

    assert args.dis in ['nor', 'u2u', 'd2d']
    dis_type = args.dis

    # initiate instance
    param = InsParam(dataset, epochs, n_worker, layers, n_group, del_per, del_type, dis_type)
    ins = Instance(param)

    # begin instance
    if n_group == 0:
        ins.runFull(is_save=True, 
                    verbose=verbose)
    else:
        group_types = ['emb-ot']
        for group_type in group_types:
            ins.runGroup(is_save=True,
                        learn_type=learn_type, 
                        group_type=group_type,
                        n_group=n_group,  
                        verbose=verbose)

if __name__ == '__main__':
    main()