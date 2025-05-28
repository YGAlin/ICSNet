import argparse

def opt():
    parser = argparse.ArgumentParser(description='',formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--data_path',type=str,default=r'E:\LY\Bilinear\suni\500-suni0.4/')
    parser.add_argument('--payload', type=int, default=0.4, help='The payload.')
    parser.add_argument('--data_path_source', type=str, default='a',
                        help='Root of train data set of the source domain')
    parser.add_argument('--data_path_target', type=str, default='b',
                        help='Root of train data set of the target domain')
    parser.add_argument('--data_path_test', type=str, default='b',
                        help='Root of the test data set of the target domain')
    parser.add_argument('--src', type=str, default='train1000',
                        help='choose between train | test')
    parser.add_argument('--tar', type=str, default='test',
                        help='choose between train | test')
    parser.add_argument('--test', type=str, default='test',
                        help='choose between train | test')
    parser.add_argument('--output_dir', type=str, default='',help='choose between train | test')
    parser.add_argument('--output_pkl', type=str, default='',help='choose between train | test')
    parser.add_argument('--unet_pkl', type=str, default='', help='choose between train | test')
    parser.add_argument('--pkl', type=str, default='', help='choose between train | test')

    # Optimization options
    parser.add_argument('--epochs', '-e', type=int, default=100, help='Number of epochs to train')
    parser.add_argument('--batch_size_s', '-b-s', type=int, default=8, help='Batch size of the source data.')
    parser.add_argument('--batch_size_t', '-b-t', type=int, default=8, help='Batch size of the target data.')
    parser.add_argument('--lr', '--learning_rate', type=float, default=0.001, help='The Learning Rate.')
    parser.add_argument('--momentum', '-m', type=float, default=0.9, help='Momentum.')
    parser.add_argument('--weight_decay', '-wd', type=float, default=1e-3, help='Weight decay (L2 penalty).')
    parser.add_argument('--seed', type=int, nargs='+', default=42,
                        help='Decrease learning rate at these epochs[used in step decay].')
    parser.add_argument('--weight', type=float, default=0.1, help='LR is multiplied by gamma on schedule.')
    parser.add_argument('--net', type=str, default='SRNet', help='Backbone network used by the model')
    parser.add_argument('--log', type=str, default='./checkpoints', help='Log folder')
    args = parser.parse_args()

    args.output_dir ='acctxt' + '\\'+args.net+'\\' + args.data_path.split('-')[1] + '\\'+args.data_path_source + '_' + args.data_path_target
    args.output_pkl = 'checkpoints' + '\\'+args.net+'\\' + args.data_path.split('-')[
        1] + '\\' + args.data_path_source + '_' + args.data_path_target
    args.pkl =args.data_path + '\\'+args.data_path_source.split('_')[0]+'-' + args.data_path.split('-')[1]+args.net
    args.unet_pkl = 'unet'+'\\'+args.data_path.split('-')[1] + '\\' + args.data_path_source + '_' + args.data_path_target
    args.D_pkl = 'd' + '\\' + args.data_path.split('-')[
        1] + '\\' + args.data_path_source + '_' + args.data_path_target
    return args