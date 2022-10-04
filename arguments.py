import argparse


def arg_parse():
    parser = argparse.ArgumentParser(description='GraphACL Arguments.')
    parser.add_argument('--gnn', type=str, default='gin',
                        help='GNN gin, or gcn (default: gin)')
    parser.add_argument('--pooling_global', type=str, default='sum',
                        help='')
    parser.add_argument('--drop_ratio', type=float, default=0.0,
                        help='dropout ratio (default: 0.0)')
    parser.add_argument('--decay', type=float, default=0.99,
                        help='moving_average_decay (default: 0.99)')
    parser.add_argument('--num_layer', type=int, default=2,
                        help='number of GNN message passing layers (default: 2)')
    parser.add_argument('--emb_dim', type=int, default=512,
                        help='dimensionality of hidden units in GNNs (default: 512)')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='input batch size for training (default: 32)')
    parser.add_argument('--epochs', type=int, default=50,
                        help='number of epochs to train (default: 50)')
    parser.add_argument('--num_workers', type=int, default=0,
                        help='number of workers (default: 0)')
    parser.add_argument('--dataset', type=str, default="IMDB-MULTI",
                        help='dataset name (default: IMDB-MULTI)')

    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--runs', type=int, default=1)

    parser.add_argument('--test_freq', type=int, default=1)
    parser.add_argument('--projection_size', type=int, default=512)
    parser.add_argument('--prediction_size', type=int, default=512)
    parser.add_argument('--projection_hidden_size', type=int, default=64)
    parser.add_argument('--seed', type=int, default=123)
    parser.add_argument("--aug1", default='random4', type=str,
                        help='diffusion, dnodes, pedges, subgraph, mask_nodes, random2, random3, random4')
    parser.add_argument("--aug_ratio", default=0.1, type=float,
                        help='Perturbation Ratio')
    parser.add_argument('--beta', type=float, default=0.5)
    parser.add_argument('--alpha', type=float, default=1)

    parser.add_argument('--log_dir', default='log_dir', help='directory to save log')
    parser.add_argument('--log_file', type=str, default='results.txt', help='name of file for logging')
    parser.add_argument('--cluster', type=int, default=512, help="number of learnable comparison features")
    parser.add_argument('--bank_dim', default=512, type=int, help='feature dimension (default: 512)')
    parser.add_argument('--bank_t', default=0.12, type=float, help='softmax temperature (default: 0.12)')
    parser.add_argument('--lambda1', default=1.0, type=float)
    parser.add_argument('--lambda2', default=1.0, type=float)
    parser.add_argument('--adversarial', type=int, default=1)
    parser.add_argument('--model', type=str, default='graphacl',
                        help='graphacl, graphacl_star')

    args = parser.parse_args()
    return args
