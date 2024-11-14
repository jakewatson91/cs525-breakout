def add_arguments(parser):
    '''
    Add your arguments here if needed. The TA will run test.py to load
    your default arguments.

    For example:
        parser.add_argument('--batch_size', type=int, default=32, help='batch size for training')
        parser.add_argument('--learning_rate', type=float, default=0.01, help='learning rate for training')
    '''
    parser.add_argument('--batch_size', type=int, default=32, help='batch size for training')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='learning rate for training')
    parser.add_argument('--gamma', type=float, default=0.99, help='batch size for training')
    parser.add_argument('--epsilon', type=float, default=0.9, help='learning rate for training')
    parser.add_argument('--epsilon_min', type=float, default=0.025, help='batch size for training')
    parser.add_argument('--decay_rate', type=float, default=0.000004, help='learning rate for training')
    parser.add_argument('--buffer_len', type=int, default=10000, help='learning rate for training')
    parser.add_argument('--num_episodes', type=int, default=10000, help='learning rate for training')
    parser.add_argument('--alpha', type=float, default=0.6, help='learning rate for training')
    parser.add_argument('--beta', type=float, default=0.4, help='learning rate for training')
    parser.add_argument('--beta_increment', type=float, default=0.000004, help='learning rate for training')

    return parser
