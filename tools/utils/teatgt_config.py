from argparse import ArgumentParser

class Config(object):

    def __init__(self):

        self.parser = self.setup_parser()
        self.args = vars(self.parser.parse_args())    
        self.__dict__.update(self.args)
        
        if self.mode != 'predcls':  #[sgcls, sgdet]
            self.tracking = True
            self.encoder_layers = 6
            self.encoder_attention_heads = 16

    def setup_parser(self):
        
        # TEAT-GT
        parser = ArgumentParser(description='training code')
        parser.add_argument('--mode', dest='mode', help='predcls/sgcls/sgdet', default='predcls', type=str)
        parser.add_argument('--save_path', default='checkpoint/', type=str)
        parser.add_argument('--model_path', default=None, type=str)
        parser.add_argument('--data_path', default='/data/AG/', type=str)
        parser.add_argument('--output_path', default='output/', type=str)
        parser.add_argument('--datasize', dest='datasize', help='mini dataset or whole', default='large', type=str)
        parser.add_argument('--lr', dest='lr', help='learning rate', default=1e-5, type=float)
        parser.add_argument('--warmup', default=3, type=int)
        parser.add_argument('--nepoch', help='epoch number', default=10, type=int)
        parser.add_argument('--use_ctl_loss', action='store_true')
        parser.add_argument('--use_cons_str_loss', action='store_true')
        parser.add_argument('--use_cons_sem_loss', action='store_true')
        parser.add_argument('--log_iter', default=100, type=int)   
        parser.add_argument('--tracking', action='store_true')
        
        # TokenGT
        parser.add_argument('--num_atoms', type=int, default=1168)
        parser.add_argument('--num_edges', type=int, default=1)
        parser.add_argument('--num_output', type=int, default=26)
        
        parser.add_argument('--lap_node_id', action='store_true')
        parser.add_argument('--lap_node_id_k', type=int, default=50)
        parser.add_argument('--lap_node_id_sign_flip', action='store_true')        
        parser.add_argument('--lap_node_id_eig_dropout', type=float, default=0.2)
        
        parser.add_argument('--rand_node_id', action='store_true')
        parser.add_argument('--rand_node_id_dim', type=int, default=50)
        
        parser.add_argument('--orf_node_id', action='store_true')
        parser.add_argument('--orf_node_id_dim', type=int, default=50)        
        
        parser.add_argument('--type_id', action='store_true', default=True)
        parser.add_argument('--stochastic_depth', action='store_true')
        
        parser.add_argument('--encoder_embed_dim', type=int, default=768)
        parser.add_argument('--encoder_layers', type=int, default=12)
        parser.add_argument('--encoder_attention_heads', type=int, default=32)
        parser.add_argument('--encoder_ffn_embed_dim', type=int, default=768)
        parser.add_argument('--return_attention', action='store_true', default=True)
        
        return parser