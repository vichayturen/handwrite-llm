class Config:
    def __init__(
            self,
            vocab_size: int,
            num_hiddens: int,
            num_layers: int,
            num_heads: int,
            num_mlp_intermediate: int,
            dropout: float
        ):
        assert num_hiddens % num_heads == 0
        self.vocab_size = vocab_size
        self.num_hiddens = num_hiddens
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.num_mlp_intermediate = num_mlp_intermediate
        self.dropout = dropout
