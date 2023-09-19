class Config:
    def __init__(
            self,
            vocab_size: int,
            num_hiddens: int,
            num_layers: int,
            num_heads: int,
            num_mlp_intermediate: int|None=None,
            max_context_length: int=1024,
            pad_token_id: int=0,
            bos_token_id: int=1,
            eos_token_id: int=2,
            dropout: float=0.01
        ):
        assert num_hiddens % num_heads == 0
        self.vocab_size = vocab_size
        self.num_hiddens = num_hiddens
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.num_mlp_intermediate = num_mlp_intermediate if num_mlp_intermediate is not None else 4*num_hiddens
        self.max_context_length = max_context_length
        self.pad_token_id = pad_token_id
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.dropout = dropout
