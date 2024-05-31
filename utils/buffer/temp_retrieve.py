from utils.buffer.buffer_utils import temp_retrieve


class Temp_retrieve(object):
    def __init__(self, params):
        super().__init__()
        self.num_retrieve = params.eps_mem_batch

    def retrieve(self, buffer, **kwargs):
        return temp_retrieve(buffer, self.num_retrieve)
