from abc import abstractmethod

class BaseTrain:  
    @abstractmethod
    def special_compile(self):
        raise NotImplementedError()
    
    @abstractmethod
    def compile(self, **kwargs):
        raise NotImplementedError("Please use special_compile()")
    
    @abstractmethod
    def train_step(self, data): 
        raise NotImplementedError()
        
    @abstractmethod
    def test_step(self, data):
        raise NotImplementedError()
    
    @abstractmethod
    def call(self, first_frame, training=False):
        raise NotImplementedError()
    
    @staticmethod
    @abstractmethod
    def start_train():
        raise NotImplementedError()
        
