from abc import abstractmethod

class BaseModel:
    """
    __modelname__ = None
    __changes__ = None
    generator_optimizer = None
    discriminator_optimizer = None
    __normalization__ = None
    __jitter__ = None
    """
    @property
    @abstractmethod
    def __name__(self):
        raise NotImplementedError("Abstract property '__name__' is not set <unknown>")
    
    @property
    @abstractmethod
    def __changes__(self):
        raise NotImplementedError("Abstract property '__changes__' is not set <unknown>")
    
    @property
    def __model__(self):
        return [self.generator, self.discriminator]
    
    @property
    def __loss__(self):
        return {'g_loss_fn':self.g_loss, 'd_loss_fn':self.d_loss}
    
    @property
    @abstractmethod
    def g_optimizer(self):
        raise NotImplementedError("Abstract property 'g_optimizer' is not set <unknown>")
    
    @property
    @abstractmethod
    def d_optimizer(self):
        raise NotImplementedError("Abstract property 'd_optimizer' is not set <unknown>")
    
    @property
    @abstractmethod
    def __norm__(self):
        raise NotImplementedError("Abstract property '__norm__' is not set <unknown>")
    
    @property
    @abstractmethod
    def __jitter__(self):
        raise NotImplementedError("Abstract property '__jitter__' is not set <unknown>")

    @abstractmethod
    def discriminator(self):
        raise NotImplementedError("Abstract method 'discriminator' is not implemented")
    
    @abstractmethod
    def generator(self):
        raise NotImplementedError("Abstract method 'generator' is not implemented")
    
    @abstractmethod
    def g_loss(self, disc_generated_output, gen_output, target):
            raise NotImplementedError("Abstract method loss is not implemented")

    @abstractmethod
    def d_loss(self, disc_real_output, disc_generated_output):
        raise NotImplementedError("Discriminator loss is not implemented")
    
    @abstractmethod
    def __init__(self, image_shape):
        self.image_shape = image_shape
    
    