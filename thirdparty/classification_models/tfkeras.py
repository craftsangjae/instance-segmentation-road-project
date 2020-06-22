from tensorflow.python import keras
from .models_factory import ModelsFactory


class TFKerasModelsFactory(ModelsFactory):

    @staticmethod
    def get_kwargs():
        return {
            'backend': keras.backend,
            'layers': keras.layers,
            'models': keras.models,
            'utils': keras.utils,
        }


Classifiers = TFKerasModelsFactory()
