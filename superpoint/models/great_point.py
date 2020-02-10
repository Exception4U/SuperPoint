"""
A more superpoint like resembline network with rotation 
cnsciousness between two views.
"""
import tensorflow as tf
from .base_model import BaseModel, Mode

class GreatPoint(BaseModel):
    input_spec = {
        'image:'{'shape':[None,None,None,1], 'type': tf.float32}
        }
    required_config_keys = []
    default_config = {
            'data_format': 'channels_first',
            'grid_size': 8,
            'detection_threshold': 0.4,
            'descriptor_size': 256,
            'batch_size': 32,
            'learning_rate': 0.001,
            'lambda_d': 250,
            'descriptor_size': 256,
            'positive_margin': 1,
            'negative_margin': 0.2,
            'lambda_loss': 0.0001,
            'nms': 0,
            'top_k': 0,
    }
    def _model():
        """
        implements model inherited from abstract class baseModel
        """
        config['training'] = (mode == Mode.TRAIN)

        def net(image):
            """
            net architecture
            """
            if config['data_format'] == 'channels_first':
                image = tf.transpose(image, [0, 3, 1, 2])
            features = vgg_backbone(image, **config)
            detections = utils.detector_head(features, **config)
            descriptors = utils.descriptor_head(features, **config)
            return {**detections, **descriptors}

        results = net(inputs['image'])

        if config['training']:
            warped_results = net(inputs['wapred']['image'])
            results = {**results,'warped_results': warped_results,
            'homography':inputs['warped']['homography']}

    def _loss():
        """
        defn of loss function
        loss = (detector_loss + warped_detector_loss
                + config['lambda_loss'] * descriptor_loss)
        where descriptor_loss = [R|t] * [R|t]]   
        """
        
        return "Not yet Implemented"

    def _metrices():
        """
        metrices
        """
        
        return "Not yet Implemented"

