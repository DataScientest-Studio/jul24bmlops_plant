import os
import tensorflow as tf

# Navigate three directories up and then to the 'models' folder
model_path = os.path.join(os.path.dirname(__file__), '../../../models/TL_180px_32b_20e_model.keras')

model = tf.keras.models.load_model(model_path)
