import tensorflow as tf
import numpy as np

# Example data
y_true = np.array([[1, 2, 3, 0], [2, 4, 0, 0]])  # Batch size of 1, sequence length of 4
y_pred_logits = np.random.rand(2, 4, 5)  # Random logits for illustration
print(y_pred_logits)

# Convert to TensorFlow tensors
y_true_tf = tf.constant(y_true, dtype=tf.int64)
y_pred_logits_tf = tf.constant(y_pred_logits, dtype=tf.float32)

# Define the masked_loss function
def masked_loss(y_true, y_pred):
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')
    loss = loss_fn(y_true, y_pred)
    print("Loss: ", loss)

    # Check which elements of y_true are padding
    mask = tf.cast(y_true != 0, loss.dtype)
    print("Mask: ", mask)

    loss *= mask
    print("Loss mul mask: ", loss)
    # Return the total.

    print("tf reduce sum (loss): ", tf.reduce_sum(loss))
    print("tf reduce sum (mask): ", tf.reduce_sum(mask))
    return tf.reduce_sum(loss) / tf.reduce_sum(mask)

# Calculate loss using the masked_loss function
loss = masked_loss(y_true_tf, y_pred_logits_tf)
