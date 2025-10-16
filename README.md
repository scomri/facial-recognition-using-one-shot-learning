# Facial Recognition Using One-shot Learning


## Overview
A deep learning project about facial recognition using one-shot learning, based on the paper Siamese Neural Networks for One-shot Image Recognition.

## Table of Contents
1. Data Analysis  
2. Project Structure  
3. Model Architecture  
4. Model Initialization  
5. Hyperparameters  
6. Stopping Conditions
7. Experiments  
8. Results and Evaluation  
9. Conclusions  

---

## 1. **Data Analysis**

The raw data received for the assignment were images, which belong to the **Labeled Faces in the Wild (LFW)** dataset.

### **Dataset Description**

* **Images (`data/`):**
  Images are of size **250×250 pixels**, organized by person.
  Each person has one or more images, numbered sequentially starting from `0001`, `0002`, `0003`, and so on.

* **Training and Test Sets:**
  The **train** and **test** series are defined according to `.txt` files.

  **Structure of the `.txt` file:**

  * The **first line** specifies the number of records in the set for each type —
    *matching pairs* (two images of the same person) and *non-matching pairs* (two images of different people).
  * Each record then follows this format:

    * **Matching pair:**

      ```
      person_name   img1_num   img2_num
      ```
    * **Non-matching pair:**

      ```
      person1_name   img1_num   person2_name   img2_num
      ```

### **Preprocessing Steps**

1. **Image Resizing:**
   Each image was resized to **105×105 pixels**.

2. **Channel Adjustment:**
   Added the color channels to all images in the training set.

3. **Matrix Conversion:**
   Transformed the images into matrices containing pixel values.
   This ensured compatibility with the model architecture used later.

### **Dataset Characteristics**

* **Total images:** 13,233
* **Training set:** 2,200 records

  * 1,100 matching image pairs
  * 1,100 non-matching image pairs
* **Test set:** 1,000 records

  * 500 matching image pairs
  * 500 non-matching image pairs
 
### **Examples from the Training Set**

Below are examples of data records from the **training set**:

* **Example of a matching pair (two images of the same person):**

  ```
  Aaron_Peirsol   1   2
  ```

  **Images:**

  
  <img width="125" height="125" alt="image" src="https://github.com/user-attachments/assets/50b80c78-353e-4967-8e34-73864ef06527" />

  <img width="125" height="125" alt="image" src="https://github.com/user-attachments/assets/8e4b335a-777f-4007-8aee-224755084b64" />

  

* **Example of a non-matching pair (two images of different people):**

  ```
  AJ_Cook   1   Benjamin_Martinez   1
  ```

  **Images:**

  <img width="125" height="125" alt="image" src="https://github.com/user-attachments/assets/fd8eaa33-f61f-409f-aece-7ddc71205796" />

  <img width="125" height="125" alt="image" src="https://github.com/user-attachments/assets/48335c48-c299-4889-80bf-607106ea9351" />


---

# **2. Project Structure**

## **Directories**

* **`data/` directory**
  Contains the actual images, the `.txt` files downloaded from the dataset, and the `.pkl` files we generated (which store the preprocessed images).

* **`logs/` directory**
  Added to store runtime logs, printed outputs, and recorded results at important checkpoints.
  This allows us to **analyze past runs** and understand the model’s behavior during training.

* **`results/` directory**
  Contains the **experiment outputs** and **graphs** describing the **training** and **testing** processes.


## **Files**

* **`load_data.py`**
  Responsible for **loading**, **preprocessing**, and **saving** the data.
  Also includes **initial exploratory data analysis**, with preprocessed data stored in `.pkl` format.

* **`logger.py`**
  Initializes the **logging configuration** (`logger`) according to a predefined format.
  The file uses Python’s built-in `logging` module for unified log handling across the project.

* **`requirements.txt`**
  Lists all Python packages used in the project.
  This ensures that anyone who wants to run the project can easily install all required dependencies.

* **`siamese_network.py`**
  Implements the **Siamese neural network** class.
  Includes initialization of **weights**, construction of the **CNN architecture**, and the **forward** pass — as described in the reference paper.

* **`trainer.py`**
  Defines a dedicated **Trainer class** for **training** and **evaluation**.
  Training and experiment execution are separated for clarity and maintainability, ensuring the experiment runner is not overloaded.
  The trainer loads data from the `.pkl` files and performs predictions on the **test set**.

* **`run_experiments.py`**
  Executes the experiments using a **grid search** approach, testing multiple combinations of hyperparameters selected for optimization.


---

# **3. Model Architecture**

The model can be divided into two main components, which connect during the **forward** phase:

1. A **convolutional network** that extracts features from input images.
2. A **prediction network**, which receives the outputs from the convolutional network and computes the similarity between them.

In essence, our implementation uses **a single convolutional network** that is applied **twice** (once per image) during inference to produce two feature vectors.
A **distance** is then calculated between these vectors, and the result is passed as input to the **prediction network**.


## **Convolutional Network Structure**

The convolutional network was logically divided into **five blocks**, each responsible for progressively extracting higher-level features.

### **Block 1**

* **Input:**
  Images of size **105×105 pixels**, grayscale (single channel).
* **Layers:**

  1. Convolutional layer with **64 kernels** of size **10×10**.
     → Output size: **96×96×64**
  2. Batch Normalization layer
  3. ReLU activation
  4. MaxPooling layer
* **Output:**
  **48×48×64**
* **Representation:**

  ```
  Conv → BatchNorm → ReLU → MaxPool2d
  ```

### **Block 2**

* **Input:**
  Output of Block 1 → **48×48×64**
* **Layers:**

  1. Convolutional layer with **128 kernels** of size **7×7**.
     → Output size: **42×42×128**
  2. Batch Normalization
  3. ReLU activation
  4. MaxPooling
* **Output:**
  **21×21×128**
* **Representation:**

  ```
  Conv → BatchNorm → ReLU → MaxPool2d
  ```

### **Block 3**

* **Input:**
  Output of Block 2 → **21×21×128**
* **Layers:**

  1. Convolutional layer with **128 kernels** of size **4×4**.
     → Output size: **18×18×128**
  2. Batch Normalization
  3. ReLU activation
  4. MaxPooling
* **Output:**
  **9×9×128**
* **Representation:**

  ```
  Conv → BatchNorm → ReLU → MaxPool2d
  ```

### **Block 4**

* **Input:**
  Output of Block 3 → **9×9×128**
* **Layers:**

  1. Convolutional layer with **256 kernels** of size **4×4**.
     → Output size: **6×6×256**
  2. Batch Normalization
  3. ReLU activation
     *(No pooling layer in this block)*
* **Output:**
  **6×6×256**
* **Representation:**

  ```
  Conv → BatchNorm → ReLU
  ```

### **Block 5**

* **Input:**
  Output of Block 4 → **6×6×256**
* **Layers:**

  1. Flatten layer → produces a vector of size **9216** (6×6×256).
  2. Fully Connected (Dense) layer → **4096 units**.
  3. Sigmoid activation.
* **Representation:**

  ```
  Flatten → Linear(Dense) → Sigmoid
  ```

This completes the **convolutional subnetwork** that forms the feature extractor of the **Siamese Network**.


## **Prediction Network**

The **prediction network** is simpler and consists of the following layers:

1. **Dropout layer** *(optional)* — to randomly drop features during training (controllable dropout rate).
2. **Fully Connected (Linear) layer** — input size **4096**, output size **1**.
3. **Sigmoid activation** — converts the single output value into a **probability score** representing the similarity between the two input images.


## **Integration within the Siamese Network**

In the model implementation:

* The **convolutional subnetwork** is stored as `self.conv_module`.
* The **prediction network** is stored as `self.prediction_module`.
* The **forward** function combines both to build the complete **Siamese Network** architecture.


## **Forward Function**

* **Input:**
  Two images (`img1`, `img2`).

* **Process:**

  1. Each image passes independently through the convolutional network to produce two feature vectors.
  2. The **L1 distance** between these vectors is computed using:

     ```python
     torch.abs(output1 - output2)
     ```

     (Resulting vector shape: **4096×1**)
  3. This distance vector is passed through the **prediction network**.

* **Output:**
  A **probability value** indicating the **similarity** between the two input images.



**Note:**
All blocks and layers are wrapped using `nn.Sequential` for modularity and cleaner implementation.

---

# **4. Model Initialization**

Model initialization in our implementation focuses on two main aspects:

1. **Setting a random seed**
2. **Initializing network weights**

## **1. Random Seed Setup**

To ensure **reproducibility** of results, we defined a fixed random seed within the model class.
We implemented a helper function called `setup_seeds()` which initializes all relevant **random generators** across the following libraries:

* `random` (Python built-in)
* `numpy.random`
* `torch`

This guarantees consistent behavior and reproducible outcomes across multiple training runs, regardless of system-level randomness.

## **2. Weight Initialization**

For initializing network weights, we used a dedicated function named `setup_weights()`.

This function initializes weights **according to the layer type**:

* **Convolutional layers (Conv):** initialized as recommended in the reference paper (consistent with standard CNN initialization practices).
* **Fully connected (Linear) layers:**
  Slightly modified initialization strategy based on our own experimental observations,
  which yielded **improved model performance** during training.


In summary, the model initialization ensures both **deterministic reproducibility** and **optimized parameter initialization**, forming a consistent foundation for the training process.


---

# **Part 5: Hyperparameters**

This section details both the **fixed** and **variable** hyperparameters used during model training, as well as the reasoning behind their selection.


## **Fixed Hyperparameters**

* **Validation set size:**
  We used a standard validation split of **20%**.
  To allow the model to train on more data, we also tested a **15%** split but found no significant difference in performance.

* **Optimizer:**
  We performed several experiments with different optimizers and chose **Adam**, which consistently produced **better results** compared to **SGD**.

* **Loss function:**
  We used **Binary Cross-Entropy (BCE)**, as recommended in the reference paper.


## **Variable Hyperparameters**

### **Batch Size**

We tested smaller batch sizes — **16** and **32** — instead of the more typical **32** and **64**,
because the training dataset was relatively small (**2,200 records**).
Using smaller batches allowed the model to **perform more updates per epoch**, helping it learn more effectively from limited data.


### **Learning Rate**

We experimented with two learning rates: **0.001** and **0.005**.

To manage learning rate decay, we used the **StepLR scheduler** (from PyTorch), with the following rule: `new_lr = old_lr / 0.95`.

This configuration reduces the learning rate by **5% every 5 epochs**,
allowing for **slower, more stable convergence** as the model approaches the optimum.


### **Regularization (Weight Decay)**

To prevent **overfitting**, we tested small regularization (L2) coefficients: `[0.0, 0.0001, 0.0005]`.

However, since our dataset was small, we aimed to avoid **underfitting** as well.
Hence, we used relatively small values for this parameter.


### **Number of Epochs**

We trained models for **15**, **25**, and **50 epochs**,
seeking a balance between giving the model enough time to learn and avoiding excessive training that might lead to **overfitting**.

Because the training dataset was limited, longer runs increased the risk that the model would begin to memorize rather than generalize,
raising the **overfitting risk** — particularly in later epochs.


### **Dropout Rate**

To further mitigate potential overfitting, we considered using **dropout layers** that randomly drop a portion of the neurons during training.
However, due to the **small dataset size**, overfitting was not a major concern.
Therefore, we only tested **small dropout values**: `[0.0, 0.2]`.


### **Summary of Tested Hyperparameters**

| **Parameter**      | **Values Tested**    |
| ------------------ | -------------------- |
| Validation split   | 0.15, 0.20           |
| Optimizer          | Adam, SGD            |
| Loss Function      | Binary Cross-Entropy |
| Batch Size         | 16, 32               |
| Learning Rate      | 0.001, 0.005         |
| Regularization (λ) | 0.0, 0.0001, 0.0005  |
| Epochs             | 15, 25, 50           |
| Dropout Rate       | 0.0, 0.2             |

---


# **6. Stopping Conditions**

During the training phase, we implemented an **early stopping mechanism** to automatically halt training when validation performance stopped improving.

Within the `Trainer` class, we defined a parameter called **`patience`**,
which determines the **maximum number of consecutive epochs** that can occur **without improvement in validation accuracy** before stopping the training process.

After each epoch:

* The model checks whether **validation accuracy** improved compared to the best previous epoch.
* If an improvement is detected → the **patience counter** is **reset**.
* If no improvement is observed → the **counter decreases by 1**.

When the counter reaches **zero**, training **stops automatically**.

```python
if validation_accuracy_improved(better_than_best_epoch):
    reset_patience_counter()
else:
    patience_counter -= 1

if patience_counter == 0:
    stop_training()
```

Throughout our experiments, we noticed that this stopping condition **sometimes halted training too early**, preventing the model from reaching its **optimal performance**.

This early termination negatively affected the **final accuracy** and **generalization** of the model.
We believe that a **more sophisticated early stopping criterion** could lead to **better results** while still reducing unnecessary training time.

---

# **7. Experiments**

Throughout the project, we designed our codebase to be as **modular** as possible, allowing us to test and compare a wide range of **hyperparameter configurations** efficiently.

In practice, due to the **large number of possible hyperparameters**, we did not test every possible combination.
If each parameter had **three values to test**, the total number of potential combinations would exceed **400**.

Limited computational capacity (personal machine GPU) made it impractical to run all combinations within reasonable time frames.


## **Initial Experiment Phase**

As a first step, we conducted a **general experiment** to establish a baseline.
This involved selecting hyperparameters **based on prior exploratory runs and observations** conducted earlier in the project.

These initial experiments provided insights into **which parameters had the greatest influence on performance**, and helped narrow down the **search space** for more focused testing in subsequent stages.

<img width="1089" height="698" alt="image" src="https://github.com/user-attachments/assets/71b3f260-5b1d-4eea-8590-e2863ded3cc0" />

## **Focused Experimentation**

Following the initial general experiment, we conducted a series of **more targeted experiments**.
Before doing so, we **narrowed down and fixed several hyperparameters** based on the first round of results:

* **Learning rate:**
  Based on prior outcomes, we fixed the learning rate at **0.005**, as it consistently yielded better performance.

* **Number of epochs:**
  We limited the range of tested values to **[15, 25, 50]**.
  We hypothesized that **15 epochs were insufficient** for the model to learn effectively,
  while higher values might provide a better balance between training time and model accuracy.


## **Regularization Experiment**

At this stage, we focused on identifying the **optimal regularization coefficient (λ)**.
We ran experiments with the following values: `[0.0, 0.0001, 0.0005]`.

The goal was to determine how **L2 regularization** (weight decay) affects generalization performance, and to find the best balance between **overfitting prevention** and **training stability**.

<img width="1368" height="266" alt="image" src="https://github.com/user-attachments/assets/96eabad5-b318-41f6-a7cf-03b38d85a7a7" />


## **Selection of Best Experiments**

The **best results** were obtained from **Experiments 1 and 6**.
We therefore decided to **retain the hyperparameter values** from these experiments for further analysis, and later expand on the **performance of the resulting models**.

At this stage, we decided to **set aside Experiment 6** and its parameters, since its results were very similar to those of Experiments 1 and 2, but the latter showed **more stable performance** (the only difference being the number of training epochs).

As a result, we **fixed the regularization coefficient (λ)** to a value of **0**.


## **Further Parameter Exploration**

Next, we investigated two additional parameters: **batch size** and **dropout rate**.
The following values were tested:

```
batch_size   = [16, 32]
dropout_rate = [0.0, 0.2]
```

Because the previous experiments with **25 and 50 epochs** produced **similar results**, we chose to conduct this next round of experiments using **25 epochs only**.

<img width="1360" height="244" alt="image" src="https://github.com/user-attachments/assets/254e9841-4b77-4af5-8df7-14beaae838eb" />

## **Findings from the Dropout and Batch Size Experiment**

From this experiment, we observed that applying a **dropout rate of 0.2** resulted in **better performance**, particularly in **Experiments 2 and 4**.

Interestingly, the **best overall results** in this round were achieved with a **batch size of 16** (Experiment #2).
This was somewhat **unexpected**, as in earlier tests (before the first general experiment), a batch size of **32** appeared to produce better results.
However, in this case, the smaller batch size led to **improved accuracy and stability**.

Therefore, we decided to **retain the hyperparameter configuration of Experiment #2**.

---

# **8. Results and Evaluation**

This section presents the **four best-performing models**, ranked according to their **Test Accuracy** results.


## **Model 1 — Test Accuracy = 0.743**

### **Model Parameters**

| **Parameter**           | **Value**      |
| ----------------------- | -------------- |
| **Epochs**              | 50             |
| **Batch Size**          | 32             |
| **Learning Rate**       | 0.005          |
| **Regularization (λ)**  | 0.0005         |
| **Dropout Rate**        | 0              |
| **Training Time**       | ≈ 13.2 minutes |
| **Training Loss**       | 0.418          |
| **Validation Loss**     | 0.571          |
| **Validation Accuracy** | 0.709          |
| **Testing Time**        | 8.56 seconds   |
| **Test Loss**           | 25.7           |


### **Model Performance**

<img width="420.3" height="315.67" alt="image" src="https://github.com/user-attachments/assets/7b33d8ca-a4dc-41d7-9705-bebc07da9ca4" />

The model demonstrated a significant improvement in performance after just a few epochs (fewer than five). Following this initial gain, it converged but continued to enhance its performance throughout the remaining epochs of the training run.

<img width="420.3" height="315.67" alt="image" src="https://github.com/user-attachments/assets/fa8d04e1-76f1-45e8-9463-de6d3e48a423" />

From the graph, it is clear that only after approximately 30 epochs does the model begin to generalize effectively, maintaining an accuracy of around 0.7 on the validation set.

<img width="420.3" height="315.67" alt="image" src="https://github.com/user-attachments/assets/df3fcebc-d555-4bcd-8861-d557ee7420ce" />

The previous graph indicates that the loss value reaches a state of convergence after 30 epochs. This trend is observed as accuracy improves; as the accuracy increases, the loss value correspondingly decreases. This relationship is clearly illustrated in the graph above.

---

## **Model 2 — Test Accuracy = 0.732**

### **Model Parameters**

| **Parameter**           | **Value**      |
| ----------------------- | -------------- |
| **Epochs**              | 25             |
| **Batch Size**          | 32             |
| **Learning Rate**       | 0.005          |
| **Regularization (λ)**  | 0              |
| **Dropout Rate**        | 0              |
| **Training Time**       | ≈ 6.43 minutes |
| **Training Loss**       | 0.565          |
| **Validation Loss**     | 0.586          |
| **Validation Accuracy** | 0.699          |
| **Testing Time**        | 8.77 seconds   |
| **Test Loss**           | 26.8           |

### **Model Performance**

<img width="420.3" height="315.67" alt="image" src="https://github.com/user-attachments/assets/e29e4901-4900-4bcd-8d2f-1333bee70463" />

The model showed a **significant improvement in performance** after only a **few epochs**—specifically around **epoch #3 to #5**.
From that point onward, it demonstrated **steady convergence**, continuing to **improve gradually throughout the remaining training epochs**. This indicates that the model reached **stable learning dynamics early on** and maintained consistent progress until the end of training.

<img width="420.3" height="315.67" alt="image" src="https://github.com/user-attachments/assets/9adbfd38-0b93-4c6b-a0a9-64cc0440c04e" />

From the graph, it can be observed that **only after nearly 17–20 epochs** the model begins to **generalize effectively** — as reflected by a **significant reduction in fluctuations** in the validation accuracy curve.
From this stage onward, the model maintains **stable validation accuracy values** around **0.7**, indicating consistent and reliable generalization performance.

<img width="420.3" height="315.67" alt="image" src="https://github.com/user-attachments/assets/916e479a-06e5-4178-956d-94b94f2c1552" />

In alignment with the previous graph, we can see that around **epoch #17**, the **loss curve** also becomes smoother and **less volatile**.
At this point, the model appears to **converge**, and as the **accuracy increases**, the **loss decreases** correspondingly — demonstrating a clear **inverse relationship** between these two metrics as training progresses.


## **Model 3 — Test Accuracy = 0.739**

### **Model Parameters**

| **Parameter**           | **Value**      |
| ----------------------- | -------------- |
| **Epochs**              | 50             |
| **Batch Size**          | 32             |
| **Learning Rate**       | 0.005          |
| **Regularization (λ)**  | 0              |
| **Dropout Rate**        | 0              |
| **Training Time**       | ≈ 13.1 minutes |
| **Training Loss**       | 0.480          |
| **Validation Loss**     | 0.577          |
| **Validation Accuracy** | 0.703          |
| **Testing Time**        | 8.61 seconds   |
| **Test Loss**           | 26.1           |

### **Model Performance**

<img width="420.3" height="315.67" alt="image" src="https://github.com/user-attachments/assets/bfb6de2c-167c-42f2-b789-2ea731a8a55a" />

The model achieved **significant convergence in training performance** after approximately **5 epochs**, and continued to **improve steadily** throughout the remainder of the training process until completion.

<img width="420.3" height="315.67" alt="image" src="https://github.com/user-attachments/assets/ffac9acf-b069-47f6-99ef-f77cc4d0fd28" />

According to the graph, the model begins to **generalize effectively** only after approximately **20 epochs**. From that point onward, the **validation accuracy** stabilizes and remains consistently **around 0.7**, indicating solid and reliable generalization performance.

<img width="420.3" height="315.67" alt="image" src="https://github.com/user-attachments/assets/b280cfdc-285e-4824-94c9-4a757a8fd3b9" />

In accordance with the previous graph, as the **accuracy increases and improves**, the **loss value decreases**, reflecting the model’s effective learning progress — a clear indication of **inverse correlation** between the two metrics throughout training.


## **Model 4 — Test Accuracy = 0.749**

### **Model Parameters**

| **Parameter**           | **Value**      |
| ----------------------- | -------------- |
| **Epochs**              | 25             |
| **Batch Size**          | 16             |
| **Learning Rate**       | 0.005          |
| **Regularization (λ)**  | 0              |
| **Dropout Rate**        | 0.2            |
| **Training Time**       | ≈ 8.18 minutes |
| **Training Loss**       | 0.578          |
| **Validation Loss**     | 0.585          |
| **Validation Accuracy** | 0.725          |
| **Testing Time**        | 8.84 seconds   |
| **Test Loss**           | 25.1           |

### **Model Performance**

<img width="420.3" height="315.67" alt="image" src="https://github.com/user-attachments/assets/01ca4b07-3d7c-4075-ad91-18861e6139f7" />

According to the graph, the model begins to achieve **strong performance after approximately 3–4 epochs**, and continues to **improve at a slower, more gradual pace** thereafter. Significant convergence occurs around **epoch #12**, and the model maintains **steady improvement** until the end of the training process.

<img width="420.3" height="315.67" alt="image" src="https://github.com/user-attachments/assets/3e1dbdac-742e-47be-abbd-4e0dd55ea7e9" />

Interestingly, there is a noticeable **drop in validation accuracy around epoch #14**, but the model quickly recovers and continues to show **consistent progress** throughout training. Starting around **epoch #20**, the performance fluctuations decrease, indicating **model convergence and stable generalization** over the validation data.

<img width="420.3" height="315.67" alt="image" src="https://github.com/user-attachments/assets/22342f4f-1a2a-4bf5-8e56-b0ba19b97fc9" />

### **Loss–Accuracy Relationship**

Consistent with the previous graph, the model exhibits some **fluctuations during training**, but these **decrease significantly after approximately epoch #14**.
As the **accuracy increases and improves**, the **loss correspondingly decreases**, demonstrating a clear **inverse relationship** between the two metrics, as reflected in the graph.


### **Runtime Comparison**

In addition, we performed a **comparison of the training runtimes** across all models. (This comparison **did not include testing runtimes**, as those were extremely short and therefore not meaningful for evaluation.)

<img width="421.5" height="341.25" alt="image" src="https://github.com/user-attachments/assets/0778fd45-fe6f-43d8-a9a5-6c637ebd25e0" />

As expected, **Models 1 and 3**, which were trained for **50 epochs**, required approximately **twice the training time** compared to **Models 2 and 4**, which were trained for **25 epochs**.

Interestingly, **Model 4** took slightly longer to train than **Model 2**, despite having the same number of epochs.
We hypothesize that this difference is due to the **smaller batch size** in Model 4 (**16**, half the size of Model 2’s batch), which effectively **doubles the number of batches per epoch** and therefore increases total training time.
Additionally, the **dropout rate** in Model 4 (0.2) may have also contributed slightly to longer training durations.


So far, the results indicate that **Model 4** achieved the **best overall performance**.
Therefore, we conducted additional experiments on this model to **examine the impact of specific parameters** and explore whether further performance improvements could be achieved.


## **Effect of Batch Size**

<img width="480" height="360" alt="image" src="https://github.com/user-attachments/assets/9203d7ff-f799-4642-a53c-2f1257edf421" />

We fixed all other hyperparameters and varied only the **batch size**: `[8, 16, 32, 64]`.
We initially hypothesized that **too small a batch size** would lead to **poorer performance**, as frequent weight updates based on small amounts of data might introduce excessive noise and instability.
Indeed, when training with a **batch size of 8**, the model achieved **reasonable but suboptimal results**.
Surprisingly, the **best performance** was achieved with a **batch size of 64**, contrary to previous experiments conducted earlier in the assignment, where larger batch sizes did **not yield good results**.


## **Effect of Learning Rate**

<img width="480" height="360" alt="image" src="https://github.com/user-attachments/assets/3b5a67a1-94cd-45c4-8876-3bc13dfdc508" />

Next, we tested four different values for the **learning rate**, spanning **two orders of magnitude**: `[0.0001, 0.0005, 0.001, 0.005]`.
From this experiment, it was observed that learning rates on the order of $10^{-2}$ produced **better results** than those on the order of $10^{-3}$.
We infer that **too small a learning rate** causes the model to take **very small optimization steps**, making it **difficult to reach the optimum** efficiently within a reasonable number of epochs.


## **Effect of the Number of Epochs**

<img width="480" height="360" alt="image" src="https://github.com/user-attachments/assets/51710128-83e4-4a6e-8610-cb5c012ad9d6" />

In this experiment, we examined how the **number of training epochs** affected the model’s accuracy: `[5, 15, 25, 50]`.
We hypothesized that **increasing the number of epochs** would improve the model’s performance — and indeed, we observed better results as the number of epochs increased.
However, when comparing **50 epochs** to **25 epochs**, the improvement was **minimal**, even though the training time was **doubled**. This suggests that the model had already **approached convergence** by epoch 25.
We assume that extending the training further — for example, to **75 or 100 epochs** — would likely lead to **overfitting**, causing the model’s performance to **decrease significantly** as it would fail to generalize effectively to unseen data.


## **Model Selection**

After testing additional parameters on **Model 4**, we concluded that it was **not sufficiently stable**.
Repeated runs of this configuration produced **inconsistent results**, suggesting that the previously observed high accuracy may have been **partly due to randomness**.

Therefore, we decided to select **Model 1** as our **final chosen model**.
Although its **test accuracy (0.743)** was slightly lower than that of Model 4 (**0.749**), Model 1 demonstrated **greater stability and consistency** across multiple training runs. Moreover, the difference in accuracy between the two models was **negligible**.

To better illustrate the model’s predictive performance, we generated a **confusion matrix** for **Model 1**, which we identified as the **optimal and most reliable configuration**.

### **Model 1 Parameters**

| **Parameter**          | **Value** |
| ---------------------- | --------- |
| **Epochs**             | 25        |
| **Batch Size**         | 32        |
| **Learning Rate**      | 0.005     |
| **Regularization (λ)** | 0         |
| **Dropout Rate**       | 0         |

### **Confusion Matrix**

<img width="404" height="343.5" alt="image" src="https://github.com/user-attachments/assets/df06f064-dccf-482f-bf44-65fe044d600f" />

---

In addition to the model evaluations, we present **examples of image pairs** that were **correctly and incorrectly classified** by the model.

### **Example — Correctly Classified Similar Images**

**Pair:** `Chris_Bell_0001` and `Chris_Bell_0002`

<img width="125" height="125" alt="image" src="https://github.com/user-attachments/assets/054adf92-70b6-46bc-a7a1-3b43b8b1e3b1" />
<img width="125" height="125" alt="image" src="https://github.com/user-attachments/assets/323496fb-3571-4a31-ad8a-7be7f32c42eb" />

The model successfully identified these two images as belonging to the **same person**.
We assume the high classification confidence resulted from the **strong visual similarity** between the images: the individual is wearing **identical clothing** in both photos, and the presence of the **U.S. flag on the left side** — a **rare and distinctive feature** — likely served as a strong visual cue for the model’s decision.


#### **Example — Incorrectly Classified as Similar**

**Pair:** `Don_Carcieri_0001` and `Jane_Rooney_0001`

<img width="125" height="125" alt="image" src="https://github.com/user-attachments/assets/006c44d6-c0c8-4289-a264-03c0fa49833d" />
<img width="125" height="125" alt="image" src="https://github.com/user-attachments/assets/52581983-08fc-4951-b1d9-cc41e3ed92b6" />

In this case, the model **incorrectly classified** the two images as belonging to the **same person** with a **high similarity score**.
This misclassification can likely be attributed to the **facial expressions**, which appear **very similar** across both images.
Additionally, although one image depicts a **woman**, her **short hair** resembles that of the **man** in the other photo, potentially confusing the model’s feature extractor.
Moreover, the image of the woman contains **significant visual noise** — specifically, a **partially visible face** of another person in the **bottom-left corner**, which may have further disrupted the model’s prediction.


#### **Example — Correctly Classified as Non-Matching**

**Pair:** `Bing_Crosby_0001` and `John_Moxley_0001`

<img width="125" height="125" alt="image" src="https://github.com/user-attachments/assets/f2e01e6d-b76d-4cac-ad75-86a089746842" />
<img width="125" height="125" alt="image" src="https://github.com/user-attachments/assets/e9578cbb-1929-4913-8197-cb332f0f6f42" />

The model correctly classified these two images as **non-matching** with **high accuracy**.
Visually, the two individuals are indeed **completely different**, and the **distinctive hat** worn by the person on the right likely served as a **strong distinguishing feature**, helping the model confidently separate the two faces.


#### **Example — Incorrectly Classified as Non-Matching**

**Pair:** `Janica_Kostelic_0001` and `Janica_Kostelic_0002`

<img width="125" height="125" alt="image" src="https://github.com/user-attachments/assets/05cb051d-865f-4671-a977-a5a9096dcdb4" />
<img width="125" height="125" alt="image" src="https://github.com/user-attachments/assets/74496081-e7f5-40f1-8fc1-04da3a5705ee" />

In this case, the model **incorrectly classified** the two images as **non-matching**, even though they actually depict the **same person**.
The likely causes for this error include the **noticeable differences in headwear** — in one image, the individual is wearing a **hat**, while in the other, a **headband**.
Additionally, there are clear differences in **image quality and lighting conditions**, and it is possible that the two photos were taken **years apart**, leading to visible **age-related changes** that made the model’s task more difficult.

---


# **9. Conclusions**

1. **Overall Findings:** The task we addressed was **highly complex**, and consequently, the results obtained were **not fully satisfactory** —    none of our models surpassed a **0.75 accuracy** level.
   While multiple factors may have contributed to this limitation, we believe the **primary cause** was the **small amount of training data**, which significantly hindered the model’s ability to generalize effectively.

2. **Number of Epochs:** Based on our experiments, we recommend using **25 epochs** during training, unless the user has access to **significantly stronger computational resources**.
   In such cases, increasing the number of epochs to **50 or more** may lead to **slightly better results**.
   However, we observed that increasing the number of epochs **beyond 25** did **not yield substantial improvements** in accuracy.

3. **Batch Size Recommendation:** After extensive testing, we chose to recommend a **batch size of 32**, even though a batch size of **16** produced marginally higher accuracy.
   This decision was made to prioritize **stability and reproducibility** over minor accuracy gains.
   The reasoning is that each **batch update** affects the model weights:
   * Smaller batches lead to **frequent, noisy updates**, which may make convergence unstable.
   * Larger batches reduce noise but also **limit the number of weight updates**, possibly “skipping over” the global optimum. Thus, **batch size = 32** provided the best **balance between stability and learning efficiency**.

4. **Learning Rate Scheduling:** We found that the **learning rate** can be relatively small, but **not too small** — the model needs to take **larger steps** at the beginning of training and **smaller steps** as it approaches the optimum.
   We successfully implemented this principle using a **learning rate scheduler**, which reduced the learning rate every **5 epochs**, allowing for **gradual convergence** near the optimal point.

5. **Regularization (Weight Decay):** We recommend keeping the **regularization coefficient (λ)** at **0 or a very small value**.
   Increasing λ too much caused the model to behave **randomly**, leading to unstable learning and degraded performance.
   Given the model’s inherent complexity, adding further regularization only **increased the training difficulty** and **led to failure to converge**.

6. **Dropout:** For similar reasons, we concluded that it is **better not to use dropout** in this task.
   Since the model is already learning a **complex classification problem**, adding dropout only made learning harder, causing the network to **over-generalize** and, in some cases, **fail to learn meaningful representations**.

7. **Future Work and Preprocessing Improvements:** Due to time constraints, we did not perform additional image preprocessing steps such as **cropping**, **rotation**, or **augmentation**.
   Such preprocessing techniques could likely have **improved model performance further** by increasing data diversity.
   If this research were to continue, we recommend beginning with this step — implementing **advanced data augmentation** as a first direction for improvement.

